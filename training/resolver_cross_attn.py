#!/usr/bin/env python3
"""
Cross-Object Attention Resolver v2: 跨 object 注意力 + 完整多轮交互训练。

架构创新（vs 之前所有 resolver）：
  所有 objects 的 32ch decoder features 在每个空间位置做 cross-attention，
  让模型在仲裁时看到所有竞争者的 features，而非仅看自己的。

训练机制（继承 resolver_click.py 的全部 tricks）：
  1. Click-based 交互（匹配评估分布）
  2. 多轮交互（默认 2 rounds），每轮都产生训练样本
  3. Click 累积 + 0.9 衰减
  4. 基于模型预测误差的 follow-up click
  5. prev_pred 跨轮传递
  6. Click distance maps (2ch)
  7. Image intensity (1ch)
  8. prev_pred channel (1ch)
  9. Round index (1ch)

输入: 40ch = 32ch decoder features + 3ch competition + 2ch click_dist
             + 1ch image + 1ch prev_pred + 1ch round_idx

用法:
  conda activate nnInteractive
  # 快速验证
  python -u training/resolver_cross_attn.py --max_files 50 --epochs 5 --device cuda:0
  # 完整训练
  python -u training/resolver_cross_attn.py --max_files 200 --epochs 30 --device cuda:0
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.ndimage import distance_transform_edt as _edt_fn

import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.trainer import build_network, autocast_ctx
from training.interaction_sim import InteractionManager, generate_point_blob, \
    sample_point_from_error_region, POINT_RADIUS
from training.run_resolver import MultiObjectDataset

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/" \
               "nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "resolver_cross_attn"

MAX_OBJECTS = 4
CONTESTED_WEIGHT = 10.0
IN_CH = 40  # 32 feat + 3 competition + 2 click_dist + 1 image + 1 prev_pred + 1 round_idx


# ─── Click Distance Map (from resolver_click.py) ────────────────────────────

def compute_click_distance_maps(shape, fg_clicks, bg_clicks):
    """Compute normalized distance to nearest fg/bg click. Returns 2 channels."""
    def _dist_to_clicks(clicks):
        if not clicks:
            return np.ones(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        for c in clicks:
            z, y, x = int(c[0]), int(c[1]), int(c[2])
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                mask[z, y, x] = 1
        if not mask.any():
            return np.ones(shape, dtype=np.float32)
        d = _edt_fn(1 - mask).astype(np.float32)
        return d / max(d.max(), 1.0)

    return _dist_to_clicks(fg_clicks), _dist_to_clicks(bg_clicks)


# ─── Model ──────────────────────────────────────────────────────────────────


class CrossObjectResolver(nn.Module):
    """
    Cross-object attention resolver (40ch).

    每个 object 的输入经过：
      1) 共享 conv encoder → per-object 空间编码
      2) Per-position cross-object self-attention → objects 互相看到对方的 features
      3) Post-attention conv → 把竞争决策传播到空间邻居
      4) Head → 残差 delta（零初始化）
    """

    def __init__(self, in_ch=IN_CH, hidden=32, num_heads=4):
        super().__init__()
        self.hidden = hidden

        # Phase 1: per-object spatial encoder (shared)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
        )

        # Phase 2: cross-object attention
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn_ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden * 2, hidden),
        )
        self.ff_norm = nn.LayerNorm(hidden)

        # Phase 3: post-attention spatial conv
        self.post_attn = nn.Sequential(
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
        )

        # Phase 4: output head (zero-init → residual starts at 0)
        self.head = nn.Conv3d(hidden, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_list, logit_list):
        """
        Args:
            input_list:  K × (1, 40, D, H, W) — per-object full input
            logit_list:  K × (D, H, W)        — per-object margin logits (for residual)
        Returns:
            K × (D, H, W) refined logits
        """
        K = len(input_list)

        # Phase 1: encode each object (shared weights)
        encoded = [self.encoder(input_list[k]) for k in range(K)]

        # Phase 2: cross-object attention at each spatial position
        if K > 1:
            D, H, W = encoded[0].shape[2:]
            N = D * H * W

            # (N, K, hidden) — batch=spatial positions, seq=objects
            stacked = torch.stack(
                [e[0].reshape(self.hidden, N).T for e in encoded], dim=1
            )

            # Process in chunks to control memory
            chunk_size = min(N, 100000)
            out_chunks = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = stacked[start:end]
                attn_out, _ = self.cross_attn(chunk, chunk, chunk)
                chunk = self.attn_norm(chunk + attn_out)
                chunk = self.ff_norm(chunk + self.attn_ff(chunk))
                out_chunks.append(chunk)
            stacked = torch.cat(out_chunks, dim=0)

            for k in range(K):
                encoded[k] = stacked[:, k, :].T.reshape(
                    1, self.hidden, D, H, W)

        # Phase 3 + 4: post-attention + head
        refined = []
        for k in range(K):
            post = self.post_attn(encoded[k])
            delta = self.head(post)[0, 0]  # (D, H, W)
            refined.append(logit_list[k] + delta)

        return refined


# ─── Training ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rounds", type=int, default=2,
                        help="Interaction rounds per step (2=initial + 1 follow-up)")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Backbone (frozen) + feature hook ──
    backbone, _ = build_network(args.checkpoint, deep_supervision=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    captured_features = {}

    def hook_fn(module, input, output):
        captured_features['feat'] = output.detach()

    handle = backbone.decoder.stages[-1].register_forward_hook(hook_fn)

    # ── Resolver ──
    resolver = CrossObjectResolver(
        in_ch=IN_CH, hidden=32, num_heads=4).to(device)
    n_params = sum(p.numel() for p in resolver.parameters())
    print(f"CrossObjectResolver {IN_CH}ch: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(
        resolver.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Dataset ──
    dataset = MultiObjectDataset(
        data_dir=args.data_dir, max_files=args.max_files, augment=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    print(f"\nTraining: {args.epochs} epochs, {len(dataset)} files, "
          f"{args.rounds} rounds/step, contested_weight={CONTESTED_WEIGHT}",
          flush=True)

    BG_LOGIT = 0.0
    best_cacc = 0.0

    for epoch in range(args.epochs):
        resolver.train()
        losses, accs, caccs = [], [], []
        n_steps = 0
        t0 = time.time()

        for batch in loader:
            image = batch['image'].to(device).unsqueeze(0)  # (1, 1, D, H, W)
            gt_np = batch['gt'].numpy()
            image_np = batch['image'].numpy().squeeze(0)  # (D, H, W) z-scored
            oids = batch['object_ids']
            if isinstance(oids, torch.Tensor):
                oids = oids.tolist()
            if len(oids) < 2:
                continue

            K = min(len(oids), MAX_OBJECTS)
            if len(oids) > K:
                oids = random.sample(oids, K)

            spatial = tuple(image.shape[2:])

            # ── Multi-round interaction (from resolver_click.py) ──
            obj_clicks = {oid: {'fg': [], 'bg': []} for oid in oids}
            prev_preds = {oid: np.zeros(spatial, dtype=np.uint8)
                          for oid in oids}

            # Collect (fg_logits, feat_list, click_info, round_idx) per round
            round_samples = []

            for round_idx in range(args.rounds):
                fg_logits = []
                feat_list = []
                click_info = []

                for oid in oids:
                    gt_binary = (gt_np == oid).astype(np.uint8)
                    mgr = InteractionManager(spatial)

                    # prev_pred from last round
                    if round_idx > 0:
                        mgr.set_prev_pred(prev_preds[oid])

                    if round_idx == 0:
                        # Initial click from GT center
                        mgr.set_initial_point(gt_binary, is_fg=True)
                        coords = np.argwhere(gt_binary > 0)
                        if len(coords) > 0:
                            dt = _edt_fn(gt_binary)
                            center = tuple(np.unravel_index(
                                dt.argmax(), dt.shape))
                            obj_clicks[oid]['fg'].append(center)
                    else:
                        # Restore previous clicks with decay
                        for c in obj_clicks[oid]['fg']:
                            blob = generate_point_blob(
                                spatial, c, POINT_RADIUS)
                            decay = 0.9 ** round_idx
                            mgr.interactions[3] = np.maximum(
                                mgr.interactions[3], blob * decay)
                        for c in obj_clicks[oid]['bg']:
                            blob = generate_point_blob(
                                spatial, c, POINT_RADIUS)
                            decay = 0.9 ** round_idx
                            mgr.interactions[4] = np.maximum(
                                mgr.interactions[4], blob * decay)

                        # Follow-up click from prediction error
                        fn_region = (gt_binary > 0) & \
                                    (prev_preds[oid] == 0)
                        fp_region = (gt_binary == 0) & \
                                    (prev_preds[oid] > 0)

                        if fn_region.sum() > fp_region.sum():
                            center = sample_point_from_error_region(fn_region)
                            if center:
                                obj_clicks[oid]['fg'].append(center)
                                blob = generate_point_blob(
                                    spatial, center, POINT_RADIUS)
                                mgr.interactions[3] = np.maximum(
                                    mgr.interactions[3], blob)
                        elif fp_region.sum() > 0:
                            center = sample_point_from_error_region(fp_region)
                            if center:
                                obj_clicks[oid]['bg'].append(center)
                                blob = generate_point_blob(
                                    spatial, center, POINT_RADIUS)
                                mgr.interactions[4] = np.maximum(
                                    mgr.interactions[4], blob)

                    inter = torch.from_numpy(
                        mgr.get_numpy()).unsqueeze(0).to(device)
                    input_8ch = torch.cat([image, inter], dim=1)

                    with torch.no_grad(), autocast_ctx():
                        output = backbone(input_8ch)

                    fg = (output[0, 1] - output[0, 0]).float().cpu()
                    fg_logits.append(fg)
                    feat_list.append(
                        captured_features['feat'][0].float().cpu())
                    del input_8ch, output, inter

                    # Update prev_pred for next round
                    prev_preds[oid] = (fg.numpy() > 0).astype(np.uint8)

                    click_info.append({
                        'fg': list(obj_clicks[oid]['fg']),
                        'bg': list(obj_clicks[oid]['bg']),
                    })

                round_samples.append(
                    (fg_logits, feat_list, click_info, round_idx))

            # ── Train resolver on ALL rounds ──
            backbone.cpu()
            torch.cuda.empty_cache()

            total_loss = 0
            n_round_samples = 0

            for fg_logits, feat_list, click_info, round_idx in round_samples:
                # Check contested region
                contested_any = False
                for i in range(K):
                    for j in range(i + 1, K):
                        if ((fg_logits[i] > 0) & (fg_logits[j] > 0)).any():
                            contested_any = True
                            break
                    if contested_any:
                        break
                if not contested_any:
                    continue

                # ── Crop to foreground ROI ──
                any_fg_np = np.zeros(spatial, dtype=bool)
                for fg in fg_logits:
                    any_fg_np |= (fg.numpy() > -2)
                fg_coords = np.argwhere(any_fg_np)
                if len(fg_coords) == 0:
                    continue

                roi_margin = 8
                roi_min = np.maximum(fg_coords.min(0) - roi_margin, 0)
                roi_max = np.minimum(
                    fg_coords.max(0) + roi_margin + 1,
                    [spatial[0], spatial[1], spatial[2]])
                roi_size = roi_max - roi_min
                max_roi = 96
                for d in range(3):
                    if roi_size[d] > max_roi:
                        center = (roi_min[d] + roi_max[d]) // 2
                        roi_min[d] = max(0, center - max_roi // 2)
                        roi_max[d] = min(spatial[d], roi_min[d] + max_roi)

                roi_sl = tuple(
                    slice(int(mn), int(mx))
                    for mn, mx in zip(roi_min, roi_max))
                roi_shape = tuple(
                    int(mx - mn) for mn, mx in zip(roi_min, roi_max))

                # ── Build 40ch resolver inputs on ROI ──
                resolver_inputs = []
                own_logit_list = []

                with autocast_ctx():
                    for k in range(K):
                        own = fg_logits[k][roi_sl]
                        others = [fg_logits[j][roi_sl]
                                  for j in range(K) if j != k]

                        if others:
                            stacked_others = torch.stack(others)
                            max_comp = stacked_others.max(0)[0]
                            sum_press = torch.sigmoid(
                                stacked_others).sum(0)
                        else:
                            max_comp = torch.zeros_like(own)
                            sum_press = torch.zeros_like(own)

                        # Click distance maps
                        fg_dist, bg_dist = compute_click_distance_maps(
                            spatial,
                            click_info[k]['fg'],
                            click_info[k]['bg'])
                        fg_dist_roi = fg_dist[roi_sl]
                        bg_dist_roi = bg_dist[roi_sl]

                        # prev_pred channel
                        pp = torch.from_numpy(
                            prev_preds[oids[k]][roi_sl].astype(np.float32)
                        ) if round_idx > 0 else torch.zeros(roi_shape)

                        # round index
                        round_ch = torch.full(roi_shape, round_idx / 5.0)

                        # image intensity
                        img_roi = torch.from_numpy(
                            image_np[roi_sl].astype(np.float32))

                        # decoder features
                        feat_roi = feat_list[k][
                            :, roi_sl[0], roi_sl[1], roi_sl[2]]

                        # 40ch: 3 comp + 32 feat + 2 click_dist
                        #        + 1 image + 1 prev_pred + 1 round_idx
                        resolver_in = torch.cat([
                            own.unsqueeze(0),              # 1: own logit
                            max_comp.unsqueeze(0),         # 2: max competitor
                            sum_press.unsqueeze(0),        # 3: sum pressure
                            feat_roi,                      # 4-35: decoder feat
                            torch.from_numpy(
                                fg_dist_roi).unsqueeze(0), # 36: fg click dist
                            torch.from_numpy(
                                bg_dist_roi).unsqueeze(0), # 37: bg click dist
                            img_roi.unsqueeze(0),          # 38: image
                            pp.unsqueeze(0),               # 39: prev_pred
                            round_ch.unsqueeze(0),         # 40: round idx
                        ], dim=0).unsqueeze(0).to(device)

                        resolver_inputs.append(resolver_in)
                        own_logit_list.append(own.to(device))

                    # ── Resolver forward (cross-attention) ──
                    refined_list = resolver(resolver_inputs, own_logit_list)

                    # ── Loss ──
                    bg = torch.full(roi_shape, BG_LOGIT, device=device)
                    stacked = torch.stack([bg] + refined_list)

                    gt_roi = gt_np[roi_sl]
                    gt_remapped = torch.zeros(
                        roi_shape, dtype=torch.long, device=device)
                    for new_idx, oid in enumerate(oids):
                        gt_remapped[gt_roi == oid] = new_idx + 1

                    fg_pos = torch.stack(
                        [fg_logits[j][roi_sl] > 0
                         for j in range(K)]).to(device)
                    contested = (fg_pos.sum(0) >= 2)

                    weight_map = torch.ones(
                        roi_shape, dtype=torch.float32, device=device)
                    weight_map[contested] = CONTESTED_WEIGHT

                    log_probs = F.log_softmax(stacked.float(), dim=0)
                    nll = F.nll_loss(
                        log_probs.unsqueeze(0),
                        gt_remapped.unsqueeze(0),
                        reduction='none')[0]
                    loss = (nll * weight_map).mean()

                total_loss = total_loss + loss
                n_round_samples += 1

                # Stats
                with torch.no_grad():
                    pred = stacked.argmax(0)
                    overall_acc = (pred == gt_remapped).float().mean().item()
                    n_contested = contested.sum().item()
                    if n_contested > 0:
                        cacc = (pred[contested] == gt_remapped[contested]) \
                            .float().mean().item()
                    else:
                        cacc = float('nan')

                losses.append(loss.item())
                accs.append(overall_acc)
                if not np.isnan(cacc):
                    caccs.append(cacc)
                n_steps += 1

                # Cleanup per-round
                del resolver_inputs, own_logit_list, refined_list
                del stacked, gt_remapped, fg_pos, contested, weight_map

            # Backward on accumulated loss across rounds
            if n_round_samples > 0:
                optimizer.zero_grad()
                (total_loss / n_round_samples).backward()
                torch.nn.utils.clip_grad_norm_(
                    resolver.parameters(), max_norm=5.0)
                optimizer.step()

            del fg_logits, feat_list, round_samples
            backbone.to(device)
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        ml = np.mean(losses) if losses else float('nan')
        ma = np.mean(accs) if accs else float('nan')
        mc = np.mean(caccs) if caccs else float('nan')
        print(f"Epoch {epoch:02d}: loss={ml:.4f}, acc={ma:.3f}, "
              f"contested_acc={mc:.3f} ({len(caccs)}/{n_steps}), "
              f"time={elapsed:.1f}s", flush=True)

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'resolver_state': resolver.state_dict(),
            'loss': ml,
            'contested_acc': mc,
            'config': {
                'in_ch': IN_CH, 'hidden': 32, 'num_heads': 4,
                'rounds': args.rounds,
            },
        }
        torch.save(ckpt, output_dir / f"epoch{epoch:02d}.pth")

        if not np.isnan(mc) and mc > best_cacc:
            best_cacc = mc
            torch.save(ckpt, output_dir / "best.pth")
            print(f"  -> New best contested_acc: {mc:.3f}", flush=True)

    # Save final
    torch.save(
        {'resolver_state': resolver.state_dict(),
         'config': {'in_ch': IN_CH, 'hidden': 32, 'num_heads': 4,
                    'rounds': args.rounds}},
        output_dir / "final.pth")
    print(f"\nTraining complete. Best contested_acc: {best_cacc:.3f}",
          flush=True)
    handle.remove()


if __name__ == "__main__":
    main()
