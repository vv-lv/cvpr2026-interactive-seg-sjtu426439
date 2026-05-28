#!/usr/bin/env python3
"""
在预计算数据上训练 CrossObjectResolver（零 feature gap）。

数据来自 scripts/precompute_resolver_features.py，features 由 nnInteractive
session 生成，与推理时 100% 一致。

训练极快：无需 backbone forward，每 epoch 只需加载 .pt + resolver forward + backward。

用法:
  python -u training/train_resolver_precomputed.py --epochs 30
  python -u training/train_resolver_precomputed.py --epochs 30 --data_dir experiments/precomputed_resolver_feats
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
from scipy.ndimage import distance_transform_edt as _edt_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.resolver_cross_attn import CrossObjectResolver

DEFAULT_DATA = PROJECT_ROOT / "experiments" / "precomputed_resolver_feats"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "resolver_precomputed"

IN_CH = 40
CONTESTED_WEIGHT = 10.0
MAX_ROI = 96


def compute_click_distance_maps(shape, fg_clicks, bg_clicks):
    def _dist(clicks):
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
    return _dist(fg_clicks), _dist(bg_clicks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all precomputed files
    pt_files = sorted(args.data_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} precomputed files", flush=True)

    # Resolver
    resolver = CrossObjectResolver(
        in_ch=IN_CH, hidden=32, num_heads=4).to(device)
    n_params = sum(p.numel() for p in resolver.parameters())
    print(f"CrossObjectResolver {IN_CH}ch: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(
        resolver.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\nTraining: {args.epochs} epochs, cosine lr {args.lr}→1e-5, "
          f"contested_weight={CONTESTED_WEIGHT}", flush=True)

    BG_LOGIT = 0.0
    best_cacc = 0.0

    for epoch in range(args.epochs):
        resolver.train()
        losses, accs, caccs, sigmoid_caccs = [], [], [], []
        n_steps = 0
        t0 = time.time()

        random.shuffle(pt_files)

        for fi, pt_path in enumerate(pt_files):
            try:
                data = torch.load(pt_path, weights_only=False)
            except Exception:
                continue

            gt_full = data['gt']
            image_zs = data['image_zscore'].astype(np.float32)
            shape_full = data['shape']
            oids = data['oids']

            if len(oids) < 2:
                continue

            prev_pred = None

            for rd in data['rounds']:
                roi_min = np.array(rd['roi_min'])
                roi_max = np.array(rd['roi_max'])
                round_idx = rd['round_idx']

                # Cap ROI to MAX_ROI
                roi_size = roi_max - roi_min
                for d in range(3):
                    if roi_size[d] > MAX_ROI:
                        c = (roi_min[d] + roi_max[d]) // 2
                        roi_min[d] = max(0, c - MAX_ROI // 2)
                        roi_max[d] = min(shape_full[d],
                                         roi_min[d] + MAX_ROI)

                roi_sl = tuple(
                    slice(int(mn), int(mx))
                    for mn, mx in zip(roi_min, roi_max))
                roi_shape = tuple(
                    int(mx - mn) for mn, mx in zip(roi_min, roi_max))

                # Collect per-object data
                per_obj = rd['per_obj']
                valid_oids = [o for o in oids if o in per_obj]
                K = len(valid_oids)
                if K < 2:
                    continue

                # Get per-object logits and check contested
                logits_np = {}
                for oid in valid_oids:
                    od = per_obj[oid]
                    # The saved data is cropped to the round's ROI
                    # But our ROI might be re-capped, need to sub-crop
                    saved_roi_min = np.array(rd['roi_min'])
                    offset = roi_min - saved_roi_min
                    offset = np.maximum(offset, 0)
                    sub_sl = tuple(
                        slice(int(off), int(off) + rs)
                        for off, rs in zip(offset, roi_shape))
                    logits_np[oid] = od['margin_logit'][
                        sub_sl].astype(np.float32)

                contested = np.zeros(roi_shape, dtype=bool)
                for i, oi in enumerate(valid_oids):
                    for j, oj in enumerate(valid_oids):
                        if j <= i:
                            continue
                        contested |= (
                            (logits_np[oi] > 0) & (logits_np[oj] > 0))

                if not contested.any():
                    # Update prev_pred anyway
                    prev_pred_new = np.zeros(shape_full, dtype=np.uint8)
                    for oid in valid_oids:
                        od = per_obj[oid]
                        saved_roi_min = np.array(rd['roi_min'])
                        saved_roi_max = np.array(rd['roi_max'])
                        saved_sl = tuple(
                            slice(int(mn), int(mx))
                            for mn, mx in zip(saved_roi_min, saved_roi_max))
                        prev_pred_new[saved_sl][od['mask'] > 0] = oid
                    prev_pred = prev_pred_new
                    continue

                # ── Build 40ch resolver inputs ──
                resolver_inputs = []
                own_logit_list = []

                for oid in valid_oids:
                    od = per_obj[oid]
                    saved_roi_min = np.array(rd['roi_min'])
                    offset = roi_min - saved_roi_min
                    offset = np.maximum(offset, 0)
                    sub_sl = tuple(
                        slice(int(off), int(off) + rs)
                        for off, rs in zip(offset, roi_shape))

                    own = torch.from_numpy(logits_np[oid])
                    others = [torch.from_numpy(logits_np[o])
                              for o in valid_oids if o != oid]

                    if others:
                        stacked = torch.stack(others)
                        max_comp = stacked.max(0)[0]
                        sum_press = torch.sigmoid(stacked).sum(0)
                    else:
                        max_comp = torch.zeros_like(own)
                        sum_press = torch.zeros_like(own)

                    # Features (sub-crop from saved ROI)
                    feat = torch.from_numpy(
                        od['feat'][:, sub_sl[0], sub_sl[1], sub_sl[2]]
                        .astype(np.float32))

                    # Click distance (compute in full image, crop to ROI)
                    clicks = od.get('clicks', {'fg': [], 'bg': []})
                    fg_dist, bg_dist = compute_click_distance_maps(
                        shape_full, clicks['fg'], clicks['bg'])
                    fg_dist_roi = torch.from_numpy(
                        fg_dist[roi_sl]).unsqueeze(0)
                    bg_dist_roi = torch.from_numpy(
                        bg_dist[roi_sl]).unsqueeze(0)

                    # Image (crop to ROI)
                    img_roi = torch.from_numpy(
                        image_zs[roi_sl]).unsqueeze(0)

                    # Prev pred
                    pp = torch.zeros(roi_shape)
                    if prev_pred is not None:
                        pp = torch.from_numpy(
                            (prev_pred[roi_sl] == oid).astype(np.float32))

                    # Round index
                    round_ch = torch.full(
                        roi_shape, round_idx / 5.0).unsqueeze(0)

                    resolver_in = torch.cat([
                        own.unsqueeze(0),
                        max_comp.unsqueeze(0),
                        sum_press.unsqueeze(0),
                        feat,
                        fg_dist_roi,
                        bg_dist_roi,
                        img_roi,
                        pp.unsqueeze(0),
                        round_ch,
                    ], dim=0).unsqueeze(0).to(device)

                    resolver_inputs.append(resolver_in)
                    own_logit_list.append(own.to(device))

                # ── Resolver forward ──
                refined_list = resolver(resolver_inputs, own_logit_list)

                # ── Loss ──
                bg = torch.full(roi_shape, BG_LOGIT, device=device)
                stacked_all = torch.stack([bg] + refined_list)

                gt_roi = gt_full[roi_sl]
                gt_remapped = torch.zeros(
                    roi_shape, dtype=torch.long, device=device)
                for new_idx, oid in enumerate(valid_oids):
                    gt_remapped[gt_roi == oid] = new_idx + 1

                contested_t = torch.from_numpy(contested).to(device)
                weight_map = torch.ones(
                    roi_shape, dtype=torch.float32, device=device)
                weight_map[contested_t] = CONTESTED_WEIGHT

                log_probs = F.log_softmax(stacked_all.float(), dim=0)
                nll = F.nll_loss(
                    log_probs.unsqueeze(0),
                    gt_remapped.unsqueeze(0),
                    reduction='none')[0]
                loss = (nll * weight_map).mean()

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    resolver.parameters(), max_norm=5.0)
                optimizer.step()

                # Stats
                with torch.no_grad():
                    pred = stacked_all.argmax(0)
                    overall_acc = (
                        pred == gt_remapped).float().mean().item()
                    n_contested = contested_t.sum().item()
                    if n_contested > 0:
                        cacc = (pred[contested_t] ==
                                gt_remapped[contested_t]) \
                            .float().mean().item()

                        # Sigmoid baseline: argmax of raw logits
                        sig_stacked = torch.stack(
                            [bg] + own_logit_list)
                        sig_pred = sig_stacked.argmax(0)
                        sig_cacc = (sig_pred[contested_t] ==
                                    gt_remapped[contested_t]) \
                            .float().mean().item()
                    else:
                        cacc = float('nan')
                        sig_cacc = float('nan')

                losses.append(loss.item())
                accs.append(overall_acc)
                if not np.isnan(cacc):
                    caccs.append(cacc)
                if not np.isnan(sig_cacc):
                    sigmoid_caccs.append(sig_cacc)
                n_steps += 1

                # Update prev_pred
                prev_pred_new = np.zeros(shape_full, dtype=np.uint8)
                for oid in valid_oids:
                    od = per_obj[oid]
                    saved_roi_min = np.array(rd['roi_min'])
                    saved_roi_max = np.array(rd['roi_max'])
                    saved_sl = tuple(
                        slice(int(mn), int(mx))
                        for mn, mx in zip(saved_roi_min, saved_roi_max))
                    prev_pred_new[saved_sl][od['mask'] > 0] = oid
                prev_pred = prev_pred_new

                # Cleanup
                del resolver_inputs, own_logit_list, refined_list
                del stacked_all, gt_remapped, contested_t, weight_map
                torch.cuda.empty_cache()

        scheduler.step()
        elapsed = time.time() - t0
        ml = np.mean(losses) if losses else float('nan')
        ma = np.mean(accs) if accs else float('nan')
        mc = np.mean(caccs) if caccs else float('nan')
        ms = np.mean(sigmoid_caccs) if sigmoid_caccs else float('nan')
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d}: loss={ml:.4f}, acc={ma:.3f}, "
              f"contested_acc={mc:.3f}, sigmoid_baseline={ms:.3f}, "
              f"Δ={mc-ms:+.3f}, lr={lr_now:.1e}, "
              f"({len(caccs)}/{n_steps} steps), time={elapsed:.1f}s",
              flush=True)

        # Save
        ckpt = {
            'epoch': epoch,
            'resolver_state': resolver.state_dict(),
            'loss': ml,
            'contested_acc': mc,
            'sigmoid_baseline': ms,
            'config': {'in_ch': IN_CH, 'hidden': 32, 'num_heads': 4},
        }
        torch.save(ckpt, args.output_dir / f"epoch{epoch:02d}.pth")

        if not np.isnan(mc) and mc > best_cacc:
            best_cacc = mc
            torch.save(ckpt, args.output_dir / "best.pth")
            print(f"  -> New best contested_acc: {mc:.3f} "
                  f"(sigmoid: {ms:.3f}, Δ: {mc-ms:+.3f})", flush=True)

    print(f"\nDone. Best contested_acc: {best_cacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
