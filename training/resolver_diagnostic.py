#!/usr/bin/env python3
"""
Resolver 诊断实验：35ch + contested 加权 loss + 固定 bg_logit。

目的：验证 decoder features 是否包含足够的仲裁信息。
这是诊断实验，不是最终方案。

三个改动（vs Phase 1）：
1. 35ch 输入 = 3 competition + 32 decoder features（通过 hook 提取）
2. bg_logit 固定为 0（去掉捷径）
3. Contested voxels 加权 10x（让梯度聚焦竞争区域）

用法:
  python -u training/resolver_diagnostic.py --max_files 50 --epochs 15 --device cuda:0
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

import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.trainer import build_network, autocast_ctx
from training.interaction_sim import InteractionManager
from training.run_resolver import MultiObjectDataset

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "resolver_diagnostic"

MAX_OBJECTS = 4
CONTESTED_WEIGHT = 10.0  # contested voxels 的 loss 权重


class Resolver35ch(nn.Module):
    """35ch resolver: 3 competition + 32 decoder features → 1 refined logit。"""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(35, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)  # (B, 1, D, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backbone (冻结) + feature hook
    backbone, _ = build_network(args.checkpoint, deep_supervision=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Hook: 捕获 decoder 最后一层 features (32ch, 192³)
    captured_features = {}
    def hook_fn(module, input, output):
        captured_features['feat'] = output.detach()
    handle = backbone.decoder.stages[-1].register_forward_hook(hook_fn)

    # Resolver (35ch)
    resolver = Resolver35ch(hidden=32).to(device)
    n_params = sum(p.numel() for p in resolver.parameters())
    print(f"Resolver 35ch: {n_params:,} params")

    optimizer = torch.optim.AdamW(resolver.parameters(), lr=args.lr, weight_decay=1e-4)

    # Dataset
    dataset = MultiObjectDataset(data_dir=args.data_dir, max_files=args.max_files, augment=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    # Training
    print(f"\nDiagnostic: {args.epochs} epochs, {len(dataset)} files, "
          f"35ch input, contested_weight={CONTESTED_WEIGHT}, bg_logit=fixed(0)")

    BG_LOGIT = 0.0  # 固定，不可学习

    for epoch in range(args.epochs):
        resolver.train()
        losses, accs, caccs, n_steps = [], [], [], 0
        t0 = time.time()

        for batch in loader:
            image = batch['image'].to(device)      # (1, D, H, W) from custom collate
            image = image.unsqueeze(0)              # → (1, 1, D, H, W)
            gt_np = batch['gt'].numpy()              # (D, H, W) int64
            oids = batch['object_ids']
            if isinstance(oids, torch.Tensor):
                oids = oids.tolist()
            if len(oids) < 2:
                continue

            K = min(len(oids), MAX_OBJECTS)
            if len(oids) > K:
                oids = random.sample(oids, K)

            fg_logits = []
            feat_list = []
            spatial = tuple(image.shape[2:])  # (D, H, W)

            for oid in oids:
                gt_binary = (gt_np == oid).astype(np.uint8)
                mgr = InteractionManager(spatial)
                mgr.set_initial_bbox(gt_binary, jitter=0.05)
                inter = torch.from_numpy(mgr.get_numpy()).unsqueeze(0).to(device)
                input_8ch = torch.cat([image, inter], dim=1)

                with torch.no_grad(), autocast_ctx():
                    output = backbone(input_8ch)

                fg = (output[0, 1] - output[0, 0]).float().cpu()  # → CPU
                fg_logits.append(fg)
                feat_list.append(captured_features['feat'][0].float().cpu())  # (32, D, H, W) → CPU
                del input_8ch, output
            # Backbone 用完了，临时移到 CPU 释放 GPU 给 resolver
            backbone.cpu()
            torch.cuda.empty_cache()

            # Resolver forward (AMP 减半 activation 内存)
            scaler = torch.cuda.amp.GradScaler()
            refined_list = []

            with autocast_ctx():
                for k in range(K):
                    own = fg_logits[k]
                    others = [fg_logits[j] for j in range(K) if j != k]
                    if others:
                        stacked_others = torch.stack(others)
                        max_comp = stacked_others.max(0)[0]
                        sum_press = torch.sigmoid(stacked_others).sum(0)
                    else:
                        max_comp = torch.zeros_like(own)
                        sum_press = torch.zeros_like(own)

                    resolver_in = torch.cat([
                        own.unsqueeze(0),
                        max_comp.unsqueeze(0),
                        sum_press.unsqueeze(0),
                        feat_list[k],
                    ], dim=0).unsqueeze(0).to(device)

                    refined = resolver(resolver_in)
                    refined_list.append(refined[0, 0])
                    del resolver_in

                # Assembly
                bg = torch.full_like(refined_list[0], BG_LOGIT)
                stacked = torch.stack([bg] + refined_list)

                # Remap GT
                gt_tensor = torch.from_numpy(gt_np).to(device)
                gt_remapped = torch.zeros_like(gt_tensor)
                for new_idx, oid in enumerate(oids):
                    gt_remapped[gt_np == oid] = new_idx + 1

                # Contested-weighted CE loss
                fg_pos = torch.stack(fg_logits) > 0
                contested = (fg_pos.sum(0) >= 2).to(device)

                weight_map = torch.ones_like(gt_remapped, dtype=torch.float32)
                weight_map[contested] = CONTESTED_WEIGHT

                log_probs = F.log_softmax(stacked.float(), dim=0)
                nll = F.nll_loss(log_probs.unsqueeze(0), gt_remapped.unsqueeze(0),
                                 reduction='none')[0]
                loss = (nll * weight_map).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Stats
            with torch.no_grad():
                pred = stacked.argmax(0)
                overall_acc = (pred == gt_remapped).float().mean().item()
                n_contested = contested.sum().item()
                if n_contested > 0:
                    cacc = (pred[contested] == gt_remapped[contested]).float().mean().item()
                else:
                    cacc = float('nan')

            losses.append(loss.item())
            accs.append(overall_acc)
            if not np.isnan(cacc):
                caccs.append(cacc)
            n_steps += 1

            del fg_logits, feat_list, refined_list, stacked, gt_tensor, gt_remapped
            backbone.to(device)  # 移回 GPU 准备下一步
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        ml = np.mean(losses) if losses else float('nan')
        ma = np.mean(accs) if accs else float('nan')
        mc = np.mean(caccs) if caccs else float('nan')
        print(f"Epoch {epoch:02d}: loss={ml:.4f}, acc={ma:.3f}, "
              f"contested_acc={mc:.3f} ({len(caccs)}/{n_steps}), "
              f"time={elapsed:.1f}s")

        # 每 epoch 存 checkpoint（防 OOM 打断丢失）
        torch.save({'epoch': epoch, 'resolver_state': resolver.state_dict(),
                    'loss': ml, 'contested_acc': mc},
                   output_dir / f"epoch{epoch:02d}.pth")

    # Save final
    torch.save({'resolver_state': resolver.state_dict()}, output_dir / "final.pth")
    print(f"\nSaved to {output_dir}")
    handle.remove()


if __name__ == "__main__":
    main()
