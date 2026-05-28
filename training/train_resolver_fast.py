#!/usr/bin/env python3
"""
Phase 1 Resolver 快速训练：从预计算的 logits 训练，不跑 backbone。

每步只需读 .pt 文件 → 构造 3ch → resolver forward → CE loss → backward。
1330 参数的小网络，每 epoch 几十秒。

用法:
  # 先预计算 logits
  python -u training/precompute_logits.py --max_files 500 --device cuda:0

  # 再训 resolver
  python -u training/train_resolver_fast.py --epochs 15 --lr 1e-3

  # 评估
  python -u training/train_resolver_fast.py --mode eval
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
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.resolver import PerObjectResolver, compute_competition_channels

DEFAULT_LOGITS_DIR = PROJECT_ROOT / "experiments" / "precomputed_logits"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "resolver_p1"


class PrecomputedLogitsDataset(Dataset):
    """从磁盘读预计算的 per-object logits + multi-class GT。"""

    def __init__(self, logits_dir: str, min_objects: int = 2):
        self.logits_dir = Path(logits_dir)
        self.files = sorted(self.logits_dir.glob("*.pt"))
        # 过滤 ≥min_objects 的文件（应该全部满足）
        print(f"PrecomputedLogitsDataset: {len(self.files)} files from {logits_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu')
        # data = {'logits': {oid: (D,H,W) fp16}, 'gt_patch': (D,H,W) uint8, 'patch_oids': list}
        return {
            'logits': {k: v.float() for k, v in data['logits'].items()},
            'gt_patch': torch.from_numpy(data['gt_patch'].astype(np.int64)),
            'patch_oids': data['patch_oids'],
            'name': self.files[idx].stem,
        }


def train_step(resolver, batch, device):
    """一个训练步骤。返回 (loss, overall_acc, contested_acc, n_contested)。"""
    logits_dict = batch['logits']
    gt = batch['gt_patch'].to(device)      # (D, H, W) int64
    oids = batch['patch_oids']

    if len(oids) < 2:
        return None

    # 限制 K
    K = min(len(oids), 4)
    if len(oids) > K:
        oids = random.sample(oids, K)

    # 构建 fg_logits 列表
    fg_logits = [logits_dict[oid].to(device) for oid in oids]

    # Resolver forward
    refined_list = []
    for k in range(K):
        comp_ch = compute_competition_channels(fg_logits, k)  # (3, D, H, W)
        resolver_in = comp_ch.unsqueeze(0)  # (1, 3, D, H, W)
        refined = resolver(resolver_in)      # (1, 1, D, H, W)
        refined_list.append(refined[0, 0])   # (D, H, W)

    # Assembly
    bg = resolver.bg_logit.expand_as(refined_list[0])
    stacked = torch.stack([bg] + refined_list)  # (K+1, D, H, W)

    # Remap GT: 0=bg, 1..K=objects
    gt_remapped = torch.zeros_like(gt)
    for new_idx, oid in enumerate(oids):
        gt_remapped[gt == oid] = new_idx + 1

    # CE loss
    loss = F.cross_entropy(stacked.unsqueeze(0), gt_remapped.unsqueeze(0))

    # 统计
    with torch.no_grad():
        pred = stacked.argmax(0)
        overall_acc = (pred == gt_remapped).float().mean().item()

        # Contested: ≥2 objects 的原始 fg logit > 0
        fg_pos = torch.stack(fg_logits) > 0
        contested = fg_pos.sum(0) >= 2
        n_contested = contested.sum().item()
        if n_contested > 0:
            contested_acc = (pred[contested] == gt_remapped[contested]).float().mean().item()
        else:
            contested_acc = float('nan')

    return loss, overall_acc, contested_acc, n_contested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--logits_dir", default=str(DEFAULT_LOGITS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PrecomputedLogitsDataset(args.logits_dir)
    # batch_size=1 因为每个 case 的 object 数量不同
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])  # 直接返回 dict，不做 batch collate

    resolver = PerObjectResolver(hidden=16).to(device)
    n_params = sum(p.numel() for p in resolver.parameters())
    print(f"Resolver: {n_params} params")

    if args.mode == "train":
        optimizer = torch.optim.AdamW(resolver.parameters(), lr=args.lr, weight_decay=1e-4)

        print(f"\nTraining: {args.epochs} epochs, {len(dataset)} files")
        best_loss = float('inf')

        for epoch in range(args.epochs):
            resolver.train()
            losses, accs, caccs, n_steps = [], [], [], 0
            t0 = time.time()

            for batch in loader:
                optimizer.zero_grad()
                ret = train_step(resolver, batch, device)
                if ret is None:
                    continue
                loss, acc, cacc, n_cont = ret
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accs.append(acc)
                if not np.isnan(cacc):
                    caccs.append(cacc)
                n_steps += 1

            elapsed = time.time() - t0
            ml = np.mean(losses) if losses else float('nan')
            ma = np.mean(accs) if accs else float('nan')
            mc = np.mean(caccs) if caccs else float('nan')
            bg = resolver.bg_logit.item()

            print(f"Epoch {epoch:02d}: loss={ml:.4f}, acc={ma:.3f}, "
                  f"contested_acc={mc:.3f} ({len(caccs)}/{n_steps} batches), "
                  f"bg_logit={bg:.3f}, time={elapsed:.1f}s")

            if ml < best_loss:
                best_loss = ml
                torch.save({'epoch': epoch, 'resolver_state': resolver.state_dict(),
                            'loss': ml, 'contested_acc': mc},
                           output_dir / "best.pth")

        torch.save({'epoch': args.epochs - 1, 'resolver_state': resolver.state_dict()},
                   output_dir / "final.pth")
        print(f"\nBest loss: {best_loss:.4f}")
        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
