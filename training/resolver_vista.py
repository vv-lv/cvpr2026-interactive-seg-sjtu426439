#!/usr/bin/env python3
"""
VISTA3D-guided Resolver: 用外部 encoder features 替代 nnInteractive decoder features。

输入 ch:
  ch0:     own_margin_logit        — 当前 object 的 margin (fg-bg)
  ch1:     max_competitor           — 最强竞争者的 margin
  ch2:     sum_pressure             — sum(sigmoid(其他 objects 的 margin))
  ch3-18:  encoder_features (16ch)  — VISTA3D Stage2 features, PCA 降维到 16ch
  总计: 19ch

或完整版（不降维）:
  ch0-2:   competition (3ch)
  ch3-194: encoder_features (192ch)
  总计: 195ch → 用 1x1 conv 先压缩

训练:
  python -u training/resolver_vista.py --epochs 30 --lr 1e-3

评估（offline, 用预计算数据）:
  python -u training/resolver_vista.py --mode eval --checkpoint experiments/resolver_vista/best.pth
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

DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "resolver_vista_crops"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "resolver_vista"

MAX_OBJECTS = 6
CONTESTED_WEIGHT = 10.0
CROP_MARGIN = 8         # contested bbox 外扩 voxels
MIN_CROP_SIZE = 16      # 最小 crop 尺寸
MAX_CROP_SIZE = 128     # 最大 crop 尺寸（防 OOM）


class ResolverVista(nn.Module):
    """Resolver with external encoder features.

    压缩 encoder features 到 16ch，然后和 3ch competition 一起处理。
    """

    def __init__(self, encoder_ch=192, compress_ch=16, hidden=32):
        super().__init__()
        # 1x1 conv 压缩 encoder features
        self.compress = nn.Sequential(
            nn.Conv3d(encoder_ch, compress_ch, 1),
            nn.InstanceNorm3d(compress_ch),
            nn.LeakyReLU(inplace=True),
        )
        # Main resolver: 3ch competition + compress_ch encoder features
        in_ch = 3 + compress_ch
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, competition_3ch, encoder_feat):
        """
        Args:
            competition_3ch: (B, 3, D, H, W) — own, max_comp, sum_press
            encoder_feat: (B, 192, D', H', W') — VISTA3D features (可能低分辨率)

        Returns:
            refined: (B, 1, D, H, W)
        """
        # 上采样 encoder features 到 competition 分辨率
        target_size = competition_3ch.shape[2:]
        if encoder_feat.shape[2:] != target_size:
            encoder_feat = F.interpolate(encoder_feat, size=target_size, mode='trilinear', align_corners=False)

        compressed = self.compress(encoder_feat)  # (B, 16, D, H, W)
        x = torch.cat([competition_3ch, compressed], dim=1)  # (B, 19, D, H, W)
        return self.net(x)


class ResolverCropDataset(Dataset):
    """从预提取的 contested crops 加载（小文件，快速读取）。"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        print(f"ResolverCropDataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        except Exception:
            return None
        if 'comp_crops' not in data:
            return None
        return data


def find_contested_crop(fg_logits, vol_shape, margin=CROP_MARGIN,
                        min_size=MIN_CROP_SIZE, max_size=MAX_CROP_SIZE):
    """找到 contested region 的 crop bbox。

    Args:
        fg_logits: list of (D, H, W) tensors — 每个 object 的 margin logit
        vol_shape: (D, H, W)
        margin: bbox 外扩的 voxels

    Returns:
        slices: tuple of 3 slices，或 None（无 contested 区域）
        contested_mask_crop: (D', H', W') bool tensor（crop 内的 contested mask）
    """
    # 找 contested voxels: ≥2 objects 的 logit > 0
    fg_pos = torch.stack(fg_logits) > 0  # (K, D, H, W)
    contested = fg_pos.sum(0) >= 2       # (D, H, W)
    n_contested = contested.sum().item()

    if n_contested == 0:
        # 无竞争区域 → 用所有 fg 的 union 代替
        any_fg = fg_pos.any(0)
        if not any_fg.any():
            return None, None
        contested = any_fg

    coords = torch.nonzero(contested)  # (N, 3)
    mn = coords.min(0).values.tolist()
    mx = coords.max(0).values.tolist()

    slices = []
    for d in range(3):
        lo = max(0, mn[d] - margin)
        hi = min(vol_shape[d], mx[d] + 1 + margin)
        # 限制最大 crop 尺寸
        if hi - lo > max_size:
            center = (mn[d] + mx[d]) // 2
            lo = max(0, center - max_size // 2)
            hi = min(vol_shape[d], lo + max_size)
            lo = max(0, hi - max_size)
        # 确保最小尺寸
        if hi - lo < min_size:
            center = (lo + hi) // 2
            lo = max(0, center - min_size // 2)
            hi = min(vol_shape[d], lo + min_size)
            lo = max(0, hi - min_size)
        slices.append(slice(lo, hi))

    slices = tuple(slices)
    contested_crop = contested[slices]
    return slices, contested_crop


def train_step(resolver, batch, device, scaler):
    """一个训练步：直接从预提取的 contested crops 操作。"""
    comp_crops = batch['comp_crops']       # list of (3, d, h, w) half tensors
    enc_feat_crop = batch['enc_feat_crop']  # (192, d', h', w') half
    gt_crop = batch['gt_crop']              # (d, h, w) int64
    contested_mask = batch['contested_mask']  # (d, h, w) bool
    oids = batch['oids']

    K = len(comp_crops)
    if K < 2:
        return None

    enc_feat_batch = enc_feat_crop.float().unsqueeze(0).to(device)
    refined_list = []

    with torch.autocast('cuda', enabled=True):
        for k in range(K):
            comp_3ch = comp_crops[k].float().unsqueeze(0).to(device)  # (1, 3, d, h, w)
            refined = resolver(comp_3ch, enc_feat_batch)
            refined_list.append(refined[0, 0])
            del comp_3ch

        # Assembly + Loss
        BG_LOGIT = 0.0
        bg = torch.full_like(refined_list[0], BG_LOGIT)
        stacked = torch.stack([bg] + refined_list)

        gt_dev = gt_crop.to(device)
        gt_remapped = torch.zeros_like(gt_dev)
        for new_idx, oid in enumerate(oids):
            gt_remapped[gt_dev == oid] = new_idx + 1

        contested_dev = contested_mask.to(device)
        weight_map = torch.ones_like(gt_remapped, dtype=torch.float32)
        weight_map[contested_dev] = CONTESTED_WEIGHT

        log_probs = F.log_softmax(stacked.float(), dim=0)
        nll = F.nll_loss(log_probs.unsqueeze(0), gt_remapped.unsqueeze(0), reduction='none')[0]
        loss = (nll * weight_map).mean()

    with torch.no_grad():
        pred = stacked.argmax(0)
        overall_acc = (pred == gt_remapped).float().mean().item()
        n_contested = contested_dev.sum().item()
        if n_contested > 0:
            cacc = (pred[contested_dev] == gt_remapped[contested_dev]).float().mean().item()
        else:
            cacc = float('nan')

    del refined_list, stacked, gt_dev, gt_remapped, enc_feat_batch, contested_dev
    return loss, overall_acc, cacc, n_contested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_ch", type=int, default=192)
    parser.add_argument("--compress_ch", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = ResolverCropDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x[0])

    resolver = ResolverVista(
        encoder_ch=args.encoder_ch,
        compress_ch=args.compress_ch,
        hidden=args.hidden,
    ).to(device)
    n_params = sum(p.numel() for p in resolver.parameters())
    print(f"ResolverVista: {n_params:,} params "
          f"(encoder_ch={args.encoder_ch}, compress={args.compress_ch}, hidden={args.hidden})")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        resolver.load_state_dict(ckpt['resolver_state'])
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.mode == "train":
        optimizer = torch.optim.AdamW(resolver.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        print(f"\nTraining: {args.epochs} epochs, {len(dataset)} files")
        best_cacc = 0

        for epoch in range(args.epochs):
            resolver.train()
            losses, accs, caccs, n_steps = [], [], [], 0
            t0 = time.time()

            for batch in loader:
                if batch is None:
                    continue
                optimizer.zero_grad()
                ret = train_step(resolver, batch, device, scaler)
                if ret is None:
                    continue
                loss, acc, cacc, n_cont = ret
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())
                accs.append(acc)
                if not np.isnan(cacc):
                    caccs.append(cacc)
                n_steps += 1

            elapsed = time.time() - t0
            ml = np.mean(losses) if losses else float('nan')
            ma = np.mean(accs) if accs else float('nan')
            mc = np.mean(caccs) if caccs else float('nan')

            print(f"Epoch {epoch:02d}: loss={ml:.4f}, acc={ma:.3f}, "
                  f"contested_acc={mc:.3f} ({len(caccs)}/{n_steps}), "
                  f"time={elapsed:.1f}s")

            if mc > best_cacc and not np.isnan(mc):
                best_cacc = mc
                torch.save({
                    'epoch': epoch, 'resolver_state': resolver.state_dict(),
                    'loss': ml, 'contested_acc': mc,
                    'config': {'encoder_ch': args.encoder_ch, 'compress_ch': args.compress_ch,
                               'hidden': args.hidden},
                }, args.output / "best.pth")

            # Save every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch, 'resolver_state': resolver.state_dict(),
                    'loss': ml, 'contested_acc': mc,
                    'config': {'encoder_ch': args.encoder_ch, 'compress_ch': args.compress_ch,
                               'hidden': args.hidden},
                }, args.output / f"epoch{epoch:02d}.pth")

        torch.save({'resolver_state': resolver.state_dict(),
                    'config': {'encoder_ch': args.encoder_ch, 'compress_ch': args.compress_ch,
                               'hidden': args.hidden}},
                   args.output / "final.pth")
        print(f"\nBest contested_acc: {best_cacc:.3f}")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
