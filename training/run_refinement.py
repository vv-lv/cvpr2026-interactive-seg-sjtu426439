"""
Train RefinementModule on precomputed data from scripts/precompute_refinement_data.py.

Per-round per-object supervised learning:
  sample = (F_global_low, pred_nn_k_low, pred_prev_low, click_dist_fg, click_dist_bg,
            memory_tokens [teacher-forced from GT], gt_binary)
  loss  = 0.5 * Dice(refined, gt) + 0.5 * BCE(refined, gt)

Memory is teacher-forced: tokens for rounds 0..k-1 are encoded from GT-derived
soft labels + real click history. Avoids error compounding at train time.

Usage:
    python -u training/run_refinement.py \
        --data_dir experiments/refinement_data \
        --save_dir experiments/refinement_v1 \
        --epochs 10 --num_files 50 --lr 1e-3 --gpu 0
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.refinement_module import RefinementModule, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_edt_channel(
    shape_low,              # (R, R, R)
    click_positions_low,    # list of (z, y, x) ints in low-res space
):
    """Compute Euclidean distance map where each click position is 0,
    decaying outward. Returns float32 of shape shape_low, clamped to [0, 20]
    and normalized to [0, 1] (inverted: 1 near click, 0 far)."""
    if not click_positions_low:
        return np.zeros(shape_low, dtype=np.float32)
    mask = np.ones(shape_low, dtype=bool)
    for z, y, x in click_positions_low:
        if 0 <= z < shape_low[0] and 0 <= y < shape_low[1] and 0 <= x < shape_low[2]:
            mask[z, y, x] = False
    edt = distance_transform_edt(mask)
    edt = np.clip(edt, 0, 20) / 20.0
    return (1.0 - edt).astype(np.float32)


def click_dicts_to_low(clicks: List[Dict], orig_shape, R: int, fg: bool):
    """Extract positions of fg or bg clicks, convert to low-res int coords."""
    coords = []
    for c in clicks:
        if bool(c['fg']) != fg:
            continue
        pn = c['pos_norm']
        z = int(pn[0] * (R - 1))
        y = int(pn[1] * (R - 1))
        x = int(pn[2] * (R - 1))
        coords.append((z, y, x))
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class RefinementDataset(Dataset):
    """Yields per (case, round, oid) samples."""

    def __init__(self, pt_files: List[Path], R: int = 96):
        self.R = R
        self.entries = []     # list of (path, round_idx, oid_idx)
        self.valid_files = []
        for p in pt_files:
            try:
                # Light-touch peek without loading full blob
                blob = torch.load(p, weights_only=False, map_location='cpu')
            except Exception as e:
                print(f"  skip (load error) {p.name}: {e!r}")
                continue
            if blob.get('F_global_low') is None:
                print(f"  skip (missing F_global) {p.name}")
                continue
            n_rounds = int(blob['preds_soft'].shape[0])
            n_obj = len(blob['oids'])
            self.valid_files.append(p)
            for r in range(n_rounds):
                for k in range(n_obj):
                    self.entries.append((p, r, k))
        print(f"  Dataset: {len(self.valid_files)} files, {len(self.entries)} samples")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pt, round_idx, oid_idx = self.entries[idx]
        blob = torch.load(pt, weights_only=False, map_location='cpu')
        R = int(blob['R'])
        oids = blob['oids']
        oid = oids[oid_idx]

        F_global = blob['F_global_low'].float()                # (192, F_R, F_R, F_R)
        preds_soft = blob['preds_soft'].float()                 # (n_rounds, K, R, R, R)
        gt_low = blob['gt_low']                                 # (R, R, R) uint8
        clicks = blob['clicks']                                 # n_rounds × K × list[dict]
        assembled = blob['assembled_low']                       # (n_rounds, R, R, R)

        pred_nn_k = preds_soft[round_idx, oid_idx]              # (R, R, R)
        # pred_prev: assembled prediction restricted to this oid at round_idx-1
        if round_idx > 0:
            prev_assembled = assembled[round_idx - 1]
            pred_prev = (prev_assembled == oid).float()
        else:
            pred_prev = torch.zeros_like(pred_nn_k)

        # Click dist maps for this round (all clicks up to this round, for this oid)
        click_list = clicks[round_idx][oid_idx]
        fg_low = click_dicts_to_low(click_list, blob['orig_shape'], R, fg=True)
        bg_low = click_dicts_to_low(click_list, blob['orig_shape'], R, fg=False)
        cd_fg = torch.from_numpy(compute_edt_channel((R, R, R), fg_low))
        cd_bg = torch.from_numpy(compute_edt_channel((R, R, R), bg_low))

        # Teacher-forced memory: for rounds 0..round_idx-1, encode (GT-for-oid, clicks_for_oid, round)
        gt_for_oid = (gt_low == oid).float()                    # (R, R, R)
        mem_inputs = []
        for prev_r in range(round_idx):
            prev_clicks = clicks[prev_r][oid_idx]
            mem_inputs.append({
                'pred_for_mem': gt_for_oid.clone(),             # teacher forcing
                'clicks': prev_clicks,
                'round': prev_r,
            })

        gt_binary = gt_for_oid
        return {
            'F_global': F_global,
            'pred_nn_k': pred_nn_k.unsqueeze(0),                # (1, R, R, R)
            'pred_prev': pred_prev.unsqueeze(0),                # (1, R, R, R)
            'cd_fg': cd_fg.unsqueeze(0),
            'cd_bg': cd_bg.unsqueeze(0),
            'mem_inputs': mem_inputs,                           # list of dicts
            'gt_binary': gt_binary.unsqueeze(0),                # (1, R, R, R)
            'round': round_idx,
            'oid': oid,
            'name': blob['name'],
        }


def collate_fn(batch):
    """Simple collate: stack tensors, keep mem_inputs as list-of-lists."""
    return {
        'F_global': torch.stack([b['F_global'] for b in batch]),
        'pred_nn_k': torch.stack([b['pred_nn_k'] for b in batch]),
        'pred_prev': torch.stack([b['pred_prev'] for b in batch]),
        'cd_fg': torch.stack([b['cd_fg'] for b in batch]),
        'cd_bg': torch.stack([b['cd_bg'] for b in batch]),
        'gt_binary': torch.stack([b['gt_binary'] for b in batch]),
        'mem_inputs': [b['mem_inputs'] for b in batch],
        'rounds': [b['round'] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dw = dice_weight
        self.bw = bce_weight
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        # Dice
        inter = (pred * target).sum(dim=[2, 3, 4])
        union = pred.sum(dim=[2, 3, 4]) + target.sum(dim=[2, 3, 4])
        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        # BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none').mean(dim=[1, 2, 3, 4])
        return (self.dw * dice.mean() + self.bw * bce.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────
def build_memory_tokens(model: RefinementModule, mem_inputs_batch, device):
    """Given a list-of-lists of mem_inputs (B × T_b), return a padded tensor
    (B, T_max, token_dim) with zero-padding for shorter sequences.

    For simplicity: since each sample has a fixed round_idx worth of memory,
    we group by batch-member and pad.
    """
    B = len(mem_inputs_batch)
    token_dim = model.token_dim
    T_max = max(len(m) for m in mem_inputs_batch) if mem_inputs_batch else 0
    if T_max == 0:
        return None
    out = torch.zeros(B, T_max, token_dim, device=device)
    for bi, mem_list in enumerate(mem_inputs_batch):
        for ti, m in enumerate(mem_list):
            pred = m['pred_for_mem'].to(device).unsqueeze(0).unsqueeze(0)  # (1,1,R,R,R)
            tok = model.encode_memory_token(pred, m['clicks'], m['round'])
            out[bi, ti] = tok[0]
    return out


def train_epoch(model, loader, optimizer, criterion, device, R_target: int):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        F_global = batch['F_global'].to(device)
        # Upsample F_global to R_target
        if F_global.shape[-1] != R_target:
            F_global = F.interpolate(
                F_global, size=(R_target, R_target, R_target),
                mode='trilinear', align_corners=False)
        pred_nn_k = batch['pred_nn_k'].to(device)
        pred_prev = batch['pred_prev'].to(device)
        cd_fg = batch['cd_fg'].to(device)
        cd_bg = batch['cd_bg'].to(device)
        gt = batch['gt_binary'].to(device)

        mem_tokens = build_memory_tokens(model, batch['mem_inputs'], device)

        refined, delta = model(F_global, pred_nn_k, pred_prev, cd_fg, cd_bg, mem_tokens)
        loss = criterion(refined, gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = F_global.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def compute_baseline_loss(loader, criterion, device):
    """Loss of pred_nn_k (without refinement) against GT — the reference target."""
    total = 0.0
    n = 0
    for batch in loader:
        pred_nn = batch['pred_nn_k'].to(device)
        gt = batch['gt_binary'].to(device)
        loss = criterion(pred_nn, gt)
        bs = pred_nn.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path, default=PROJECT_ROOT / 'experiments/refinement_data')
    p.add_argument('--save_dir', type=Path, default=PROJECT_ROOT / 'experiments/refinement_v1')
    p.add_argument('--num_files', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--R', type=int, default=96)
    p.add_argument('--R_attn', type=int, default=48)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--freeze_sanity', action='store_true',
                   help="Only compute baseline loss (no training)")
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda', args.gpu)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(args.data_dir.glob("*.pt"))
    if args.num_files > 0:
        pt_files = pt_files[:args.num_files]
    print(f"Found {len(pt_files)} precomputed files in {args.data_dir}")
    assert pt_files, "no data"

    dataset = RefinementDataset(pt_files, R=args.R)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    model = RefinementModule(R=args.R, R_attn=args.R_attn).to(device)
    print(f"Model params: {count_parameters(model):,}")

    criterion = DiceBCELoss().to(device)

    # ── Freeze sanity check ────────────────────────────────────────────────
    baseline_loss = compute_baseline_loss(loader, criterion, device)
    print(f"Baseline loss (pred_nn_k vs GT): {baseline_loss:.4f}")

    model.eval()
    with torch.no_grad():
        init_losses = []
        for batch in loader:
            F_global = batch['F_global'].to(device)
            if F_global.shape[-1] != args.R:
                F_global = F.interpolate(F_global, size=(args.R, args.R, args.R),
                                         mode='trilinear', align_corners=False)
            pred_nn_k = batch['pred_nn_k'].to(device)
            pred_prev = batch['pred_prev'].to(device)
            cd_fg = batch['cd_fg'].to(device)
            cd_bg = batch['cd_bg'].to(device)
            gt = batch['gt_binary'].to(device)
            mem = build_memory_tokens(model, batch['mem_inputs'], device)
            refined, delta = model(F_global, pred_nn_k, pred_prev, cd_fg, cd_bg, mem)
            init_losses.append(criterion(refined, gt).item())
    init_loss = sum(init_losses) / len(init_losses)
    diff = abs(init_loss - baseline_loss)
    print(f"Init refined loss: {init_loss:.4f} (|Δ|={diff:.6e})")
    assert diff < 1e-4, f"Init loss should equal baseline (zero-init failed): Δ={diff}"
    print("✓ Zero-init identity confirmed.\n")

    if args.freeze_sanity:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, criterion, device, args.R)
        dt = time.time() - t0
        print(f"Epoch {epoch:2d} | loss={loss:.4f} (baseline={baseline_loss:.4f}, "
              f"Δ={baseline_loss-loss:+.4f}) | {dt:.1f}s")
        ckpt_path = args.save_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'baseline_loss': baseline_loss,
            'args': vars(args),
        }, ckpt_path)

    print(f"\nTraining done. Checkpoints in {args.save_dir}")


if __name__ == '__main__':
    main()
