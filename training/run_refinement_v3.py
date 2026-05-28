"""
RefinementModule training on precomputed data (v3).

Loads precomputed nnInt 6-round rollout data + F_global features.
Runs refinement sequentially through rounds with:
  - pred_nn_k: from precomputed (off-policy for clicks)
  - pred_prev: from refined output of previous round (on-policy)
  - memory tokens: from refined output (on-policy, detached)
  - clicks/cd maps: from precomputed click positions

This is ~25× faster than v2 (no nnInt session during training) and can
train on any volume size (precompute handles large volumes offline).

Usage:
    python -u training/run_refinement_v3.py \
        --data_dir experiments/refinement_v3_data \
        --feats_dir experiments/refinement_train_feats \
        --save_dir experiments/refinement_v3 \
        --gpu 0 --epochs 10 --lr 1e-3
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.refinement_module import RefinementModule, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_click_dist_map(shape_low, click_positions_low):
    """EDT from click positions, normalized to [0,1] (1=near, 0=far)."""
    if not click_positions_low:
        return np.zeros(shape_low, dtype=np.float32)
    mask = np.ones(shape_low, dtype=bool)
    for (z, y, x) in click_positions_low:
        z, y, x = int(z), int(y), int(x)
        if 0 <= z < shape_low[0] and 0 <= y < shape_low[1] and 0 <= x < shape_low[2]:
            mask[z, y, x] = False
    edt = distance_transform_edt(mask)
    return (1.0 - np.clip(edt, 0, 20) / 20.0).astype(np.float32)


def clicks_to_low_coords(click_list, R, fg_only=None):
    """Extract (z,y,x) in R³ space from click dicts.
    fg_only: True=fg only, False=bg only, None=all."""
    coords = []
    for c in click_list:
        if fg_only is not None and bool(c['fg']) != fg_only:
            continue
        pn = c['pos_norm']
        z = int(pn[0] * (R - 1))
        y = int(pn[1] * (R - 1))
        x = int(pn[2] * (R - 1))
        coords.append((z, y, x))
    return coords


class DiceBCE(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        inter = (pred * target).sum(dim=[2, 3, 4])
        union = pred.sum(dim=[2, 3, 4]) + target.sum(dim=[2, 3, 4])
        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        bce = F.binary_cross_entropy(pred, target, reduction='none').mean(dim=[1, 2, 3, 4])
        return 0.5 * dice.mean() + 0.5 * bce.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
class RefinementV3Trainer:

    def __init__(self, gpu, lr, R, R_attn, feats_dir, frozen=False):
        self.device = torch.device('cuda', gpu)
        self.R = R
        self.feats_dir = feats_dir
        self.frozen = frozen

        self.refiner = RefinementModule(R=R, R_attn=R_attn).to(self.device)
        print(f"RefinementModule: {count_parameters(self.refiner):,} params")

        self.criterion = DiceBCE().to(self.device)
        if not frozen:
            self.optimizer = torch.optim.AdamW(
                self.refiner.parameters(), lr=lr, weight_decay=1e-5)

    def load_F_global(self, case_stem):
        """Load F_global from feats_dir, upsample to R³."""
        p = self.feats_dir / f"{case_stem}.pt"
        if not p.exists():
            return None
        blob = torch.load(p, weights_only=False, map_location='cpu')
        feat = blob['F_global_low'].float()
        feat = F.interpolate(feat[None], size=(self.R,) * 3,
                             mode='trilinear', align_corners=False)[0]
        return feat.to(self.device)

    def _train_case(self, data_pt, F_global_R):
        """One case: sequential rounds, on-policy memory + pred_prev."""
        device = self.device
        R = self.R

        preds_soft = data_pt['preds_soft'].float()     # (n_rounds, K, R, R, R)
        gt_low = data_pt['gt_low']                      # (R, R, R) uint8
        clicks = data_pt['clicks']                      # n_rounds × K × var_len
        oids = data_pt['oids']
        n_rounds = preds_soft.shape[0]
        K = len(oids)

        if not self.frozen:
            self.optimizer.zero_grad()

        total_loss = 0.0
        n_bwd = 0
        refined_prev = {oid: torch.zeros(1, 1, R, R, R, device=device)
                        for oid in oids}
        memory_bank = {oid: [] for oid in oids}

        for round_idx in range(n_rounds):
            for oid_idx, oid in enumerate(oids):
                pred_nn_k = preds_soft[round_idx, oid_idx].to(device)  # (R, R, R)
                pred_nn_k = pred_nn_k.unsqueeze(0).unsqueeze(0)         # (1,1,R³)

                # Click distance maps from all clicks up to this round
                all_clicks_oid = clicks[round_idx][oid_idx]
                fg_low = clicks_to_low_coords(all_clicks_oid, R, fg_only=True)
                bg_low = clicks_to_low_coords(all_clicks_oid, R, fg_only=False)
                cd_fg = torch.from_numpy(
                    compute_click_dist_map((R, R, R), fg_low)
                )[None, None].to(device)
                cd_bg = torch.from_numpy(
                    compute_click_dist_map((R, R, R), bg_low)
                )[None, None].to(device)

                pred_prev = refined_prev[oid]
                mem = memory_bank[oid]
                mem_tokens = torch.stack(mem, 0)[None] if mem else None

                refined_low, delta = self.refiner(
                    F_global_R[None], pred_nn_k, pred_prev,
                    cd_fg, cd_bg, mem_tokens)

                gt_oid = (torch.from_numpy(
                    (gt_low.numpy() == oid).astype(np.float32)
                )[None, None]).to(device)
                loss = self.criterion(refined_low, gt_oid)

                total_loss += loss.item()
                n_bwd += 1

                if not self.frozen:
                    (loss / max(1, K * n_rounds)).backward()

                # On-policy state updates (detached)
                refined_prev[oid] = refined_low.detach()

                this_round_clicks = [
                    c for c in all_clicks_oid if c['round'] == round_idx]
                tok = self.refiner.encode_memory_token(
                    refined_low.detach(), this_round_clicks, round_idx)
                memory_bank[oid].append(tok[0].detach())

        if not self.frozen and n_bwd > 0:
            nn.utils.clip_grad_norm_(self.refiner.parameters(), max_norm=5.0)
            self.optimizer.step()

        return total_loss / max(1, n_bwd)

    def train_epoch(self, pt_files, epoch):
        random.shuffle(pt_files)
        if not self.frozen:
            self.refiner.train()
        else:
            self.refiner.eval()

        losses = []
        skipped = 0
        t0 = time.time()

        for fi, pt in enumerate(pt_files):
            try:
                data = torch.load(pt, weights_only=False, map_location='cpu')
            except Exception as e:
                print(f"  load fail {pt.name}: {e!r}")
                skipped += 1; continue

            F_global_R = self.load_F_global(data['name'])
            if F_global_R is None:
                skipped += 1; continue

            try:
                loss = self._train_case(data, F_global_R)
                losses.append(loss)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM {data['name']}")
                skipped += 1; continue
            except Exception as e:
                print(f"  error {data['name']}: {e!r}")
                import traceback; traceback.print_exc()
                skipped += 1; continue

            if (fi + 1) % 20 == 0:
                dt = time.time() - t0
                mean_l = float(np.mean(losses[-20:])) if losses else 0
                print(f"  [{fi+1}/{len(pt_files)}] loss={mean_l:.4f} "
                      f"skip={skipped} t={dt:.0f}s", flush=True)

        dt = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else 0
        print(f"Epoch {epoch}: loss={mean_loss:.4f} n={len(losses)} "
              f"skip={skipped} time={dt:.0f}s", flush=True)
        return mean_loss

    def save_checkpoint(self, path, epoch, loss, saved_args):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': self.refiner.state_dict(),
            'epoch': epoch, 'loss': loss, 'args': saved_args,
        }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--feats_dir', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_train_feats')
    p.add_argument('--save_dir', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_v3')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--R', type=int, default=96)
    p.add_argument('--R_attn', type=int, default=48)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--freeze_sanity', action='store_true')
    p.add_argument('--resume_ckpt', type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pt_files = sorted(args.data_dir.glob("*.pt"))
    # Filter to those with F_global
    have_feats = {p.stem for p in args.feats_dir.glob('*.pt')}
    pt_files = [f for f in pt_files if f.stem in have_feats]
    print(f"Training files: {len(pt_files)} (with F_global)")
    assert pt_files

    trainer = RefinementV3Trainer(
        gpu=args.gpu, lr=args.lr, R=args.R, R_attn=args.R_attn,
        feats_dir=args.feats_dir, frozen=args.freeze_sanity,
    )

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location='cpu', weights_only=False)
        trainer.refiner.load_state_dict(ckpt['model'])
        print(f"Resumed from {args.resume_ckpt}")

    saved_args = {k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}

    if args.freeze_sanity:
        print("\n=== Freeze sanity (zero-init identity check) ===")
        trainer.train_epoch(pt_files[:10], epoch=-1)
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(pt_files, epoch)
        ckpt_path = args.save_dir / f"epoch_{epoch:02d}.pt"
        trainer.save_checkpoint(ckpt_path, epoch, loss, saved_args)
        print(f"  → saved {ckpt_path}")

    print("Training done.")


if __name__ == '__main__':
    main()
