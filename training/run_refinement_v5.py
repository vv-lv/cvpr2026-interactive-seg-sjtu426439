#!/usr/bin/env python3
"""
Train PatchRefinementModule (v5) on v4 full-resolution rollouts.

Loop (per case):
  - load v4 .pt (preds_soft_full (6,K,D,H,W) fp16, gt_full, clicks, oids)
  - load F_global (48^3) → extract_global_tokens once
  - per oid:
      refined_full_prev = zeros(D,H,W) fp16 on CPU
      memory_bank = []
      per round_idx in 0..5:
          pred_nn_k_full = preds_soft_full[r, oid_idx]
          sample 4 patches (2 boundary + 2 random-on-object)
          for each patch: crop inputs, build EDT cube locally, forward,
                          BCE+Dice loss, backward (scaled).
          write refined patches into refined_full_prev (fp16)
          encode memory token from refined_full_prev downsampled to 96^3.

Checkpointing: epoch_XX.pt under args.output.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy._core  # noqa: F401
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.patch_refinement_module import (
    PatchRefinementModule, count_parameters,
)


# ─────────────────────────────────────────────────────────────────────────────
# EDT per patch
# ─────────────────────────────────────────────────────────────────────────────
def compute_edt_patch(patch_shape, patch_offset, clicks_full, clip=20):
    """Local EDT inside a patch. clicks_full: list of (z,y,x) full-res coords.
    Returns float32 (P,P,P), values in [0,1] (1 = at click, 0 = ≥clip away)."""
    z0, y0, x0 = patch_offset
    P = patch_shape
    result = np.full(P, clip, dtype=np.float32)
    for (cz, cy, cx) in clicks_full:
        lz, ly, lx = cz - z0, cy - y0, cx - x0
        if (lz < -clip or lz > P[0] + clip or
            ly < -clip or ly > P[1] + clip or
            lx < -clip or lx > P[2] + clip):
            continue
        zmin = max(0, lz - clip); zmax = min(P[0], lz + clip + 1)
        ymin = max(0, ly - clip); ymax = min(P[1], ly + clip + 1)
        xmin = max(0, lx - clip); xmax = min(P[2], lx + clip + 1)
        if zmin >= zmax or ymin >= ymax or xmin >= xmax:
            continue
        dz = (np.arange(zmin, zmax) - lz).astype(np.float32)
        dy = (np.arange(ymin, ymax) - ly).astype(np.float32)
        dx = (np.arange(xmin, xmax) - lx).astype(np.float32)
        dist = np.sqrt(
            dz[:, None, None] ** 2
            + dy[None, :, None] ** 2
            + dx[None, None, :] ** 2
        )
        result[zmin:zmax, ymin:ymax, xmin:xmax] = np.minimum(
            result[zmin:zmax, ymin:ymax, xmin:xmax], dist)
    return (1.0 - np.clip(result, 0, clip) / clip).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Patch sampling
# ─────────────────────────────────────────────────────────────────────────────
def object_bbox(gt_oid_np, pad):
    coords = np.argwhere(gt_oid_np > 0)
    if len(coords) == 0:
        D, H, W = gt_oid_np.shape
        return (0, D), (0, H), (0, W)
    mn = coords.min(0); mx = coords.max(0) + 1
    D, H, W = gt_oid_np.shape
    z0 = max(0, mn[0] - pad); z1 = min(D, mx[0] + pad)
    y0 = max(0, mn[1] - pad); y1 = min(H, mx[1] + pad)
    x0 = max(0, mn[2] - pad); x1 = min(W, mx[2] + pad)
    return (z0, z1), (y0, y1), (x0, x1)


def sample_patches(shape, bbox, error_mask, n_boundary, n_random, R, rng):
    """Sample patch offsets (z0, y0, x0). Patches are size R each, clipped to volume."""
    D, H, W = shape
    (bz0, bz1), (by0, by1), (bx0, bx1) = bbox
    patches = []

    # 2 boundary: pick from error_mask nonzero positions
    nz = np.argwhere(error_mask > 0)
    if len(nz) > 0:
        n_pick = min(n_boundary, len(nz))
        idx = rng.choice(len(nz), size=n_pick, replace=False)
        for i in idx:
            z, y, x = nz[i]
            z0 = int(np.clip(z - R // 2, 0, D - R))
            y0 = int(np.clip(y - R // 2, 0, H - R))
            x0 = int(np.clip(x - R // 2, 0, W - R))
            patches.append((z0, y0, x0))

    # 2 random on-object (use expanded bbox)
    def rand_in(lo, hi, dim_max):
        lo_c = max(0, lo - R // 2)
        hi_c = min(dim_max - R, max(lo_c, hi - R // 2))
        if hi_c <= lo_c:
            return max(0, min(dim_max - R, (lo + hi) // 2 - R // 2))
        return int(rng.integers(lo_c, hi_c + 1))

    need_rand = (n_boundary + n_random) - len(patches)
    for _ in range(need_rand):
        z0 = rand_in(bz0, bz1, D)
        y0 = rand_in(by0, by1, H)
        x0 = rand_in(bx0, bx1, W)
        patches.append((z0, y0, x0))

    return patches


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
class DiceBCE(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        inter = (pred * target).sum(dim=[2, 3, 4])
        union = pred.sum(dim=[2, 3, 4]) + target.sum(dim=[2, 3, 4])
        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        bce = F.binary_cross_entropy(pred, target, reduction='none').mean(
            dim=[1, 2, 3, 4])
        return 0.5 * dice.mean() + 0.5 * bce.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class RefinementV5Trainer:

    def __init__(self, gpu, lr, R, R_attn, feats_dir, num_global_queries,
                 n_boundary, n_random, edt_clip, frozen=False):
        self.device = torch.device('cuda', gpu)
        self.R = R
        self.feats_dir = feats_dir
        self.n_boundary = n_boundary
        self.n_random = n_random
        self.edt_clip = edt_clip
        self.frozen = frozen

        self.refiner = PatchRefinementModule(
            R=R, R_attn=R_attn,
            num_global_queries=num_global_queries,
        ).to(self.device)
        print(f"PatchRefinementModule: "
              f"{count_parameters(self.refiner):,} params "
              f"(R={R}, R_attn={R_attn}, K={num_global_queries})")

        self.criterion = DiceBCE().to(self.device)
        if not frozen:
            self.optimizer = torch.optim.AdamW(
                self.refiner.parameters(), lr=lr, weight_decay=1e-5)

    # ------------------------------------------------------------------
    def load_F_global(self, case_stem):
        p = self.feats_dir / f"{case_stem}.pt"
        if not p.exists():
            return None
        blob = torch.load(p, weights_only=False, map_location='cpu')
        return blob['F_global_low'].float().to(self.device)   # (192, 48, 48, 48)

    # ------------------------------------------------------------------
    def _crop(self, arr, z0, y0, x0):
        R = self.R
        return arr[z0:z0 + R, y0:y0 + R, x0:x0 + R]

    def _train_case(self, data_pt, F_global):
        device = self.device
        R = self.R

        preds_soft_full = data_pt['preds_soft_full']    # (6,K,D,H,W) fp16 cpu
        gt_full = data_pt['gt_full']                    # (D,H,W) uint8 cpu
        clicks_all = data_pt['clicks']                  # n_rounds × K × var_len
        oids = data_pt['oids']
        n_rounds = preds_soft_full.shape[0]
        K_obj = preds_soft_full.shape[1]
        D0, H0, W0 = gt_full.shape
        # Pad spatial dims to at least R (ACDC has D<96)
        D, H, W = max(D0, R), max(H0, R), max(W0, R)
        if (D, H, W) != (D0, H0, W0):
            pz, py, px = D - D0, H - H0, W - W0
            preds_soft_full = F.pad(
                preds_soft_full, (0, px, 0, py, 0, pz), mode='constant', value=0)
            gt_full = F.pad(
                gt_full.unsqueeze(0).unsqueeze(0),
                (0, px, 0, py, 0, pz), mode='constant', value=0
            ).squeeze(0).squeeze(0)

        if not self.frozen:
            self.optimizer.zero_grad()

        total_loss_sum = 0.0
        n_bwd = 0
        rng = np.random.default_rng()

        for oid_idx, oid in enumerate(oids):
            gt_oid_np = (gt_full.numpy() == oid).astype(np.uint8)
            if gt_oid_np.sum() == 0:
                continue
            bbox = object_bbox(gt_oid_np, pad=R // 2)

            refined_full_prev = torch.zeros(
                (D, H, W), dtype=torch.float16)
            memory_tokens_list = []

            for round_idx in range(n_rounds):
                pred_nn_k_full = preds_soft_full[round_idx, oid_idx]  # fp16 cpu
                pred_nn_k_np = pred_nn_k_full.float().numpy()

                all_clicks_oid = clicks_all[round_idx][oid_idx]
                fg_coords_full = []
                bg_coords_full = []
                for c in all_clicks_oid:
                    pn = c['pos_norm']
                    z = int(pn[0] * (D0 - 1))
                    y = int(pn[1] * (H0 - 1))
                    x = int(pn[2] * (W0 - 1))
                    if c.get('fg', True):
                        fg_coords_full.append((z, y, x))
                    else:
                        bg_coords_full.append((z, y, x))

                pred_bin = (pred_nn_k_np > 0.5).astype(np.uint8)
                error_mask = (pred_bin != gt_oid_np).astype(np.uint8)

                patches = sample_patches(
                    (D, H, W), bbox, error_mask,
                    self.n_boundary, self.n_random, R, rng)
                N = len(patches)

                # Build batch tensors
                pred_nn_k_batch = np.zeros((N, 1, R, R, R), dtype=np.float32)
                pred_prev_batch = np.zeros((N, 1, R, R, R), dtype=np.float32)
                cd_fg_batch = np.zeros((N, 1, R, R, R), dtype=np.float32)
                cd_bg_batch = np.zeros((N, 1, R, R, R), dtype=np.float32)
                gt_batch = np.zeros((N, 1, R, R, R), dtype=np.float32)
                centers = np.zeros((N, 3), dtype=np.float32)

                for pi, (z0, y0, x0) in enumerate(patches):
                    pred_nn_k_batch[pi, 0] = pred_nn_k_np[
                        z0:z0 + R, y0:y0 + R, x0:x0 + R]
                    pred_prev_batch[pi, 0] = (
                        refined_full_prev[z0:z0 + R, y0:y0 + R,
                                          x0:x0 + R].float().numpy())
                    cd_fg_batch[pi, 0] = compute_edt_patch(
                        (R, R, R), (z0, y0, x0),
                        fg_coords_full, clip=self.edt_clip)
                    cd_bg_batch[pi, 0] = compute_edt_patch(
                        (R, R, R), (z0, y0, x0),
                        bg_coords_full, clip=self.edt_clip)
                    gt_batch[pi, 0] = gt_oid_np[
                        z0:z0 + R, y0:y0 + R, x0:x0 + R].astype(np.float32)
                    centers[pi] = [
                        (z0 + R / 2) / D, (y0 + R / 2) / H, (x0 + R / 2) / W]

                pred_nn_k_t = torch.from_numpy(pred_nn_k_batch).to(device)
                pred_prev_t = torch.from_numpy(pred_prev_batch).to(device)
                cd_fg_t = torch.from_numpy(cd_fg_batch).to(device)
                cd_bg_t = torch.from_numpy(cd_bg_batch).to(device)
                gt_t = torch.from_numpy(gt_batch).to(device)
                centers_t = torch.from_numpy(centers).to(device)

                global_tokens_case = self.refiner.extract_global_tokens(
                    F_global.unsqueeze(0))                # (1, K, d)
                global_tokens_N = global_tokens_case.expand(N, -1, -1)
                if memory_tokens_list:
                    mem_stack = torch.stack(memory_tokens_list, 0)   # (T, d)
                    mem_tokens = mem_stack.unsqueeze(0).expand(N, -1, -1)
                else:
                    mem_tokens = None

                refined, delta = self.refiner(
                    pred_nn_k_t, pred_prev_t, cd_fg_t, cd_bg_t,
                    global_tokens_N, centers_t, mem_tokens)

                loss = self.criterion(refined, gt_t)
                total_loss_sum += loss.item()
                n_bwd += 1
                if not self.frozen:
                    (loss / max(1, K_obj * n_rounds)).backward()

                # Update state (detached, CPU)
                refined_det = refined.detach().cpu().half()   # (N, 1, R, R, R)
                for pi, (z0, y0, x0) in enumerate(patches):
                    refined_full_prev[z0:z0 + R, y0:y0 + R,
                                      x0:x0 + R] = refined_det[pi, 0]

                # Encode memory token from refined_full_prev → downsample to R^3
                mem_pred_low = F.interpolate(
                    refined_full_prev.unsqueeze(0).unsqueeze(0).float().to(device),
                    size=(R, R, R), mode='trilinear', align_corners=False)
                this_round_clicks = [
                    c for c in all_clicks_oid if c['round'] == round_idx]
                tok = self.refiner.encode_memory_token(
                    mem_pred_low, this_round_clicks, round_idx)
                memory_tokens_list.append(tok[0].detach())

        if not self.frozen and n_bwd > 0:
            nn.utils.clip_grad_norm_(self.refiner.parameters(), max_norm=5.0)
            self.optimizer.step()

        return total_loss_sum / max(1, n_bwd)

    # ------------------------------------------------------------------
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

            F_global = self.load_F_global(data['name'])
            if F_global is None:
                skipped += 1; continue

            try:
                loss = self._train_case(data, F_global)
                losses.append(loss)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM {data['name']}")
                skipped += 1; continue
            except Exception as e:
                print(f"  error {data['name']}: {e!r}")
                import traceback; traceback.print_exc()
                skipped += 1; continue

            if (fi + 1) % 10 == 0:
                dt = time.time() - t0
                rate = (fi + 1) / max(dt, 1e-6)
                eta = (len(pt_files) - fi - 1) / max(rate, 1e-6)
                print(f"  epoch {epoch}  [{fi+1}/{len(pt_files)}]  "
                      f"loss={np.mean(losses[-10:]):.4f}  "
                      f"{rate:.2f} case/s  eta {eta:.0f}s",
                      flush=True)

            del data, F_global
            torch.cuda.empty_cache()

        mean_loss = float(np.mean(losses)) if losses else float('nan')
        dt = time.time() - t0
        print(f"Epoch {epoch} done: mean_loss={mean_loss:.4f} "
              f"skipped={skipped} time={dt:.0f}s")
        return mean_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_v4_data')
    p.add_argument('--feats_dir', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_train_feats')
    p.add_argument('--output', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_v5')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--R', type=int, default=96)
    p.add_argument('--R_attn', type=int, default=48)
    p.add_argument('--num_global_queries', type=int, default=8)
    p.add_argument('--n_boundary', type=int, default=2)
    p.add_argument('--n_random', type=int, default=2)
    p.add_argument('--edt_clip', type=int, default=20)
    p.add_argument('--max_cases', type=int, default=0)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--finetune', action='store_true',
                   help='Load model weights only (skip optimizer, reset epoch to 0)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pt_files = sorted(args.data_dir.glob('*.pt'))
    # Filter to those with F_global precomputed
    feats_stems = {p.stem for p in args.feats_dir.glob('*.pt')}
    pt_files = [p for p in pt_files if p.stem in feats_stems]
    if args.max_cases > 0:
        pt_files = pt_files[:args.max_cases]
    print(f"Train files: {len(pt_files)}  feats dir: {args.feats_dir}")

    trainer = RefinementV5Trainer(
        gpu=args.gpu, lr=args.lr, R=args.R, R_attn=args.R_attn,
        feats_dir=args.feats_dir,
        num_global_queries=args.num_global_queries,
        n_boundary=args.n_boundary, n_random=args.n_random,
        edt_clip=args.edt_clip)

    start_epoch = 0
    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        trainer.refiner.load_state_dict(ck['model'])
        if args.finetune:
            print(f"Fine-tune: loaded model from {args.resume}, fresh optimizer (lr={args.lr}), start epoch 0")
        else:
            trainer.optimizer.load_state_dict(ck['optim'])
            start_epoch = ck.get('epoch', 0) + 1
            print(f"Resumed from {args.resume} → epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        mean_loss = trainer.train_epoch(list(pt_files), epoch)
        out_p = args.output / f"epoch_{epoch:02d}.pt"
        torch.save({
            'model': trainer.refiner.state_dict(),
            'optim': trainer.optimizer.state_dict(),
            'epoch': epoch,
            'loss': mean_loss,
            'args': vars(args),
        }, out_p)
        print(f"Saved {out_p}")


if __name__ == '__main__':
    main()
