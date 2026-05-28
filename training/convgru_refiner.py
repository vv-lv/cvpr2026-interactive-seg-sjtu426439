#!/usr/bin/env python3
"""
ConvGRU Refiner: Cross-round memory with VISTA3D spatial context.

Solves per-object prediction quality (regression, early convergence, smoothing).
Multi-object competition is handled separately by sigmoid assembly.

Architecture:
    Per round t, per object k:
      soft_pred_t = sigmoid(margin_logit)                      (1ch)
      vista_compressed = compress(VISTA3D_features)            (16ch)
      x_t = concat(soft_pred_t, vista_compressed)              (17ch)
      h_t = ConvGRU(x_t, h_{t-1})                             (hidden_ch)
      delta = residual_scale * tanh(output_head(h_t))          (1ch)
      refined_t = clamp(soft_pred_t + delta, 0, 1)

Training (multi-round unrolling):
    python -u training/convgru_refiner.py --epochs 30

Eval (offline, on precomputed crops):
    python -u training/convgru_refiner.py --mode eval --checkpoint experiments/convgru_refiner/best.pth
"""
import argparse
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

DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "gru_refiner_crops"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "convgru_refiner"

CROP_MARGIN = 8
MIN_CROP_SIZE = 16
MAX_CROP_SIZE = 128
MAX_OBJECTS = 6


# ─── ConvGRU Cell ────────────────────────────────────────────────────────────
class ConvGRUCell3D(nn.Module):
    """3D Convolutional GRU cell."""

    def __init__(self, input_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel_size // 2
        self.reset_gate = nn.Conv3d(input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)
        self.update_gate = nn.Conv3d(input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)
        self.candidate = nn.Conv3d(input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h):
        """
        Args:
            x: (B, input_ch, D, H, W)
            h: (B, hidden_ch, D, H, W)
        Returns:
            h_new: (B, hidden_ch, D, H, W)
        """
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        combined_r = torch.cat([x, r * h], dim=1)
        h_hat = torch.tanh(self.candidate(combined_r))
        h_new = (1 - z) * h + z * h_hat
        return h_new


# ─── ConvGRU Refiner ─────────────────────────────────────────────────────────
class ConvGRURefiner(nn.Module):
    """Cross-round refiner: improves per-object predictions using temporal
    memory and VISTA3D spatial context.

    Input per round: soft_pred (1ch) + compressed VISTA3D features (compress_ch)
    Output: residually refined prediction (1ch)
    """

    def __init__(self, encoder_ch=192, compress_ch=16, hidden_ch=16,
                 residual_scale=0.1):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.residual_scale = residual_scale

        # Compress VISTA3D features: 192 → compress_ch
        self.compress = nn.Sequential(
            nn.Conv3d(encoder_ch, compress_ch, 1),
            nn.InstanceNorm3d(compress_ch),
            nn.LeakyReLU(inplace=True),
        )
        # ConvGRU: input = soft_pred (1ch) + compressed vista (compress_ch)
        input_ch = 1 + compress_ch
        self.gru = ConvGRUCell3D(input_ch, hidden_ch)

        # Output: hidden → 1ch delta
        self.output_head = nn.Conv3d(hidden_ch, 1, 1)

    def compress_vista(self, enc_feat, target_size):
        """Compress and upsample VISTA3D features to target resolution.

        Args:
            enc_feat: (B, 192, D', H', W') — VISTA3D Stage 2 features
            target_size: (D, H, W) — full resolution

        Returns:
            (B, compress_ch, D, H, W)
        """
        compressed = self.compress(enc_feat)
        if compressed.shape[2:] != target_size:
            compressed = F.interpolate(
                compressed, size=target_size, mode='trilinear', align_corners=False)
        return compressed

    def forward_step(self, soft_pred, vista_compressed, h_prev):
        """One GRU step.

        Args:
            soft_pred: (B, 1, D, H, W) — sigmoid of margin logit, [0,1]
            vista_compressed: (B, compress_ch, D, H, W) — pre-compressed
            h_prev: (B, hidden_ch, D, H, W) or None

        Returns:
            refined: (B, 1, D, H, W)
            h_new: (B, hidden_ch, D, H, W)
        """
        B = soft_pred.shape[0]
        spatial = soft_pred.shape[2:]

        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_ch, *spatial,
                                 device=soft_pred.device, dtype=soft_pred.dtype)

        x = torch.cat([soft_pred, vista_compressed], dim=1)
        h_new = self.gru(x, h_prev)
        delta = self.residual_scale * torch.tanh(self.output_head(h_new))
        refined = torch.clamp(soft_pred + delta, 0, 1)
        return refined, h_new


# ─── Loss functions ──────────────────────────────────────────────────────────
def dice_loss(pred, target, smooth=1.0):
    """Soft Dice loss. pred and target in [0, 1]."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(refined, gt_binary):
    """BCE + Dice loss for per-object binary prediction."""
    # Disable autocast: BCE is unsafe under fp16 autocast
    with torch.amp.autocast('cuda', enabled=False):
        r32 = refined.float()
        g32 = gt_binary.float()
        bce = F.binary_cross_entropy(r32, g32, reduction='mean')
        dl = dice_loss(r32, g32)
    return bce + dl


# ─── Dataset ─────────────────────────────────────────────────────────────────
class GRUCropDataset(Dataset):
    """Loads multi-round cropped data for ConvGRU training.

    Each file contains:
        'per_round_margins': list of dicts {oid: margin_fp16 (d,h,w)}
        'enc_feat_crop': (192, d', h', w') fp16
        'gt_crop': (d, h, w) int64
        'oids': list of int
        'n_rounds': int
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.pt"))
        print(f"GRUCropDataset: {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location='cpu', weights_only=False)
        except Exception:
            return None
        if 'per_round_margins' not in data or len(data.get('oids', [])) < 2:
            return None
        return data


# ─── Training step ───────────────────────────────────────────────────────────
def train_step(refiner, batch, optimizer, scaler, device):
    """One training step: unroll GRU across all rounds for all objects.

    Uses per-round gradient accumulation to limit GPU memory:
    backward() is called after each round, so only one round's
    computation graph is alive at a time.

    Returns:
        metrics: dict with per-round and per-object stats
    """
    per_round_margins = batch['per_round_margins']
    enc_feat = batch['enc_feat_crop']
    gt = batch['gt_crop']
    oids = batch['oids']
    n_rounds = len(per_round_margins)
    K = len(oids)

    if K < 2 or n_rounds < 2:
        return None

    # Limit objects
    if K > MAX_OBJECTS:
        import random
        selected = sorted(random.sample(range(K), MAX_OBJECTS))
        oids = [oids[i] for i in selected]
        K = MAX_OBJECTS

    # Get spatial shape from first round, first object
    first_oid = oids[0]
    first_margin = per_round_margins[0][first_oid]
    if isinstance(first_margin, np.ndarray):
        spatial = first_margin.shape
    else:
        spatial = tuple(first_margin.shape)

    # Prepare encoder features on device (recompute compress per round for gradient flow)
    if isinstance(enc_feat, np.ndarray):
        enc_feat = torch.from_numpy(enc_feat.astype(np.float32))
    enc_feat_batch = enc_feat.float().unsqueeze(0).to(device)

    # Pre-compute GT on device
    if isinstance(gt, torch.Tensor):
        gt_dev = gt.to(device)
    else:
        gt_dev = torch.from_numpy(gt.astype(np.int64)).to(device)

    optimizer.zero_grad()

    h_states = {oid: None for oid in oids}
    round_losses = []
    total_loss_val = 0.0
    n_loss_terms = 0

    for r in range(n_rounds):
        # Recompute compress each round: allows gradient flow to compress
        # weights while freeing graph after per-round backward()
        with torch.autocast('cuda', enabled=True):
            vista_compressed = refiner.compress_vista(enc_feat_batch, spatial)

        round_margins = per_round_margins[r]
        round_loss = torch.tensor(0.0, device=device)
        round_k = 0

        for oid in oids:
            if oid not in round_margins:
                continue

            margin = round_margins[oid]
            if isinstance(margin, np.ndarray):
                margin = torch.from_numpy(margin.astype(np.float32))
            else:
                margin = margin.float()
            soft_pred = torch.sigmoid(margin).unsqueeze(0).unsqueeze(0).to(device)

            with torch.autocast('cuda', enabled=True):
                refined, h_new = refiner.forward_step(
                    soft_pred, vista_compressed, h_states[oid])
            h_states[oid] = h_new.detach()

            gt_binary = (gt_dev == oid).float().unsqueeze(0).unsqueeze(0)
            obj_loss = combined_loss(refined, gt_binary)
            round_loss = round_loss + obj_loss
            round_k += 1

            del soft_pred, refined, gt_binary

        if round_k > 0:
            # Backward per round to free computation graph
            avg_round = round_loss / (round_k * n_rounds)
            scaler.scale(avg_round).backward()
            round_losses.append(round_loss.item() / round_k)
            total_loss_val += round_loss.item()
            n_loss_terms += round_k

    scaler.step(optimizer)
    scaler.update()

    metrics = {
        'loss': total_loss_val / max(n_loss_terms, 1),
        'round_losses': round_losses,
        'n_rounds': n_rounds,
        'n_objects': K,
    }
    return metrics


# ─── Evaluation step ─────────────────────────────────────────────────────────
@torch.no_grad()
def eval_step(refiner, batch, device):
    """Evaluate one case: compute per-round Dice with and without GRU."""
    per_round_margins = batch['per_round_margins']
    enc_feat = batch['enc_feat_crop']
    gt = batch['gt_crop']
    oids = batch['oids']
    n_rounds = len(per_round_margins)
    K = len(oids)

    if K < 2 or n_rounds < 2:
        return None

    first_oid = oids[0]
    first_margin = per_round_margins[0][first_oid]
    if isinstance(first_margin, np.ndarray):
        spatial = first_margin.shape
    else:
        spatial = tuple(first_margin.shape)

    if isinstance(enc_feat, np.ndarray):
        enc_feat = torch.from_numpy(enc_feat.astype(np.float32))
    enc_feat_batch = enc_feat.float().unsqueeze(0).to(device)

    if isinstance(gt, np.ndarray):
        gt_np = gt
    else:
        gt_np = gt.numpy()

    with torch.autocast('cuda', enabled=True):
        vista_compressed = refiner.compress_vista(enc_feat_batch, spatial)

    h_states = {oid: None for oid in oids}
    base_dices = []
    refined_dices = []

    for r in range(n_rounds):
        round_margins = per_round_margins[r]

        # Baseline: sigmoid assembly
        base_seg = np.zeros(spatial, dtype=np.uint8)
        best_sig = np.full(spatial, -np.inf, dtype=np.float32)

        # Refined: GRU + sigmoid assembly
        refined_seg = np.zeros(spatial, dtype=np.uint8)
        best_ref = np.full(spatial, -np.inf, dtype=np.float32)

        for oid in oids:
            if oid not in round_margins:
                continue

            margin = round_margins[oid]
            if isinstance(margin, np.ndarray):
                margin_np = margin.astype(np.float32)
                margin_t = torch.from_numpy(margin_np)
            else:
                margin_t = margin.float()
                margin_np = margin_t.numpy()

            # Baseline sigmoid
            sig_np = 1.0 / (1.0 + np.exp(-np.clip(margin_np, -20, 20)))
            fg = sig_np > 0.5
            new_only = fg & (base_seg == 0)
            overlap = fg & (base_seg > 0)
            base_seg[new_only] = oid
            if overlap.any():
                better = overlap & (sig_np > best_sig)
                base_seg[better] = oid
            best_sig[fg] = np.maximum(best_sig[fg], sig_np[fg])

            # GRU refinement
            soft_pred = torch.sigmoid(margin_t).unsqueeze(0).unsqueeze(0).to(device)
            with torch.autocast('cuda', enabled=True):
                refined, h_new = refiner.forward_step(soft_pred, vista_compressed, h_states[oid])
            h_states[oid] = h_new
            ref_np = refined[0, 0].float().cpu().numpy()

            fg_ref = ref_np > 0.5
            new_only_r = fg_ref & (refined_seg == 0)
            overlap_r = fg_ref & (refined_seg > 0)
            refined_seg[new_only_r] = oid
            if overlap_r.any():
                better_r = overlap_r & (ref_np > best_ref)
                refined_seg[better_r] = oid
            best_ref[fg_ref] = np.maximum(best_ref[fg_ref], ref_np[fg_ref])

        # Compute multi-class Dice
        base_dice = _mc_dice(gt_np, base_seg, oids)
        ref_dice = _mc_dice(gt_np, refined_seg, oids)
        base_dices.append(base_dice)
        refined_dices.append(ref_dice)

    return {
        'base_dices': base_dices,
        'refined_dices': refined_dices,
        'n_rounds': n_rounds,
        'n_objects': K,
        'name': batch.get('name', 'unknown'),
    }


def _mc_dice(gt, seg, oids):
    """Multi-class Dice."""
    dices = []
    for oid in oids:
        g = (gt == oid)
        s = (seg == oid)
        inter = (g & s).sum()
        total = g.sum() + s.sum()
        dices.append(2 * inter / max(total, 1))
    return np.mean(dices)


# ─── Main ────────────────────────────────────────────────────────────────────
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
    parser.add_argument("--hidden_ch", type=int, default=16)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = GRUCropDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=(args.mode == "train"),
                        num_workers=0, collate_fn=lambda x: x[0])

    refiner = ConvGRURefiner(
        encoder_ch=args.encoder_ch,
        compress_ch=args.compress_ch,
        hidden_ch=args.hidden_ch,
        residual_scale=args.residual_scale,
    ).to(device)

    n_params = sum(p.numel() for p in refiner.parameters())
    print(f"ConvGRURefiner: {n_params:,} params "
          f"(enc={args.encoder_ch}, compress={args.compress_ch}, "
          f"hidden={args.hidden_ch}, scale={args.residual_scale})")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        refiner.load_state_dict(ckpt['refiner_state'])
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.mode == "train":
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        print(f"\nTraining: {args.epochs} epochs, {len(dataset)} files")
        best_loss = float('inf')

        for epoch in range(args.epochs):
            refiner.train()
            losses = []
            round_loss_sums = {}
            t0 = time.time()

            for batch in loader:
                if batch is None:
                    continue

                # train_step handles optimizer.zero_grad, backward, step internally
                metrics = train_step(refiner, batch, optimizer, scaler, device)
                if metrics is None:
                    continue

                losses.append(metrics['loss'])
                for r, rl in enumerate(metrics['round_losses']):
                    if r not in round_loss_sums:
                        round_loss_sums[r] = []
                    round_loss_sums[r].append(rl)

            elapsed = time.time() - t0
            ml = np.mean(losses) if losses else float('nan')

            # Per-round loss summary
            rl_str = " ".join(
                f"R{r}={np.mean(vs):.3f}"
                for r, vs in sorted(round_loss_sums.items())
            )

            print(f"Epoch {epoch:02d}: loss={ml:.4f}  {rl_str}  "
                  f"({len(losses)} cases, {elapsed:.1f}s)")

            # Save best
            if ml < best_loss and not np.isnan(ml):
                best_loss = ml
                torch.save({
                    'epoch': epoch,
                    'refiner_state': refiner.state_dict(),
                    'loss': ml,
                    'config': {
                        'encoder_ch': args.encoder_ch,
                        'compress_ch': args.compress_ch,
                        'hidden_ch': args.hidden_ch,
                        'residual_scale': args.residual_scale,
                    },
                }, args.output / "best.pth")

            # Save periodic
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'refiner_state': refiner.state_dict(),
                    'loss': ml,
                    'config': {
                        'encoder_ch': args.encoder_ch,
                        'compress_ch': args.compress_ch,
                        'hidden_ch': args.hidden_ch,
                        'residual_scale': args.residual_scale,
                    },
                }, args.output / f"epoch{epoch:02d}.pth")

        # Save final
        torch.save({
            'refiner_state': refiner.state_dict(),
            'config': {
                'encoder_ch': args.encoder_ch,
                'compress_ch': args.compress_ch,
                'hidden_ch': args.hidden_ch,
                'residual_scale': args.residual_scale,
            },
        }, args.output / "final.pth")
        print(f"\nBest loss: {best_loss:.4f}")
        print(f"Saved to {args.output}")

    elif args.mode == "eval":
        refiner.eval()
        print(f"\nEvaluating: {len(dataset)} files")

        all_results = []
        for batch in loader:
            if batch is None:
                continue
            result = eval_step(refiner, batch, device)
            if result is None:
                continue
            all_results.append(result)

            bd = result['base_dices']
            rd = result['refined_dices']
            delta = [r - b for b, r in zip(bd, rd)]
            print(f"  {result['name'][:40]:40s}  base_final={bd[-1]:.3f}  "
                  f"ref_final={rd[-1]:.3f}  Δ={delta[-1]:+.3f}")

        if all_results:
            base_finals = [r['base_dices'][-1] for r in all_results]
            ref_finals = [r['refined_dices'][-1] for r in all_results]
            wins = sum(1 for b, r in zip(base_finals, ref_finals) if r > b + 0.01)
            losses = sum(1 for b, r in zip(base_finals, ref_finals) if r < b - 0.01)
            print(f"\nSummary ({len(all_results)} cases):")
            print(f"  Base final Dice:    {np.mean(base_finals):.4f}")
            print(f"  Refined final Dice: {np.mean(ref_finals):.4f}")
            print(f"  W/L (Δ>0.01): {wins}/{losses}")


if __name__ == "__main__":
    main()
