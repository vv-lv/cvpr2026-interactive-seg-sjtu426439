#!/usr/bin/env python3
"""
HOQ-CA training script.

Trains the HOQ decoder on multi-class cases (BraTS) while keeping
nnInteractive encoder frozen.

Flow per training step:
  1. Load multi-class image + GT
  2. For each object: generate interactions → run frozen encoder → collect features
  3. Run HOQ decoder on collected features
  4. Compute multi-class loss (Dice + CE)
  5. Backward through HOQ decoder only

Usage:
  python training/train_hoq.py --gpu 0 --max_files 100 --epochs 20
"""
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
from torch.cuda.amp import GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# numpy compat (only needed for older numpy without _core)
try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.trainer import build_network, autocast_ctx
from training.interaction_sim import InteractionManager
from training.hoq_decoder import HOQDecoder, compute_multiclass_loss

DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_DATA = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "hoq_train"

PATCH_SIZE = 192
FEATURE_STAGE = 3  # 256ch @ 24³


def find_multiclass_files(data_dir, max_files=200):
    """Find all multi-class npz files (BraTS, AMOS, etc.)."""
    files = []
    for root, _, fnames in os.walk(data_dir):
        for f in sorted(fnames):
            if f.endswith('.npz'):
                files.append(os.path.join(root, f))

    # Filter for multi-class
    multi = []
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=True)
            nc = len(np.unique(d['gts'])) - 1
            if nc > 1:
                multi.append(fp)
        except Exception:
            continue
        if len(multi) >= max_files:
            break

    return multi


def random_crop_3d(image, gts, crop_size=PATCH_SIZE):
    """Random crop to (crop_size, crop_size, crop_size).
    If image is smaller, pad with zeros."""
    D, H, W = image.shape

    # Pad if needed
    pad_d = max(0, crop_size - D)
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        gts = np.pad(gts, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

    D, H, W = image.shape
    # Crop centered on foreground with probability 0.7
    fg_coords = np.argwhere(gts > 0)
    if len(fg_coords) > 0 and random.random() < 0.7:
        idx = random.randint(0, len(fg_coords) - 1)
        cz, cy, cx = fg_coords[idx]
        z0 = max(0, min(cz - crop_size // 2, D - crop_size))
        y0 = max(0, min(cy - crop_size // 2, H - crop_size))
        x0 = max(0, min(cx - crop_size // 2, W - crop_size))
    else:
        z0 = random.randint(0, max(0, D - crop_size))
        y0 = random.randint(0, max(0, H - crop_size))
        x0 = random.randint(0, max(0, W - crop_size))

    image_crop = image[z0:z0+crop_size, y0:y0+crop_size, x0:x0+crop_size]
    gts_crop = gts[z0:z0+crop_size, y0:y0+crop_size, x0:x0+crop_size]

    return image_crop, gts_crop


def normalize_image(image):
    """Z-score normalization (image-level, matching nnInteractive inference)."""
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    return image


def generate_per_object_interactions(gts_crop, num_classes, class_ids, shape):
    """Generate nnInteractive-style interactions for each object.

    Returns list of (7, D, H, W) numpy arrays, one per object.
    """
    interactions_list = []
    for cls_id in class_ids:
        gt_binary = (gts_crop == cls_id).astype(np.uint8)
        mgr = InteractionManager(shape)

        if gt_binary.sum() > 0:
            # Generate bbox
            mgr.set_initial_bbox(gt_binary, jitter=0.05)
        else:
            # Object not present in this crop — generate empty interactions
            pass

        interactions_list.append(mgr.get_numpy())

    return interactions_list


class HOQTrainer:
    def __init__(self, checkpoint_path, output_dir, device='cuda:0',
                 lr=1e-4, max_epochs=50, hidden_dim=256, num_layers=3):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build frozen nnInteractive
        print("Loading nnInteractive...")
        self.network, _ = build_network(checkpoint_path, deep_supervision=True)
        self.network = self.network.to(self.device)
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"nnInteractive: {total_params:,} params (all frozen)")

        # Build HOQ decoder
        self.hoq = HOQDecoder(
            feat_channels=256,  # stage 3
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=num_layers,
            max_objects=10,
        ).to(self.device)
        hoq_params = sum(p.numel() for p in self.hoq.parameters())
        print(f"HOQ decoder: {hoq_params:,} trainable params")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.hoq.parameters(), lr=lr, weight_decay=1e-4
        )
        self.lr = lr
        self.max_epochs = max_epochs
        self.grad_scaler = GradScaler()

    def extract_per_object_features(self, image_tensor, interactions_list):
        """Run frozen encoder for each object, return features.

        Args:
            image_tensor: (1, 1, D, H, W) on device
            interactions_list: list of N numpy arrays (7, D, H, W)

        Returns:
            features: list of N tensors (256, 24, 24, 24)
        """
        features = []
        with torch.no_grad():
            for inter_np in interactions_list:
                inter_t = torch.from_numpy(inter_np).unsqueeze(0).to(self.device)
                input_8ch = torch.cat([image_tensor, inter_t], dim=1)

                with autocast_ctx():
                    skips = self.network.encoder(input_8ch)
                    feat = skips[FEATURE_STAGE]  # (1, 256, 24, 24, 24)

                features.append(feat[0])  # remove batch dim
        return features

    def train_step(self, image_np, gts_np):
        """One training step on a multi-class sample.

        Returns: loss value, num_classes
        """
        # Crop
        image_crop, gts_crop = random_crop_3d(image_np, gts_np, PATCH_SIZE)

        # Get class IDs present in this crop
        class_ids = sorted([c for c in np.unique(gts_crop) if c > 0])
        if len(class_ids) < 2:
            return None, 0  # skip single-class crops

        num_classes = len(class_ids)
        shape = (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

        # Normalize image
        image_norm = normalize_image(image_crop.astype(np.float32))
        image_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # Generate per-object interactions
        interactions_list = generate_per_object_interactions(
            gts_crop, num_classes, class_ids, shape
        )

        # Extract features (frozen encoder)
        per_obj_features = self.extract_per_object_features(image_tensor, interactions_list)

        # Remap GT classes to 1..N for loss
        gts_remapped = np.zeros_like(gts_crop)
        for new_id, old_id in enumerate(class_ids, start=1):
            gts_remapped[gts_crop == old_id] = new_id
        gts_tensor = torch.from_numpy(gts_remapped).to(self.device)

        # Forward through HOQ decoder
        self.optimizer.zero_grad()
        with autocast_ctx():
            multi_logits = self.hoq(per_obj_features, num_classes)
            feat_shape = multi_logits.shape[1:]  # (D', H', W')
            loss = compute_multiclass_loss(multi_logits, gts_tensor, feat_shape)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.hoq.parameters(), max_norm=5.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item(), num_classes

    def train(self, file_list, num_epochs):
        print(f"\nTraining HOQ on {len(file_list)} multi-class files, {num_epochs} epochs")
        print(f"lr={self.lr}, device={self.device}")

        for epoch in range(num_epochs):
            random.shuffle(file_list)
            epoch_losses = []
            epoch_classes = []
            t0 = time.time()

            for fi, fp in enumerate(file_list):
                try:
                    data = np.load(fp, allow_pickle=True)
                    image = data['imgs']
                    gts = data['gts']

                    loss, nc = self.train_step(image, gts)
                    if loss is not None:
                        epoch_losses.append(loss)
                        epoch_classes.append(nc)

                except Exception as e:
                    print(f"  [ERROR] {os.path.basename(fp)}: {e}")
                    continue

                if (fi + 1) % 20 == 0:
                    recent = epoch_losses[-20:] if len(epoch_losses) >= 20 else epoch_losses
                    print(f"  [{fi+1}/{len(file_list)}] loss={np.mean(recent):.4f}")

            elapsed = time.time() - t0
            if epoch_losses:
                mean_loss = np.mean(epoch_losses)
                mean_nc = np.mean(epoch_classes)
                print(f"Epoch {epoch:03d}: loss={mean_loss:.4f}, "
                      f"samples={len(epoch_losses)}, mean_nc={mean_nc:.1f}, "
                      f"time={elapsed:.1f}s")
            else:
                print(f"Epoch {epoch:03d}: no valid samples")

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                path = self.output_dir / f"hoq_epoch{epoch:03d}.pth"
                torch.save({
                    'epoch': epoch,
                    'hoq_state_dict': self.hoq.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'loss': mean_loss if epoch_losses else None,
                }, path)
                print(f"  Saved: {path}")

            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Find multi-class files
    print(f"Scanning {args.data_dir} for multi-class files...")
    file_list = find_multiclass_files(args.data_dir, max_files=args.max_files)
    print(f"Found {len(file_list)} multi-class files")

    if len(file_list) == 0:
        print("No multi-class files found!")
        return

    trainer = HOQTrainer(
        checkpoint_path=args.checkpoint,
        output_dir=str(args.output),
        device=device,
        lr=args.lr,
        max_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    trainer.train(file_list, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
