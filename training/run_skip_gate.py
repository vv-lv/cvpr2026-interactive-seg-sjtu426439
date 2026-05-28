"""
Skip Gate Training — gate high-resolution skip connections using prev_pred confidence.

Minimal experiment: only SkipGate on decoder stages 2,3,4 (48³/96³/192³).
No click attention, no memory, no LoRA (first validate gate mechanism alone).

Usage:
    python -m training.run_skip_gate --num_files 300 --epochs 3 --gpu 0
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.skip_gate import GatedDecoder
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.interaction_sim import generate_point_blob, POINT_RADIUS
from training.run_bottleneck_attn import (
    PATCH_SIZE, CHECKPOINT_PATH, INTERACTION_DECAY_TRAIN,
    _extract_patch_single, _paste_patch_into_buffer,
    _compute_edt_safe_eval_style, _sample_coord_eval_style,
    generate_click_eval_style,
)
from training.run_single_obj_attn import find_single_obj_files
from training.dataset import preprocess_like_inference, augment_full


def load_case(npz_path, augment=True):
    data = np.load(npz_path, allow_pickle=True)
    image = data['imgs'].astype(np.float32)
    gt = data['gts'].astype(np.uint8)
    labels = [int(l) for l in np.unique(gt) if l > 0]
    if not labels:
        return None, None, None
    label = random.choice(labels)
    image_crop, gt_crop, _ = preprocess_like_inference(image, gt)
    if (gt_crop == label).sum() == 0:
        return None, None, None
    if augment:
        image_crop, gt_crop = augment_full(image_crop, gt_crop)
        if (gt_crop == label).sum() == 0:
            return None, None, None
    return image_crop, gt_crop, label


class SkipGateTrainer:

    def __init__(self, gpu=0, lr=3e-4, num_rounds=4,
                 gate_stages=(2, 3, 4)):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds

        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device).eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        self.gated_decoder = GatedDecoder(
            self.network.decoder, gate_stages=list(gate_stages),
            gate_input_channels=1,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.gated_decoder.gates.parameters())
        print(f"SkipGate params: {n_params} on stages {list(gate_stages)}")
        print(f"Training rounds: {num_rounds}")

        self.criterion = build_loss(deep_supervision=True).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.gated_decoder.gates.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = GradScaler()

    def train_epoch(self, files, epoch, total_epochs):
        self.gated_decoder.gates.train()
        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0

        for fi, fpath in enumerate(files):
            img, gt, label = load_case(fpath)
            if img is None:
                skipped += 1
                continue
            step_loss = self._train_step(img, gt, label)
            if step_loss is not None:
                losses.append(step_loss)
            if (fi + 1) % 10 == 0:
                elapsed = time.time() - t0
                mean_l = np.mean(losses[-10:]) if losses else 0
                print(f"  [{fi+1}/{len(files)}] loss={mean_l:.4f} "
                      f"skip={skipped} t={elapsed:.0f}s")

        elapsed = time.time() - t0
        mean_loss = np.mean(losses) if losses else 0
        print(f"Epoch {epoch}: loss={mean_loss:.4f} "
              f"n={len(losses)} skip={skipped} time={elapsed:.0f}s")
        return mean_loss

    def _train_step(self, image_crop, gt_crop, label):
        device = self.device
        full_shape = image_crop.shape
        patch_shape = (PATCH_SIZE,) * 3
        gt_binary = (gt_crop == label).astype(np.float32)

        prev_pred = None
        prev_prob = None  # softmax prob for gating
        click_hist = []

        self.optimizer.zero_grad()
        total_loss_val = 0.0
        n_fwd = 0

        for round_idx in range(self.num_rounds):
            if round_idx == 0:
                edt = _compute_edt_safe_eval_style(gt_binary > 0)
                if edt.max() > 0:
                    center = _sample_coord_eval_style(edt * (gt_binary > 0))
                else:
                    coords = np.argwhere(gt_binary > 0)
                    if len(coords) == 0:
                        return None
                    center = tuple(coords[len(coords) // 2])
                click_hist.append({
                    'pos_image': center, 'is_fg': True, 'round': round_idx})
            else:
                if prev_pred is None:
                    continue
                center, is_fg = generate_click_eval_style(
                    prev_pred, (gt_binary > 0).astype(np.uint8))
                if center is not None:
                    click_hist.append({
                        'pos_image': center, 'is_fg': bool(is_fg),
                        'round': round_idx})

            if not click_hist:
                continue

            patch_center = click_hist[-1]['pos_image']

            image_patch_np = _extract_patch_single(image_crop, patch_center)
            gt_patch_np = _extract_patch_single(gt_binary, patch_center)

            interactions = np.zeros((7, *patch_shape), dtype=np.float32)
            if prev_pred is not None:
                interactions[0] = _extract_patch_single(
                    prev_pred.astype(np.float32), patch_center)
            n_clicks = len(click_hist)
            patch_start = [patch_center[d] - PATCH_SIZE // 2 for d in range(3)]
            for ci, c in enumerate(click_hist):
                cp = [c['pos_image'][d] - patch_start[d] for d in range(3)]
                if not all(0 <= x < PATCH_SIZE for x in cp):
                    continue
                blob = generate_point_blob(patch_shape, tuple(cp), POINT_RADIUS)
                decay = INTERACTION_DECAY_TRAIN ** (n_clicks - 1 - ci)
                ch = 3 if c['is_fg'] else 4
                interactions[ch] = np.maximum(interactions[ch], blob * decay)

            input_8ch = torch.from_numpy(
                np.concatenate([image_patch_np[None], interactions], axis=0)[None]
            ).to(device)

            # Set gate input: prev_pred prob (or bypass for round 0)
            is_active = round_idx > 0 and prev_prob is not None
            if is_active:
                self.gated_decoder.set_prev_pred_prob(prev_prob, patch_center)
                self.gated_decoder._bypass = False
            else:
                self.gated_decoder._bypass = True

            if is_active:
                with autocast_ctx():
                    with torch.no_grad():
                        skips = self.network.encoder(input_8ch)
                    skips = [s.detach() for s in skips]
                    outputs = self.gated_decoder(skips)

                    gt_t = torch.from_numpy(
                        gt_patch_np[None, None]).float().to(device)
                    targets = downsample_target_for_ds(gt_t)
                    loss = self.criterion(outputs, targets)

                n_fwd += 1
                self.scaler.scale(
                    loss / max(self.num_rounds - 1, 1)).backward()
                total_loss_val += loss.item()
            else:
                with torch.no_grad():
                    with autocast_ctx():
                        skips = self.network.encoder(input_8ch)
                        outputs = self.gated_decoder(skips)

            # Extract prediction + update prob buffer
            with torch.no_grad():
                out_full = outputs[0][0]
                pred_patch = (out_full.argmax(0) == 1).cpu().numpy().astype(np.uint8)
                prob_patch = torch.softmax(
                    out_full.float(), dim=0)[1].cpu().numpy().astype(np.float32)

                buf = prev_pred.copy() if prev_pred is not None else np.zeros(full_shape, dtype=np.uint8)
                _paste_patch_into_buffer(buf, pred_patch, patch_center)
                prev_pred = buf

                if prev_prob is None:
                    prev_prob = np.zeros(full_shape, dtype=np.float32)
                _paste_patch_into_buffer(prev_prob, prob_patch, patch_center)

        if n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.gated_decoder.gates.parameters(), max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path, epoch, loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'gates_state_dict': self.gated_decoder.gates.state_dict(),
            'gate_stages': list(self.gated_decoder.gate_stages),
            'epoch': epoch, 'loss': loss,
        }, path)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--gate_stages', default='2,3,4')
    parser.add_argument('--max_per_dataset', type=int, default=30)
    parser.add_argument('--save_dir', default='experiments/skip_gate_v1')
    args = parser.parse_args()

    gate_stages = [int(s) for s in args.gate_stages.split(',')]

    files = find_single_obj_files(
        args.data_root, max_per_dataset=args.max_per_dataset,
        max_total=args.num_files)
    if not files:
        print("No files found!")
        return

    trainer = SkipGateTrainer(
        gpu=args.gpu, lr=args.lr, num_rounds=args.num_rounds,
        gate_stages=gate_stages)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch, args.epochs)
        trainer.save_checkpoint(
            os.path.join(args.save_dir, f'epoch_{epoch}.pth'), epoch, loss)
        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(
                os.path.join(args.save_dir, 'best.pth'), epoch, loss)

    print("Done!")


if __name__ == '__main__':
    main()
