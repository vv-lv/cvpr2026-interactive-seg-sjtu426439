"""
Memory Stats Training — spatial hand-crafted statistics injected into decoder.

Minimal validation: MemoryStatsEncoder + LoRA, no click attention.
Tests whether mean/var/first_pred statistics from historical masks
can reduce regression.

Usage:
    python -m training.run_memory_stats --num_files 300 --epochs 5 --gpu 0
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

from training.memory_stats import (
    MemoryStatsEncoder, wrap_decoder_stage_memory_stats, compute_memory_stats,
)
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.interaction_sim import generate_point_blob, POINT_RADIUS
from training.run_bottleneck_attn import (
    PATCH_SIZE, CHECKPOINT_PATH, INTERACTION_DECAY_TRAIN,
    _extract_patch_single, _paste_patch_into_buffer,
    _compute_edt_safe_eval_style, _sample_coord_eval_style,
    generate_click_eval_style,
)
from training.run_single_obj_attn import (
    find_single_obj_files,
)
from training.dataset import preprocess_like_inference, augment_full

MAX_CLICK_GEN_SIZE = 192  # downsample to this for EDT/cc3d


def load_and_prepare_single_obj_fast(npz_path: str, augment: bool = True):
    """Load a case, pick one label. No MAX_VOLUME filter."""
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


def _downsample_for_click(mask, max_size=MAX_CLICK_GEN_SIZE):
    """Downsample a 3D mask if any dim > max_size. Returns (downsampled, scale_factors)."""
    shape = np.array(mask.shape, dtype=np.float64)
    if max(shape) <= max_size:
        return mask, np.ones(3)
    scale = max_size / shape.max()
    new_shape = np.round(shape * scale).astype(int)
    new_shape = np.maximum(new_shape, 1)
    from scipy.ndimage import zoom
    zoomed = zoom(mask.astype(np.float64), new_shape / shape, order=0)
    return zoomed.astype(np.uint8), shape / new_shape


def generate_click_fast(prev_pred, gt_binary):
    """Like generate_click_eval_style but downsamples large volumes first."""
    pred_ds, scale = _downsample_for_click(prev_pred)
    gt_ds, _ = _downsample_for_click(gt_binary)
    center_ds, is_fg = generate_click_eval_style(pred_ds, gt_ds)
    if center_ds is None:
        return None, None
    center = tuple(int(round(center_ds[d] * scale[d])) for d in range(3))
    center = tuple(min(max(c, 0), s - 1) for c, s in zip(center, prev_pred.shape))
    return center, is_fg


def generate_initial_click_fast(gt_binary):
    """Like EDT center click but downsamples large volumes."""
    gt_ds, scale = _downsample_for_click(gt_binary > 0)
    edt = _compute_edt_safe_eval_style(gt_ds)
    if edt.max() > 0:
        center_ds = _sample_coord_eval_style(edt * gt_ds)
    else:
        coords = np.argwhere(gt_ds > 0)
        if len(coords) == 0:
            return None
        center_ds = tuple(coords[len(coords) // 2])
    center = tuple(int(round(center_ds[d] * scale[d])) for d in range(3))
    center = tuple(min(max(c, 0), s - 1) for c, s in zip(center, gt_binary.shape))
    return center


class MemoryStatsTrainer:

    def __init__(self, gpu=0, lr=3e-4, num_rounds=4, stage_idx=1,
                 lora_rank=4, lora_stages='2,3', lora_lr_scale=0.1):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.lora_rank = lora_rank

        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device).eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        self.mem_encoder = MemoryStatsEncoder(in_channels=3, feat_dim=256).to(self.device)
        n_params = sum(p.numel() for p in self.mem_encoder.parameters())
        print(f"MemoryStatsEncoder: {n_params:,} params")

        self.stage_wrapper = wrap_decoder_stage_memory_stats(
            self.network.decoder, stage_idx=stage_idx,
            memory_encoder=self.mem_encoder)

        self.lora_params = []
        self._set_lora_bypass = lambda b: None
        if lora_rank > 0:
            from training.lora import (
                apply_lora_to_decoder, get_lora_params, set_lora_bypass)
            self._set_lora_bypass = lambda b: set_lora_bypass(self.network, b)
            target_stages = [int(s) for s in lora_stages.split(',') if s.strip()]
            n_lora = apply_lora_to_decoder(
                self.network.decoder, target_stages=target_stages,
                rank=lora_rank, alpha=1.0)
            self.network.decoder.to(self.device)
            self.lora_params = get_lora_params(self.network)
            for p in self.lora_params:
                p.requires_grad_(True)
            print(f"LoRA: {n_lora:,} params on stages {target_stages}, "
                  f"rank={lora_rank}, lr={lr * lora_lr_scale:.1e}")

        self.criterion = build_loss(deep_supervision=True).to(self.device)

        param_groups = [
            {'params': list(self.mem_encoder.parameters()),
             'lr': lr, 'weight_decay': 1e-4},
        ]
        if self.lora_params:
            param_groups.append({
                'params': self.lora_params,
                'lr': lr * lora_lr_scale, 'weight_decay': 1e-4,
            })
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scaler = GradScaler()

    def train_epoch(self, files, epoch, total_epochs):
        self.mem_encoder.train()
        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0

        for fi, fpath in enumerate(files):
            img, gt, label = load_and_prepare_single_obj_fast(fpath)
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

        full_mask = np.zeros(full_shape, dtype=np.float16)
        mask_snapshots = []
        prev_pred = None
        click_hist = []

        self.optimizer.zero_grad()
        total_loss_val = 0.0
        n_fwd = 0

        for round_idx in range(self.num_rounds):
            if round_idx == 0:
                center = generate_initial_click_fast(gt_binary)
                if center is None:
                    return None
                click_hist.append({'pos_image': center, 'is_fg': True, 'round': round_idx})
            else:
                if prev_pred is None:
                    continue
                center, is_fg = generate_click_fast(
                    prev_pred, (gt_binary > 0).astype(np.uint8))
                if center is not None:
                    click_hist.append({
                        'pos_image': center, 'is_fg': bool(is_fg),
                        'round': round_idx})

            if not click_hist:
                continue

            patch_center = click_hist[-1]['pos_image']
            patch_start = [patch_center[d] - PATCH_SIZE // 2 for d in range(3)]

            image_patch_np = _extract_patch_single(image_crop, patch_center)
            gt_patch_np = _extract_patch_single(gt_binary, patch_center)

            interactions = np.zeros((7, *patch_shape), dtype=np.float32)
            if prev_pred is not None:
                interactions[0] = _extract_patch_single(
                    prev_pred.astype(np.float32), patch_center)
            n_clicks = len(click_hist)
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

            # Compute memory stats from history
            history = mask_snapshots[:round_idx]
            is_active = round_idx > 0 and len(history) > 0
            if is_active:
                stats = compute_memory_stats(history, patch_center)
                self.stage_wrapper.set_memory_stats(stats)
                self.stage_wrapper._bypass = False
                self._set_lora_bypass(False)
            else:
                self.stage_wrapper.set_memory_stats(None)
                self.stage_wrapper._bypass = True
                self._set_lora_bypass(True)

            if is_active:
                with autocast_ctx():
                    with torch.no_grad():
                        skips = self.network.encoder(input_8ch)
                    skips = [s.detach() for s in skips]
                    outputs = self.network.decoder(skips)
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
                        outputs = self.network.decoder(skips)

            with torch.no_grad():
                out_full = outputs[0][0]
                pred_patch = (out_full.argmax(0) == 1).cpu().numpy().astype(np.uint8)
                buf = prev_pred.copy() if prev_pred is not None else np.zeros(full_shape, dtype=np.uint8)
                _paste_patch_into_buffer(buf, pred_patch, patch_center)
                prev_pred = buf
                snap_prob = torch.softmax(
                    out_full.float(), dim=0)[1].cpu().numpy().astype(np.float16)
                _paste_patch_into_buffer(full_mask, snap_prob, patch_center)

            mask_snapshots.append(full_mask.copy())

        if n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            all_trainable = list(self.mem_encoder.parameters()) + self.lora_params
            nn.utils.clip_grad_norm_(
                [p for p in all_trainable if p.requires_grad], max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path, epoch, loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from training.lora import LoRAConv3d
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()
        torch.save({
            'mem_encoder_state_dict': self.mem_encoder.state_dict(),
            'lora_state_dict': lora_state,
            'lora_rank': self.lora_rank,
            'epoch': epoch, 'loss': loss,
        }, path)
        print(f"Saved: {path} ({len(lora_state)} lora tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_stages', default='2,3')
    parser.add_argument('--lora_lr_scale', type=float, default=0.1)
    parser.add_argument('--max_per_dataset', type=int, default=30)
    parser.add_argument('--save_dir', default='experiments/memory_stats_v1')
    args = parser.parse_args()

    files = find_single_obj_files(
        args.data_root, max_per_dataset=args.max_per_dataset,
        max_total=args.num_files)
    if not files:
        print("No files found!")
        return

    trainer = MemoryStatsTrainer(
        gpu=args.gpu, lr=args.lr, num_rounds=args.num_rounds,
        lora_rank=args.lora_rank, lora_stages=args.lora_stages,
        lora_lr_scale=args.lora_lr_scale)

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
