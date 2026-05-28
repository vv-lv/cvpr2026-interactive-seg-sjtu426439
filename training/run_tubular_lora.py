"""
Tubular structure LoRA training — single-object, Skeleton Recall Loss.

No attention module needed. Just LoRA on decoder stages + skeleton recall loss
to improve connectivity preservation for tubular structures (airways, vessels, aorta).

Trained LoRA weights are loaded separately from the multi-object attention LoRA,
and activated at inference when tubular structure is detected.

Usage:
    python -m training.run_tubular_lora \
        --train_json data/splits/tubular_train.json \
        --epochs 5 --lr 1e-4 --gpu 0 \
        --lora_rank 4 --lora_stages 2,3 \
        --lora_dropout 0.05 --lora_wd 0.05 --lora_norm_cap 0.04 \
        --skel_recall_weight 1.0 \
        --save_dir experiments/tubular_lora
"""
import argparse
import json
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

from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.lora import apply_lora_to_decoder, get_lora_params, LoRAConv3d
from training.skeleton_recall_loss import (
    precompute_tubed_skeleton_3d, SoftSkeletonRecallLoss,
)
from training.run_bottleneck_attn import (
    PATCH_SIZE, CHECKPOINT_PATH, preprocess_like_inference,
    _extract_patch_single, generate_point_blob,
    _compute_edt_safe_eval_style, _sample_coord_eval_style,
    POINT_RADIUS, INTERACTION_DECAY_TRAIN,
)
from training.dataset import augment_full
import math


class TubularLoRATrainer:

    def __init__(self, gpu=0, lr=1e-4, lora_rank=4, lora_stages='2,3',
                 lora_dropout=0.05, lora_wd=0.05, lora_norm_cap=0.04,
                 use_rslora=False, skel_recall_weight=1.0,
                 num_rounds=3):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.lora_norm_cap = lora_norm_cap
        self.skel_recall_weight = skel_recall_weight

        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        target_stages = [int(s) for s in lora_stages.split(',') if s.strip()]
        n_lora = apply_lora_to_decoder(
            self.network.decoder, target_stages=target_stages,
            rank=lora_rank, alpha=1.0,
            dropout=lora_dropout, use_rslora=use_rslora)
        self.network.decoder.to(self.device)
        self.lora_params = get_lora_params(self.network)
        for p in self.lora_params:
            p.requires_grad_(True)
        print(f"LoRA: {n_lora:,} params on stages {target_stages}, "
              f"rank={lora_rank}, dropout={lora_dropout}, rslora={use_rslora}")

        self.criterion = build_loss(deep_supervision=True).to(self.device)
        self.skel_loss = SoftSkeletonRecallLoss()

        self._lora_modules = {}
        self._frozen_conv_norms = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                stage_num = int(name.split('.stages.')[1].split('.')[0])
                self._lora_modules[stage_num] = module
                w_norm = module.original_conv.weight.float().norm().item()
                self._frozen_conv_norms[stage_num] = w_norm

        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=lr, weight_decay=lora_wd)
        self.scaler = GradScaler()

    def train_epoch(self, files, epoch, total_epochs):
        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0
        self._clamp_counts = {}

        for fi, fpath in enumerate(files):
            step_loss = self._train_step(fpath)
            if step_loss is not None:
                losses.append(step_loss)
            else:
                skipped += 1

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                mean_l = np.mean(losses[-50:]) if losses else 0
                print(f"  [{fi+1}/{len(files)}] loss={mean_l:.4f} "
                      f"skip={skipped} t={elapsed:.0f}s")

        elapsed = time.time() - t0
        mean_loss = np.mean(losses) if losses else 0
        print(f"Epoch {epoch}: loss={mean_loss:.4f} "
              f"n={len(losses)} skip={skipped} time={elapsed:.0f}s")

        for s, lora_mod in sorted(self._lora_modules.items()):
            A = lora_mod.lora_A.weight.detach()
            B = lora_mod.lora_B.weight.detach()
            A_2d = A.reshape(A.shape[0], -1)
            B_2d = B.reshape(B.shape[0], B.shape[1])
            ba_norm = (B_2d @ A_2d).norm().item() * (lora_mod.alpha / (math.sqrt(lora_mod.rank) if lora_mod.use_rslora else lora_mod.rank))
            w_norm = self._frozen_conv_norms[s]
            ratio = ba_norm / w_norm * 100
            clamps = self._clamp_counts.get(s, 0)
            print(f"  LoRA stage {s}: ||ΔW||/||W||={ratio:.2f}%, "
                  f"clamp_count={clamps}")

        return mean_loss

    def _train_step(self, fpath):
        device = self.device
        patch_shape = (PATCH_SIZE,) * 3

        try:
            data = np.load(fpath, allow_pickle=True)
            image = data['imgs'].astype(np.float32)
            gt = data['gts'].astype(np.uint8)
        except Exception:
            return None

        labels = [l for l in np.unique(gt) if l > 0]
        if len(labels) == 0:
            return None

        image_crop, gt_crop, bbox_min = preprocess_like_inference(image, gt)
        if image_crop is None:
            return None

        image_crop, gt_crop = augment_full(image_crop, gt_crop)
        crop_labels = [l for l in np.unique(gt_crop) if l > 0]
        if len(crop_labels) == 0:
            return None

        # Pick one label (random for multi-object, only one for single)
        label = random.choice(crop_labels)
        gt_binary = (gt_crop == label).astype(np.float32)

        if gt_binary.sum() < 50:
            return None

        # Precompute skeleton
        skel = precompute_tubed_skeleton_3d(gt_binary.astype(np.uint8))
        has_skel = skel.sum() > 0

        self.optimizer.zero_grad()
        total_loss_val = 0.0
        n_fwd = 0

        # Simulate click interaction rounds
        prev_pred = None
        for round_idx in range(self.num_rounds):
            # Generate click
            if round_idx == 0:
                edt = _compute_edt_safe_eval_style(gt_binary > 0)
                if edt.max() > 0:
                    center = _sample_coord_eval_style(edt * (gt_binary > 0))
                else:
                    coords = np.argwhere(gt_binary > 0)
                    center = tuple(coords[len(coords) // 2])
            else:
                if prev_pred is not None:
                    error = ((prev_pred > 0) != (gt_binary > 0)).astype(np.uint8)
                    if error.sum() > 0:
                        edt = _compute_edt_safe_eval_style(error)
                        center = _sample_coord_eval_style(edt)
                    else:
                        continue
                else:
                    continue

            # Extract patch
            image_patch = _extract_patch_single(image_crop, center)
            gt_patch = _extract_patch_single(gt_binary, center)

            # Build interactions (8 channels)
            interactions = np.zeros((7, *patch_shape), dtype=np.float32)
            if prev_pred is not None:
                interactions[0] = _extract_patch_single(
                    prev_pred.astype(np.float32), center)

            # Click blob
            patch_start = [center[d] - PATCH_SIZE // 2 for d in range(3)]
            cp = [center[d] - patch_start[d] for d in range(3)]
            if all(0 <= x < PATCH_SIZE for x in cp):
                blob = generate_point_blob(patch_shape, tuple(cp), POINT_RADIUS)
                interactions[1] = blob  # fg click channel

            input_8ch = np.concatenate([image_patch[None], interactions], axis=0)
            input_t = torch.from_numpy(input_8ch[None]).float().to(device)

            with autocast_ctx():
                with torch.no_grad():
                    skips = self.network.encoder(input_t)
                skips = [s.detach() for s in skips]
                outputs = self.network.decoder(skips)

                gt_t = torch.from_numpy(gt_patch[None, None]).float().to(device)
                targets = downsample_target_for_ds(gt_t)
                loss = self.criterion(outputs, targets)

                # Skeleton recall loss
                if has_skel and self.skel_recall_weight > 0:
                    skel_patch = _extract_patch_single(skel, center)
                    if skel_patch.sum() > 0:
                        skel_t = torch.from_numpy(
                            skel_patch[None, None]).float().to(device)
                        pred_sm = torch.softmax(outputs[0].float(), dim=1)
                        skel_l = self.skel_loss(pred_sm, skel_t)
                        loss = loss + self.skel_recall_weight * skel_l

            self.scaler.scale(loss / self.num_rounds).backward()
            total_loss_val += loss.item()
            n_fwd += 1

            # Update prev_pred
            with torch.no_grad():
                pred_np = (outputs[0].argmax(1)[0].cpu().numpy() > 0).astype(np.uint8)
                if prev_pred is None:
                    prev_pred = np.zeros(image_crop.shape, dtype=np.uint8)
                P = PATCH_SIZE
                for d in range(3):
                    lo_s = max(0, center[d] - P // 2)
                    hi_s = min(image_crop.shape[d], center[d] + P // 2)
                    lo_p = lo_s - (center[d] - P // 2)
                    hi_p = lo_p + (hi_s - lo_s)
                    if d == 0:
                        s0, p0 = slice(lo_s, hi_s), slice(lo_p, hi_p)
                    elif d == 1:
                        s1, p1 = slice(lo_s, hi_s), slice(lo_p, hi_p)
                    else:
                        s2, p2 = slice(lo_s, hi_s), slice(lo_p, hi_p)
                prev_pred[s0, s1, s2] = pred_np[p0, p1, p2]

        if n_fwd == 0:
            return None

        # Optimizer step
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.lora_params, 1.0)

        # Norm clamp
        if self.lora_norm_cap > 0:
            with torch.no_grad():
                for s, lora_mod in self._lora_modules.items():
                    A = lora_mod.lora_A.weight
                    B = lora_mod.lora_B.weight
                    A_2d = A.reshape(A.shape[0], -1)
                    B_2d = B.reshape(B.shape[0], B.shape[1])
                    divisor = math.sqrt(lora_mod.rank) if lora_mod.use_rslora else lora_mod.rank
                    ba_norm = (B_2d.float() @ A_2d.float()).norm().item() * (lora_mod.alpha / divisor)
                    w_norm = self._frozen_conv_norms[s]
                    ratio = ba_norm / w_norm
                    if ratio > self.lora_norm_cap:
                        scale = (self.lora_norm_cap / ratio) ** 0.5
                        A.mul_(scale)
                        B.mul_(scale)
                        self._clamp_counts[s] = self._clamp_counts.get(s, 0) + 1

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path, epoch, loss):
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.cpu()

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'lora_state_dict': lora_state,
            'lora_rank': self._lora_modules[list(self._lora_modules.keys())[0]].rank,
            'type': 'tubular_lora',
        }, path)
        print(f"Saved: {path} ({len(lora_state)} lora tensors)")

    def resume_from(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'lora_state_dict' in ckpt:
            for name, module in self.network.named_modules():
                if isinstance(module, LoRAConv3d):
                    a_key = f'{name}.lora_A.weight'
                    b_key = f'{name}.lora_B.weight'
                    if a_key in ckpt['lora_state_dict']:
                        module.lora_A.weight.data.copy_(
                            ckpt['lora_state_dict'][a_key])
                        module.lora_B.weight.data.copy_(
                            ckpt['lora_state_dict'][b_key])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {path} (epoch {ckpt.get('epoch', '?')})")
        return start_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_stages', default='2,3')
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_wd', type=float, default=0.05)
    parser.add_argument('--lora_norm_cap', type=float, default=0.04)
    parser.add_argument('--use_rslora', action='store_true')
    parser.add_argument('--skel_recall_weight', type=float, default=1.0)
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--save_dir', default='experiments/tubular_lora')
    args = parser.parse_args()

    with open(args.train_json) as f:
        train_data = json.load(f)
    files = [f['path'] for f in train_data['files'] if os.path.exists(f['path'])]
    print(f"Loaded {len(files)} training files")

    trainer = TubularLoRATrainer(
        gpu=args.gpu, lr=args.lr,
        lora_rank=args.lora_rank, lora_stages=args.lora_stages,
        lora_dropout=args.lora_dropout, lora_wd=args.lora_wd,
        lora_norm_cap=args.lora_norm_cap, use_rslora=args.use_rslora,
        skel_recall_weight=args.skel_recall_weight,
        num_rounds=args.num_rounds,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.resume_from(args.resume)

    for epoch in range(start_epoch, args.epochs):
        loss = trainer.train_epoch(files, epoch, args.epochs)
        ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch}.pth')
        trainer.save_checkpoint(ckpt_path, epoch, loss)


if __name__ == '__main__':
    main()
