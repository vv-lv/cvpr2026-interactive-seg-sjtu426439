"""
Single-Object Attention Training — Phase 1 of v7+ plan.

Validates mask memory independently from v7's multi-object logic.
Each case uses one object (natural single-obj or randomly picked from multi-obj).

Key differences from run_decoder_attn.py:
- Single object per case: no assembly, no other_tokens
- Memory-based bypass: Round 0 (no history) → bypass, Round 1+ → active
- Loss only computed on active rounds (Round 1+)
- mask_snapshots passed to attention module

Usage:
    python -m training.run_single_obj_attn --num_files 1500 --epochs 20 --gpu 0
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

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

from training.single_object_attention import (
    SingleObjectAttentionModule, wrap_decoder_stage_single_obj,
)
from training.bottleneck_attention import (
    ROLE_SELF_FG, ROLE_SELF_BG,
)
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.interaction_sim import generate_point_blob, POINT_RADIUS
from training.run_bottleneck_attn import (
    PATCH_SIZE, CHECKPOINT_PATH, INTERACTION_DECAY_TRAIN,
    _extract_patch_single, _paste_patch_into_buffer,
    _compute_edt_safe_eval_style, _sample_coord_eval_style,
    generate_click_eval_style,
)
from training.dataset import preprocess_like_inference, augment_full


SINGLE_OBJ_DATASETS = [
    ('CT', 'CT_AirwayTree'), ('CT', 'CT_Aorta'), ('CT', 'CT_LungLesion'),
    ('CT', 'CT_PancreasTumor'), ('CT', 'CT_AdrenalTumor'), ('CT', 'CT_ColonTumor'),
    ('CT', 'CT_LiverTumor'), ('CT', 'CT_KidneyTumor'), ('CT', 'CT_COVID19-Infection'),
    ('CT', 'CT_Lungs'), ('CT', 'CT_LymphNode'),
    ('MRI', 'MR_LeftAtrium'), ('MRI', 'MR_ProstateT2'), ('MRI', 'MR_ProstateADC'),
    ('MRI', 'MR_ISLES_DWI'), ('MRI', 'MR_ISLES_ADC'), ('MRI', 'MR_CervicalCancer'),
]

MULTI_OBJ_DATASETS = [
    ('MRI', 'MR_BraTS-T1c'), ('MRI', 'MR_BraTS-T1n'),
    ('MRI', 'MR_BraTS-T2f'), ('MRI', 'MR_BraTS-T2w'),
    ('CT', 'CT_AMOS'), ('CT', 'CT_AbdomenAtlas'),
    ('MRI', 'MR_Heart_ACDC'), ('MRI', 'MR_HVSMR'),
    ('MRI', 'MR_HNTS-MRG_HeadTumor'),
    ('CT', 'CT_TotalSeg_organs'), ('CT', 'CT_TotalSeg_cardiac'),
]


def find_single_obj_files(data_root: str, max_per_dataset: int = 150,
                          max_total: int = 2000) -> list:
    files = []
    ds_counts = {}
    for modality, ds_name in SINGLE_OBJ_DATASETS + MULTI_OBJ_DATASETS:
        d = os.path.join(data_root, modality, ds_name)
        if not os.path.isdir(d):
            continue
        ds_files = sorted(f for f in os.listdir(d) if f.endswith('.npz'))
        random.shuffle(ds_files)
        n = min(len(ds_files), max_per_dataset)
        for f in ds_files[:n]:
            files.append(os.path.join(d, f))
        ds_counts[ds_name] = n

    random.shuffle(files)
    if len(files) > max_total:
        files = files[:max_total]

    desc = ', '.join(f'{d}:{n}' for d, n in
                     sorted(ds_counts.items(), key=lambda x: -x[1])[:10])
    if len(ds_counts) > 10:
        desc += f', ... +{len(ds_counts) - 10} more'
    print(f"Found {len(files)} files from {len(ds_counts)} datasets ({desc})")
    return files


def load_and_prepare_single_obj(npz_path: str, augment: bool = True):
    """Load a case and pick one label (random if multi-object)."""
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

    MAX_VOLUME = 50_000_000  # ~370³, skip extremely large cases for speed
    if np.prod(image_crop.shape) > MAX_VOLUME:
        return None, None, None

    if augment:
        image_crop, gt_crop = augment_full(image_crop, gt_crop)
        if (gt_crop == label).sum() == 0:
            return None, None, None

    return image_crop, gt_crop, label


class SingleObjAttentionTrainer:

    def __init__(self, gpu: int = 0, lr: float = 3e-4,
                 num_rounds: int = 4, stage_idx: int = 1,
                 internal_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4,
                 lora_rank: int = 4, lora_stages: str = '2,3',
                 lora_lr_scale: float = 0.1):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.lora_rank = lora_rank

        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        stage_configs = {0: (320, 12), 1: (256, 24), 2: (128, 48),
                         3: (64, 96), 4: (32, 192)}
        input_dim, spatial_size = stage_configs[stage_idx]
        print(f"Insert at decoder stage {stage_idx}: ({input_dim}ch, {spatial_size}³)")

        self.attention = SingleObjectAttentionModule(
            input_dim=input_dim, spatial_size=spatial_size,
            internal_dim=internal_dim, num_layers=num_layers,
            num_heads=num_heads,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.attention.parameters())
        print(f"Attention module: {n_params:,} trainable params")
        print(f"Training rounds: {num_rounds}")

        self.stage_wrapper = wrap_decoder_stage_single_obj(
            self.network.decoder, stage_idx=stage_idx, attention=self.attention)

        # LoRA on decoder stages after attention injection
        self.lora_params = []
        self._set_lora_bypass = lambda b: None
        if lora_rank > 0:
            from training.lora import apply_lora_to_decoder, get_lora_params, set_lora_bypass
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
            {'params': list(self.attention.parameters()),
             'lr': lr, 'weight_decay': 1e-4},
        ]
        if self.lora_params:
            param_groups.append({
                'params': self.lora_params,
                'lr': lr * lora_lr_scale, 'weight_decay': 1e-4,
            })
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scaler = GradScaler()

    def train_epoch(self, files: list, epoch: int, total_epochs: int):
        self.attention.train()
        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0

        for fi, fpath in enumerate(files):
            image_crop, gt_crop, label = load_and_prepare_single_obj(fpath)
            if image_crop is None:
                skipped += 1
                continue

            step_loss = self._train_step(image_crop, gt_crop, label)
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
            # ── Generate click ──
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

            # ── Extract patch ──
            patch_center = click_hist[-1]['pos_image']
            patch_start = [patch_center[d] - PATCH_SIZE // 2 for d in range(3)]

            image_patch_np = _extract_patch_single(image_crop, patch_center)
            gt_patch_np = _extract_patch_single(gt_binary, patch_center)

            # ── Interaction channels ──
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
                blob = blob * decay
                ch = 3 if c['is_fg'] else 4
                interactions[ch] = np.maximum(interactions[ch], blob)

            input_8ch = torch.from_numpy(
                np.concatenate([image_patch_np[None], interactions], axis=0)[None]
            ).to(device)

            # ── Token info (self clicks only) ──
            self_tokens = []
            for c in click_hist:
                cp_norm = [(c['pos_image'][d] - patch_start[d]) / PATCH_SIZE
                           for d in range(3)]
                cp_norm = [max(0.0, min(1.0, x)) for x in cp_norm]
                self_tokens.append({
                    'pos': torch.tensor(cp_norm, dtype=torch.float32),
                    'role': ROLE_SELF_FG if c['is_fg'] else ROLE_SELF_BG,
                    'round': c['round'],
                })
            token_info = {'clicks': self_tokens}

            # ── Set wrapper state: pass history (rounds before current) ──
            history = mask_snapshots[:round_idx]
            self.stage_wrapper.set_state(token_info, mask_snapshots=history)
            self.stage_wrapper._bypass = False

            # ── Forward ──
            is_active = round_idx > 0 and len(history) > 0
            self._set_lora_bypass(not is_active)

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

            # ── Extract prediction + update buffers ──
            with torch.no_grad():
                out_full = outputs[0][0]  # (C, D, H, W)
                pred_patch = (out_full.argmax(0) == 1).cpu().numpy().astype(
                    np.uint8)

                if prev_pred is not None:
                    pred_buffer = prev_pred.copy()
                else:
                    pred_buffer = np.zeros(full_shape, dtype=np.uint8)
                _paste_patch_into_buffer(pred_buffer, pred_patch, patch_center)
                prev_pred = pred_buffer

                snap_prob = torch.softmax(
                    out_full.float(), dim=0)[1].cpu().numpy().astype(np.float16)
                _paste_patch_into_buffer(full_mask, snap_prob, patch_center)

            mask_snapshots.append(full_mask.copy())

        # ── Optimizer step ──
        if n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            all_trainable = list(self.attention.parameters()) + self.lora_params
            nn.utils.clip_grad_norm_(
                [p for p in all_trainable if p.requires_grad],
                max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from training.lora import LoRAConv3d
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()
        torch.save({
            'attention_state_dict': self.attention.state_dict(),
            'lora_state_dict': lora_state,
            'lora_rank': self.lora_rank,
            'epoch': epoch,
            'loss': loss,
        }, path)
        print(f"Saved: {path} (attn + {len(lora_state)} lora tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=1500)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--stage_idx', type=int, default=1)
    parser.add_argument('--internal_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_stages', default='2,3')
    parser.add_argument('--lora_lr_scale', type=float, default=0.1)
    parser.add_argument('--max_per_dataset', type=int, default=150)
    parser.add_argument('--save_dir',
                        default='experiments/single_obj_attn')
    args = parser.parse_args()

    files = find_single_obj_files(
        args.data_root, max_per_dataset=args.max_per_dataset,
        max_total=args.num_files)
    if not files:
        print("No files found!")
        return

    trainer = SingleObjAttentionTrainer(
        gpu=args.gpu, lr=args.lr, num_rounds=args.num_rounds,
        stage_idx=args.stage_idx, internal_dim=args.internal_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        lora_rank=args.lora_rank, lora_stages=args.lora_stages,
        lora_lr_scale=args.lora_lr_scale,
    )

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch, args.epochs)
        trainer.save_checkpoint(
            os.path.join(args.save_dir, f'epoch_{epoch}.pth'), epoch, loss)
        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(
                os.path.join(args.save_dir, 'best.pth'), epoch, loss)

    trainer.save_checkpoint(
        os.path.join(args.save_dir, 'final.pth'), args.epochs - 1, loss)
    print("Done!")


if __name__ == '__main__':
    main()
