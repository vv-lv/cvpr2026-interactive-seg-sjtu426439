"""
Input Concat 训练脚本 — Plan A：将跨 object 信息作为额外 2 个输入通道。

核心思路：
- 原始 nnInteractive 用 8ch 输入（1 image + 7 interaction channels）
- 添加 2 个 competition-aware 通道：
  ch8: other_mask — 其他 object 当前预测的 binary mask（哪里已被占）
  ch9: competition_pressure — 其他 object 在每个 voxel 的 sigmoid 概率和（多强的竞争）
- Stem Conv3d(8→32) 扩展为 Conv3d(10→32)，新通道零初始化（初始行为=原始模型）
- 无 attention 模块，信息通过早期通道直接注入

可训练部分：
- Encoder stage 0: 完全解冻（~8.6K params）
- Encoder stages 1-2: LoRA rank=4
- Decoder stages 3-4: LoRA rank=4
- 其余全冻结

Usage:
    python -m training.run_input_concat --num_files 300 --epochs 15 --gpu 0
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

from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.interaction_sim import (
    InteractionManager, generate_point_blob, sample_point_from_error_region,
    POINT_RADIUS,
)
from training.run_bottleneck_attn import (
    find_brats_files, load_and_prepare, generate_initial_click,
    generate_followup_click, PATCH_SIZE, CHECKPOINT_PATH,
    _extract_patch_single, _paste_patch_to_full, INTERACTION_DECAY_TRAIN,
    generate_click_eval_style, _compute_edt_safe_eval_style,
    _sample_coord_eval_style, assemble_last_wins, assemble_max_prob,
    _paste_patch_into_buffer,
)
from training.lora import (
    LoRAConv3d, apply_lora_to_decoder, get_lora_params, _find_parent,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Stem expansion: Conv3d(8→32) → Conv3d(10→32)
# ═══════════════════════════════════════════════════════════════════════════════

def expand_stem(network: nn.Module, extra_channels: int = 2):
    """Expand the encoder stem from 8→10 input channels.

    Copies original 8-channel weights to the new conv, and zero-initializes
    the extra channels so that initial behavior is identical to the original model.

    The stem is at: network.encoder.stages[0][0].convs[0].all_modules[0]
    (first conv in the first block of encoder stage 0).
    But we need to find it more robustly by looking for the Conv3d(8, 32, ...).
    """
    # Find the stem conv (in_channels == 8)
    stem_conv = None
    stem_parent = None
    stem_attr = None

    for name, module in network.encoder.named_modules():
        if isinstance(module, nn.Conv3d) and module.in_channels == 8:
            stem_conv = module
            # Find parent
            parts = name.split('.')
            parent = network.encoder
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            stem_parent = parent
            stem_attr = parts[-1]
            break

    if stem_conv is None:
        raise RuntimeError("Cannot find stem Conv3d with in_channels=8")

    old_in = stem_conv.in_channels
    new_in = old_in + extra_channels
    out_ch = stem_conv.out_channels
    k = stem_conv.kernel_size
    s = stem_conv.stride
    p = stem_conv.padding

    print(f"Expanding stem: Conv3d({old_in}→{out_ch}) → Conv3d({new_in}→{out_ch})")

    new_conv = nn.Conv3d(new_in, out_ch, kernel_size=k, stride=s, padding=p,
                         bias=stem_conv.bias is not None)

    # Copy weights
    with torch.no_grad():
        new_conv.weight[:, :old_in] = stem_conv.weight.clone()
        new_conv.weight[:, old_in:] = 0.0  # zero-init extra channels
        if stem_conv.bias is not None:
            new_conv.bias.copy_(stem_conv.bias)

    # Replace
    if stem_attr.isdigit():
        stem_parent[int(stem_attr)] = new_conv
    else:
        setattr(stem_parent, stem_attr, new_conv)

    # Fix all_modules Sequential references (same pattern as LoRA)
    if hasattr(stem_parent, 'all_modules'):
        for idx, mod in enumerate(stem_parent.all_modules):
            if mod is stem_conv:
                stem_parent.all_modules[idx] = new_conv
                break

    print(f"  Stem weight shape: {list(new_conv.weight.shape)}")
    return new_conv


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA for encoder stages (mirrors apply_lora_to_decoder)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_lora_to_encoder(encoder: nn.Module, target_stages: list = None,
                          rank: int = 4, alpha: float = 1.0) -> int:
    """Apply LoRA to encoder's conv layers at specified stages.

    Same logic as apply_lora_to_decoder but operates on encoder.stages.

    Args:
        encoder: UNetEncoder module (network.encoder)
        target_stages: stage indices to add LoRA, e.g. [1, 2]
        rank: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        total new LoRA parameter count
    """
    if target_stages is None:
        target_stages = [1, 2]

    total_params = 0

    for stage_idx in target_stages:
        if stage_idx >= len(encoder.stages):
            continue

        stage = encoder.stages[stage_idx]

        # Collect target conv layers (kernel_size > 1)
        targets = []
        for block_name, block in stage.named_modules():
            if isinstance(block, nn.Conv3d) and block.kernel_size[0] > 1:
                parent, attr = _find_parent(stage, block_name, block)
                if parent is not None:
                    targets.append((block_name, block, parent, attr))

        # Replace with LoRA
        for block_name, block, parent, attr in targets:
            lora_conv = LoRAConv3d(block, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_conv)

            # Fix all_modules Sequential references
            if hasattr(parent, 'all_modules'):
                for idx, mod in enumerate(parent.all_modules):
                    if mod is block:
                        parent.all_modules[idx] = lora_conv
                        break

            n = sum(p.numel() for p in [lora_conv.lora_A.weight,
                                         lora_conv.lora_B.weight])
            total_params += n
            print(f"  LoRA added: encoder.stages.{stage_idx}.{block_name} "
                  f"({list(block.weight.shape)}) → rank={rank}, +{n} params")

    return total_params


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class InputConcatTrainer:

    def __init__(self, gpu: int = 0, lr: float = 3e-4,
                 num_rounds: int = 4,
                 lora_rank: int = 4,
                 enc_lora_stages: str = '1,2',
                 dec_lora_stages: str = '3,4',
                 lora_lr_scale: float = 0.3,
                 stem_lr_scale: float = 1.0,
                 stage0_lr_scale: float = 0.3,
                 assembly: str = 'maxprob'):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.lora_rank = lora_rank
        assert assembly in ('lastwins', 'maxprob'), f"unknown assembly: {assembly}"
        self.assembly = assembly
        print(f"Between-round assembly: {assembly}")

        # 1. Build network (all frozen initially)
        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        # 2. Expand stem: 8ch → 10ch
        self.stem_conv = expand_stem(self.network, extra_channels=2)
        self.stem_conv.to(self.device)
        # Stem params are trainable
        for p in self.stem_conv.parameters():
            p.requires_grad_(True)
        stem_params = sum(p.numel() for p in self.stem_conv.parameters() if p.requires_grad)
        print(f"Stem trainable params: {stem_params:,}")

        # 3. Unfreeze encoder stage 0 entirely
        stage0 = self.network.encoder.stages[0]
        for p in stage0.parameters():
            p.requires_grad_(True)
        stage0_params = sum(p.numel() for p in stage0.parameters() if p.requires_grad)
        print(f"Encoder stage 0 unfrozen: {stage0_params:,} params")

        # 4. LoRA on encoder stages
        self.lora_params = []
        enc_stages = [int(s) for s in enc_lora_stages.split(',') if s.strip()]
        if lora_rank > 0 and enc_stages:
            n_enc_lora = apply_lora_to_encoder(
                self.network.encoder, target_stages=enc_stages,
                rank=lora_rank, alpha=1.0)
            self.network.encoder.to(self.device)
            print(f"Encoder LoRA: {n_enc_lora:,} params on stages {enc_stages}")

        # 5. LoRA on decoder stages
        dec_stages = [int(s) for s in dec_lora_stages.split(',') if s.strip()]
        if lora_rank > 0 and dec_stages:
            n_dec_lora = apply_lora_to_decoder(
                self.network.decoder, target_stages=dec_stages,
                rank=lora_rank, alpha=1.0)
            self.network.decoder.to(self.device)
            print(f"Decoder LoRA: {n_dec_lora:,} params on stages {dec_stages}")

        # Collect all LoRA params from both encoder and decoder
        self.lora_params = get_lora_params(self.network)
        for p in self.lora_params:
            p.requires_grad_(True)
        total_lora = sum(p.numel() for p in self.lora_params)
        print(f"Total LoRA params: {total_lora:,}, rank={lora_rank}")

        # 6. Loss
        self.criterion = build_loss(deep_supervision=True).to(self.device)

        # 7. Optimizer — separate param groups for stem, stage0, LoRA
        # Stem params (the new Conv3d including original 8ch weights)
        stem_param_list = list(self.stem_conv.parameters())

        # Stage 0 params (excluding the stem conv which is already in stem group)
        stage0_params_list = []
        stem_param_ids = {id(p) for p in stem_param_list}
        for p in stage0.parameters():
            if p.requires_grad and id(p) not in stem_param_ids:
                stage0_params_list.append(p)

        param_groups = [
            {'params': stem_param_list,
             'lr': lr * stem_lr_scale, 'weight_decay': 1e-4},
        ]
        if stage0_params_list:
            param_groups.append({
                'params': stage0_params_list,
                'lr': lr * stage0_lr_scale, 'weight_decay': 1e-4,
            })
        if self.lora_params:
            param_groups.append({
                'params': self.lora_params,
                'lr': lr * lora_lr_scale, 'weight_decay': 1e-4,
            })

        total_trainable = sum(
            p.numel() for pg in param_groups for p in pg['params'])
        print(f"Total trainable params: {total_trainable:,}")

        self.optimizer = torch.optim.AdamW(param_groups)
        self.scaler = GradScaler()

    def train_epoch(self, files: list, epoch: int, total_epochs: int):
        # Put trainable parts in train mode
        self.network.encoder.stages[0].train()
        # LoRA modules are in eval parent but LoRA itself has no BN, so fine

        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0

        for fi, fpath in enumerate(files):
            image_patch, gt_patch, labels = load_and_prepare(fpath, augment=True)
            if image_patch is None or len(labels) < 2:
                skipped += 1
                continue

            step_loss = self._train_step(image_patch, gt_patch, labels)
            if step_loss is not None:
                losses.append(step_loss)

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                mean_l = np.mean(losses[-50:]) if losses else 0
                print(f"  [{fi+1}/{len(files)}] loss={mean_l:.4f} "
                      f"skip={skipped} t={elapsed:.0f}s")

        elapsed = time.time() - t0
        mean_loss = np.mean(losses) if losses else 0
        print(f"Epoch {epoch}: loss={mean_loss:.4f} "
              f"n={len(losses)} skip={skipped} time={elapsed:.0f}s")
        return mean_loss

    def _train_step(self, image_crop, gt_crop, labels):
        """Per-object forward with 10-channel input (8 original + 2 competition).

        For each object k at each round:
          ch0: image
          ch1-7: interaction channels (prev_pred, bbox, fg_click, bg_click, etc.)
          ch8: other_mask — binary mask of all OTHER objects' assembled predictions
          ch9: competition_pressure — sum of other objects' sigmoid probs (maxprob)
                                      or same as ch8 (lastwins)
        """
        device = self.device
        K = len(labels)
        full_shape = image_crop.shape
        patch_shape = (PATCH_SIZE,) * 3

        labels_sorted = sorted(labels)
        click_hist_image = defaultdict(list)
        assembled_per_obj = {k: None for k in labels}
        gt_binaries_full = {k: (gt_crop == k).astype(np.float32) for k in labels}

        # For maxprob: store per-object probability maps
        prob_maps = {k: None for k in labels}  # full-image float32

        self.optimizer.zero_grad()

        total_loss_val = 0.0
        n_fwd = 0

        for round_idx in range(self.num_rounds):
            round_preds = {}
            round_probs = {}  # for maxprob assembly

            # ── Generate clicks (based on assembled view, matching eval) ──
            for k in labels:
                gt_k_full = gt_binaries_full[k]
                if round_idx == 0:
                    edt = _compute_edt_safe_eval_style(gt_k_full > 0)
                    if edt.max() > 0:
                        center = _sample_coord_eval_style(edt * (gt_k_full > 0))
                    else:
                        coords = np.argwhere(gt_k_full > 0)
                        if len(coords) == 0:
                            continue
                        center = tuple(coords[len(coords) // 2])
                    click_hist_image[k].append({
                        'pos_image': center, 'is_fg': True, 'round': round_idx,
                    })
                else:
                    if assembled_per_obj[k] is None:
                        continue
                    per_class_seg = assembled_per_obj[k]
                    per_class_gt = (gt_k_full > 0).astype(np.uint8)
                    center, is_fg = generate_click_eval_style(per_class_seg, per_class_gt)
                    if center is not None:
                        click_hist_image[k].append({
                            'pos_image': center, 'is_fg': bool(is_fg),
                            'round': round_idx,
                        })

            # ── Forward + loss per object ──
            for k in labels:
                if not click_hist_image[k]:
                    continue

                patch_center = click_hist_image[k][-1]['pos_image']
                patch_start = [patch_center[d] - PATCH_SIZE // 2 for d in range(3)]

                image_patch_np = _extract_patch_single(image_crop, patch_center)
                gt_k_patch_np = _extract_patch_single(
                    gt_binaries_full[k], patch_center)

                # ── Build standard 7 interaction channels ──
                interactions = np.zeros((7, *patch_shape), dtype=np.float32)

                # prev_pred from assembled view
                if assembled_per_obj[k] is not None:
                    interactions[0] = _extract_patch_single(
                        assembled_per_obj[k].astype(np.float32), patch_center)

                n_clicks_self = len(click_hist_image[k])
                for ci, c in enumerate(click_hist_image[k]):
                    cp = [c['pos_image'][d] - patch_start[d] for d in range(3)]
                    if not all(0 <= x < PATCH_SIZE for x in cp):
                        continue
                    blob = generate_point_blob(patch_shape, tuple(cp), POINT_RADIUS)
                    decay = INTERACTION_DECAY_TRAIN ** (n_clicks_self - 1 - ci)
                    blob = blob * decay
                    ch = 3 if c['is_fg'] else 4
                    interactions[ch] = np.maximum(interactions[ch], blob)

                # ── Build 2 extra competition channels ──
                # ch8: other_mask — binary union of all other objects' predictions
                # ch9: competition_pressure — sum of other objects' sigmoid probs
                other_mask_full = np.zeros(full_shape, dtype=np.float32)
                competition_pressure_full = np.zeros(full_shape, dtype=np.float32)

                if round_idx > 0:
                    for j in labels:
                        if j == k:
                            continue
                        if assembled_per_obj[j] is not None:
                            other_mask_full = np.maximum(
                                other_mask_full,
                                assembled_per_obj[j].astype(np.float32))
                        if self.assembly == 'maxprob' and prob_maps[j] is not None:
                            competition_pressure_full += prob_maps[j]
                        elif assembled_per_obj[j] is not None:
                            # lastwins: pressure = binary mask (same as ch8)
                            competition_pressure_full += \
                                assembled_per_obj[j].astype(np.float32)

                    # Clip pressure to [0, 1] for numerical stability
                    competition_pressure_full = np.clip(
                        competition_pressure_full, 0.0, 1.0)

                ch8_patch = _extract_patch_single(other_mask_full, patch_center)
                ch9_patch = _extract_patch_single(competition_pressure_full, patch_center)

                # ── Assemble 10-channel input ──
                input_10ch_np = np.concatenate([
                    image_patch_np[None],       # ch0: image
                    interactions,                # ch1-7: interaction
                    ch8_patch[None],             # ch8: other_mask
                    ch9_patch[None],             # ch9: competition_pressure
                ], axis=0)[None]  # (1, 10, D, H, W)

                input_10ch = torch.from_numpy(input_10ch_np).to(device)

                # ── Forward pass ──
                # Encoder stage 0 unfrozen, stages 1-2 LoRA → need grad through skip[0-2]
                # Stages 3-5 fully frozen → detach skip[3-5] to save memory
                # Decoder stages 3-4 LoRA → grad flows via skip[0-1] concat
                with autocast_ctx():
                    skips = self.network.encoder(input_10ch)
                    # Detach skips from frozen encoder stages (3-5) to save ~5GB memory
                    for si in range(3, len(skips)):
                        skips[si] = skips[si].detach()
                    outputs = self.network.decoder(skips)

                    gt_t = torch.from_numpy(
                        gt_k_patch_np[None, None]).float().to(device)
                    targets = downsample_target_for_ds(gt_t)
                    loss = self.criterion(outputs, targets)

                n_fwd += 1
                self.scaler.scale(loss / (K * self.num_rounds)).backward()
                total_loss_val += loss.item()

                # ── Update predictions for assembly ──
                with torch.no_grad():
                    out_full = outputs[0][0]  # (C, D, H, W)
                    pred_patch = (out_full.argmax(0) == 1).cpu().numpy().astype(np.uint8)

                    if assembled_per_obj[k] is not None:
                        pred_obj_buffer = assembled_per_obj[k].copy()
                    else:
                        pred_obj_buffer = np.zeros(full_shape, dtype=np.uint8)
                    _paste_patch_into_buffer(pred_obj_buffer, pred_patch, patch_center)
                    round_preds[k] = pred_obj_buffer

                    if self.assembly == 'maxprob':
                        prob_patch = torch.softmax(
                            out_full.float(), dim=0)[1].cpu().numpy().astype(np.float32)
                        prob_obj_buffer = np.zeros(full_shape, dtype=np.float32)
                        if prob_maps[k] is not None:
                            prob_obj_buffer[:] = prob_maps[k]
                        _paste_patch_into_buffer(
                            prob_obj_buffer, prob_patch, patch_center)
                        round_probs[k] = prob_obj_buffer

            # ── Assembly after all objects this round ──
            if round_preds:
                if self.assembly == 'maxprob':
                    assembled = assemble_max_prob(
                        round_preds, round_probs, full_shape, labels_sorted)
                else:
                    assembled = assemble_last_wins(
                        round_preds, full_shape, labels_sorted)
                for k in labels:
                    assembled_per_obj[k] = (assembled == k).astype(np.uint8)

                # Update prob_maps for next round's competition_pressure
                if self.assembly == 'maxprob':
                    for k in labels:
                        if k in round_probs:
                            prob_maps[k] = round_probs[k]

        # ── Gradient step ──
        if n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            all_trainable = []
            for pg in self.optimizer.param_groups:
                all_trainable.extend(pg['params'])
            nn.utils.clip_grad_norm_(
                [p for p in all_trainable if p.requires_grad], max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Collect stem state dict
        stem_state = {
            'weight': self.stem_conv.weight.data.cpu(),
        }
        if self.stem_conv.bias is not None:
            stem_state['bias'] = self.stem_conv.bias.data.cpu()

        # Collect encoder stage 0 state dict
        stage0_state = {}
        for name, param in self.network.encoder.stages[0].named_parameters():
            stage0_state[name] = param.data.cpu()

        # Collect all LoRA state dict
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()

        torch.save({
            'stem_state_dict': stem_state,
            'stage0_state_dict': stage0_state,
            'lora_state_dict': lora_state,
            'lora_rank': self.lora_rank,
            'assembly': self.assembly,
            'epoch': epoch,
            'loss': loss,
        }, path)
        print(f"Saved: {path} (stem + stage0 + {len(lora_state)} lora tensors)")


def main():
    parser = argparse.ArgumentParser(
        description='Plan A: Input Concat training — 10ch input with competition channels')
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='LoRA rank for encoder/decoder stages')
    parser.add_argument('--enc_lora_stages', default='1,2',
                        help='Encoder stages for LoRA (after unfrozen stage 0)')
    parser.add_argument('--dec_lora_stages', default='3,4',
                        help='Decoder stages for LoRA (shallow end)')
    parser.add_argument('--lora_lr_scale', type=float, default=0.3,
                        help='LoRA lr = main lr * scale')
    parser.add_argument('--stem_lr_scale', type=float, default=1.0,
                        help='Stem conv lr = main lr * scale')
    parser.add_argument('--stage0_lr_scale', type=float, default=0.3,
                        help='Encoder stage 0 lr = main lr * scale')
    parser.add_argument('--assembly', choices=['lastwins', 'maxprob'],
                        default='maxprob',
                        help='Between-round assembly mode')
    parser.add_argument('--save_dir', default='experiments/input_concat')
    args = parser.parse_args()

    files = find_brats_files(args.data_root, max_files=args.num_files)
    if not files:
        print("No BraTS files found!")
        return

    trainer = InputConcatTrainer(
        gpu=args.gpu, lr=args.lr,
        num_rounds=args.num_rounds,
        lora_rank=args.lora_rank,
        enc_lora_stages=args.enc_lora_stages,
        dec_lora_stages=args.dec_lora_stages,
        lora_lr_scale=args.lora_lr_scale,
        stem_lr_scale=args.stem_lr_scale,
        stage0_lr_scale=args.stage0_lr_scale,
        assembly=args.assembly,
    )

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch, args.epochs)

        # Save every epoch
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
