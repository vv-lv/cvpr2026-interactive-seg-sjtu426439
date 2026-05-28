"""
Encoder Attention 训练脚本 (Plan B) — 在 encoder stage 3 输出 (24³, 256ch) 注入 click 信息。

关键区别 vs Plan C (decoder attention)：
- Attention 插入到 encoder stage 3 输出，修改 skip[3]
- LoRA 在 encoder stages 4,5 和 decoder stages 0,1（attention 下游）
- 梯度流：encoder 0-2 frozen no_grad → stage 3 frozen conv + attention WITH grad
  → enc 4-5 LoRA grad → decoder 0-1 LoRA grad → decoder 2-4 frozen

Usage:
    python -m training.run_encoder_attn --num_files 300 --epochs 15 --gpu 0
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

from training.decoder_attention import (
    DecoderAttentionModule, count_parameters,
)
from training.bottleneck_attention import (
    normalize_pos, ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG,
)
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
from training.lora import LoRAConv3d, _find_parent, get_lora_params


# ─── Encoder stage wrapper ───────────────────────────────────────────────

class EncoderStageWithAttentionWrapper(nn.Module):
    """包装 encoder stage，在其 forward 输出后调用 attention module。

    encoder.stages[i].forward(x) → feat
    本 wrapper: feat = original_stage(x) → attention(feat, token_info) → return
    """

    def __init__(self, original_stage: nn.Module, attention: DecoderAttentionModule):
        super().__init__()
        self.original_stage = original_stage
        self.attention = attention
        self._token_info = {'clicks': []}

    def set_token_info(self, token_info: dict):
        self._token_info = token_info

    def forward(self, x):
        out = self.original_stage(x)
        out = self.attention(out, self._token_info)
        return out


def wrap_encoder_stage(encoder: nn.Module, stage_idx: int,
                       attention: DecoderAttentionModule
                       ) -> EncoderStageWithAttentionWrapper:
    """替换 encoder.stages[stage_idx] 为带 attention 的 wrapper。

    Returns the wrapper so external code can call set_token_info.
    """
    original = encoder.stages[stage_idx]
    wrapper = EncoderStageWithAttentionWrapper(original, attention)
    encoder.stages[stage_idx] = wrapper
    return wrapper


# ─── LoRA for encoder stages ─────────────────────────────────────────────

def apply_lora_to_encoder(encoder: nn.Module, target_stages: list = None,
                          rank: int = 4, alpha: float = 1.0) -> int:
    """对 encoder 指定 stages 的 conv 层添加 LoRA。

    与 apply_lora_to_decoder 逻辑相同，只是作用于 encoder.stages。

    Args:
        encoder: PlainConvEncoder module
        target_stages: 要添加 LoRA 的 stage 索引列表，如 [4, 5]
        rank: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        新增的 LoRA 参数量
    """
    if target_stages is None:
        target_stages = [4, 5]

    total_params = 0

    for stage_idx in target_stages:
        if stage_idx >= len(encoder.stages):
            continue

        stage = encoder.stages[stage_idx]

        # 如果 stage 是我们的 wrapper，LoRA 应作用于 original_stage
        if isinstance(stage, EncoderStageWithAttentionWrapper):
            actual_stage = stage.original_stage
            stage_prefix = f"encoder.stages.{stage_idx}.original_stage"
        else:
            actual_stage = stage
            stage_prefix = f"encoder.stages.{stage_idx}"

        # 收集所有目标 conv（避免迭代中修改导致重复匹配）
        targets = []
        for block_name, block in actual_stage.named_modules():
            if isinstance(block, nn.Conv3d) and block.kernel_size[0] > 1:
                parent, attr = _find_parent(actual_stage, block_name, block)
                if parent is not None:
                    targets.append((block_name, block, parent, attr))

        for block_name, block, parent, attr in targets:
            lora_conv = LoRAConv3d(block, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_conv)

            # 修复 all_modules Sequential 引用
            if hasattr(parent, 'all_modules'):
                for idx, mod in enumerate(parent.all_modules):
                    if mod is block:
                        parent.all_modules[idx] = lora_conv
                        break

            n = sum(p.numel() for p in [lora_conv.lora_A.weight,
                                         lora_conv.lora_B.weight])
            total_params += n
            print(f"  LoRA added: {stage_prefix}.{block_name} "
                  f"({list(block.weight.shape)}) → rank={rank}, +{n} params")

    return total_params


# ─── Token info builder ──────────────────────────────────────────────────

def build_token_info(self_clicks, other_clicks):
    """Sparse-only token_info，用于 encoder attention。"""
    return {'clicks': self_clicks + other_clicks}


# ─── Trainer ─────────────────────────────────────────────────────────────

class EncoderAttentionTrainer:

    def __init__(self, gpu: int = 0, lr: float = 3e-4,
                 num_rounds: int = 4, frozen: bool = False,
                 enc_stage_idx: int = 3, internal_dim: int = 128,
                 num_layers: int = 1, num_heads: int = 4,
                 lora_rank: int = 0,
                 lora_enc_stages: str = '4,5',
                 lora_dec_stages: str = '0,1',
                 lora_lr_scale: float = 0.1, assembly: str = 'lastwins'):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.frozen = frozen
        self.enc_stage_idx = enc_stage_idx
        self.lora_rank = lora_rank
        self.lora_lr_scale = lora_lr_scale
        assert assembly in ('lastwins', 'maxprob'), f"unknown assembly: {assembly}"
        self.assembly = assembly
        print(f"Between-round assembly: {assembly}")

        # 1. 网络（encoder + decoder 全冻结）
        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        # 2. 确定 enc_stage_idx 对应的 input_dim 和 spatial_size
        # Encoder stages: 0=192³×32, 1=96³×64, 2=48³×128, 3=24³×256, 4=12³×320, 5=6³×320
        enc_stage_configs = {
            0: (32, 192),
            1: (64, 96),
            2: (128, 48),
            3: (256, 24),
            4: (320, 12),
            5: (320, 6),
        }
        input_dim, spatial_size = enc_stage_configs[enc_stage_idx]
        print(f"Insert at encoder stage {enc_stage_idx}: ({input_dim}ch, {spatial_size}³)")

        # 3. Attention module (reuse DecoderAttentionModule — works on any 3D tensor)
        self.attention = DecoderAttentionModule(
            input_dim=input_dim,
            spatial_size=spatial_size,
            internal_dim=internal_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bg_tokens=4,
        ).to(self.device)

        n_params = count_parameters(self.attention)
        print(f"Attention module: {n_params:,} trainable params")
        print(f"Training rounds: {num_rounds}")

        # 4. Wrap encoder stage
        self.stage_wrapper = wrap_encoder_stage(
            self.network.encoder, stage_idx=enc_stage_idx, attention=self.attention)

        # 5. LoRA on encoder stages 4,5 AND decoder stages 0,1
        self.lora_params = []
        if lora_rank > 0:
            from training.lora import apply_lora_to_decoder
            # Encoder LoRA
            enc_targets = [int(s) for s in lora_enc_stages.split(',') if s.strip()]
            n_enc_lora = apply_lora_to_encoder(
                self.network.encoder, target_stages=enc_targets,
                rank=lora_rank, alpha=1.0)
            self.network.encoder.to(self.device)

            # Decoder LoRA
            dec_targets = [int(s) for s in lora_dec_stages.split(',') if s.strip()]
            n_dec_lora = apply_lora_to_decoder(
                self.network.decoder, target_stages=dec_targets,
                rank=lora_rank, alpha=1.0)
            self.network.decoder.to(self.device)

            self.lora_params = get_lora_params(self.network)
            for p in self.lora_params:
                p.requires_grad_(True)
            print(f"LoRA encoder: {n_enc_lora:,} params on stages {enc_targets}")
            print(f"LoRA decoder: {n_dec_lora:,} params on stages {dec_targets}")
            print(f"LoRA total: {n_enc_lora + n_dec_lora:,} params, "
                  f"rank={lora_rank}, lr={lr * lora_lr_scale:.1e}")

        if frozen:
            self.attention.eval()
            for p in self.attention.parameters():
                p.requires_grad_(False)
            for p in self.lora_params:
                p.requires_grad_(False)
            print("FROZEN mode")

        # 6. Determine which encoder stages need grad vs no_grad
        # Stages 0 .. (enc_stage_idx - 1): fully frozen, can run in no_grad
        # Stage enc_stage_idx: frozen conv + attention (needs grad)
        # Stages (enc_stage_idx + 1) .. 5: may have LoRA (needs grad)
        # We store these indices for the custom forward
        self.frozen_enc_stages = list(range(enc_stage_idx))
        self.grad_enc_stages = list(range(enc_stage_idx, len(self.network.encoder.stages)))
        print(f"Encoder no_grad stages: {self.frozen_enc_stages}")
        print(f"Encoder grad stages: {self.grad_enc_stages}")

        # 7. Loss
        self.criterion = build_loss(deep_supervision=True).to(self.device)

        # 8. Optimizer — attention + LoRA 分组 lr
        if not frozen:
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

    def _encoder_forward_mixed_grad(self, input_8ch: torch.Tensor):
        """Custom encoder forward with mixed gradient flow.

        Replicates encoder.forward() but splits no_grad / grad regions:
          stem: always frozen → no_grad
          stages 0..(enc_stage_idx-1): frozen → no_grad, detach
          stages enc_stage_idx..5: attention + LoRA → with grad

        Returns list of skips matching encoder output format.
        """
        encoder = self.network.encoder

        # 1. Stem (frozen, always no_grad): 8ch → 32ch
        with torch.no_grad():
            x = encoder.stem(input_8ch)
        x = x.detach()

        # 2. Stages
        skips = []
        for i, stage in enumerate(encoder.stages):
            if i in self.frozen_enc_stages:
                with torch.no_grad():
                    x = stage(x)
                x = x.detach()
            else:
                # Grad flows through attention (in wrapped stage) and LoRA
                x = stage(x)
            skips.append(x)

        return skips

    def train_epoch(self, files: list, epoch: int, total_epochs: int):
        if not self.frozen:
            self.attention.train()
        else:
            self.attention.eval()

        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0

        for fi, fpath in enumerate(files):
            image_patch, gt_patch, labels = load_and_prepare(
                fpath, augment=not self.frozen)
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
        """Per-object patches + assembly between rounds.

        Assembly mode:
          - lastwins: 默认，简单按 sorted label 顺序覆盖。
          - maxprob: 捕获每个 object 的 fg 概率，overlap 取 argmax — 与 eval 一致。
        """
        device = self.device
        K = len(labels)
        full_shape = image_crop.shape
        patch_shape = (PATCH_SIZE,) * 3

        labels_sorted = sorted(labels)
        click_hist_image = defaultdict(list)
        assembled_per_obj = {k: None for k in labels}
        gt_binaries_full = {k: (gt_crop == k).astype(np.float32) for k in labels}

        if not self.frozen:
            self.optimizer.zero_grad()

        total_loss_val = 0.0
        n_fwd = 0

        for round_idx in range(self.num_rounds):
            round_preds = {}
            round_probs = {}  # 仅 maxprob 模式使用

            # ── 生成 click（基于 assembled view，匹配 eval）──
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
                            'pos_image': center, 'is_fg': bool(is_fg), 'round': round_idx,
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

                input_8ch_np = np.concatenate(
                    [image_patch_np[None], interactions], axis=0)[None]
                input_8ch = torch.from_numpy(input_8ch_np).to(device)

                # Token info — other clicks 只保留 patch 内的
                self_tokens = []
                for c in click_hist_image[k]:
                    cp_norm = [(c['pos_image'][d] - patch_start[d]) / PATCH_SIZE
                               for d in range(3)]
                    cp_norm = [max(0.0, min(1.0, x)) for x in cp_norm]
                    self_tokens.append({
                        'pos': torch.tensor(cp_norm, dtype=torch.float32),
                        'role': ROLE_SELF_FG if c['is_fg'] else ROLE_SELF_BG,
                        'round': c['round'],
                    })
                other_tokens = []
                for j in labels:
                    if j == k:
                        continue
                    for c in click_hist_image[j]:
                        cp_raw = [(c['pos_image'][d] - patch_start[d]) / PATCH_SIZE
                                  for d in range(3)]
                        if not all(0.0 <= x <= 1.0 for x in cp_raw):
                            continue
                        other_tokens.append({
                            'pos': torch.tensor(cp_raw, dtype=torch.float32),
                            'role': ROLE_OTHER_FG if c['is_fg'] else ROLE_OTHER_BG,
                            'round': c['round'],
                        })
                token_info = build_token_info(self_tokens, other_tokens)

                # Set token info on the encoder stage wrapper
                self.stage_wrapper.set_token_info(token_info)

                with autocast_ctx():
                    # Mixed-grad encoder forward: stages 0-2 no_grad, 3+ with grad
                    skips = self._encoder_forward_mixed_grad(input_8ch)

                    # Decoder: stages 0,1 have LoRA (grad), stages 2-4 frozen
                    # Since decoder params without LoRA have requires_grad=False,
                    # autograd naturally only flows through LoRA params.
                    outputs = self.network.decoder(skips)

                    gt_t = torch.from_numpy(gt_k_patch_np[None, None]).float().to(device)
                    targets = downsample_target_for_ds(gt_t)
                    loss = self.criterion(outputs, targets)

                n_fwd += 1

                if not self.frozen:
                    self.scaler.scale(loss / (K * self.num_rounds)).backward()

                total_loss_val += loss.item()

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
                        _paste_patch_into_buffer(prob_obj_buffer, prob_patch, patch_center)
                        round_probs[k] = prob_obj_buffer

            # 本轮所有 forward 完成后做 assembly
            if round_preds:
                if self.assembly == 'maxprob':
                    assembled = assemble_max_prob(
                        round_preds, round_probs, full_shape, labels_sorted)
                else:
                    assembled = assemble_last_wins(round_preds, full_shape, labels_sorted)
                for k in labels:
                    assembled_per_obj[k] = (assembled == k).astype(np.uint8)

        if not self.frozen and n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            all_trainable = list(self.attention.parameters()) + self.lora_params
            nn.utils.clip_grad_norm_(
                [p for p in all_trainable if p.requires_grad], max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 收集 LoRA state dict（包含 encoder 和 decoder 的 LoRA）
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()
        torch.save({
            'attention_state_dict': self.attention.state_dict(),
            'lora_state_dict': lora_state,
            'enc_stage_idx': self.enc_stage_idx,
            'lora_rank': self.lora_rank,
            'epoch': epoch,
            'loss': loss,
        }, path)
        # Count encoder vs decoder LoRA tensors
        enc_lora = sum(1 for k in lora_state if 'encoder' in k)
        dec_lora = sum(1 for k in lora_state if 'decoder' in k)
        print(f"Saved: {path} (attn + {enc_lora} enc_lora + {dec_lora} dec_lora tensors)")


def main():
    parser = argparse.ArgumentParser(
        description='Plan B: Encoder Attention at 24³')
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--enc_stage_idx', type=int, default=3,
                        help='Encoder stage to insert attention (3=24³)')
    parser.add_argument('--internal_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=0,
                        help='LoRA rank (0 = disabled)')
    parser.add_argument('--lora_enc_stages', default='4,5',
                        help='Encoder stages for LoRA (downstream of attention)')
    parser.add_argument('--lora_dec_stages', default='0,1',
                        help='Decoder stages for LoRA (receive modified skips)')
    parser.add_argument('--lora_lr_scale', type=float, default=0.1,
                        help='LoRA lr = main lr * scale')
    parser.add_argument('--assembly', choices=['lastwins', 'maxprob'],
                        default='lastwins',
                        help='Between-round assembly: lastwins (legacy) or '
                             'maxprob (matches max_prob eval)')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--save_dir', default='experiments/encoder_attn')
    args = parser.parse_args()

    files = find_brats_files(args.data_root, max_files=args.num_files)
    if not files:
        print("No BraTS files found!")
        return

    trainer = EncoderAttentionTrainer(
        gpu=args.gpu, lr=args.lr,
        num_rounds=args.num_rounds, frozen=args.frozen,
        enc_stage_idx=args.enc_stage_idx,
        internal_dim=args.internal_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lora_rank=args.lora_rank,
        lora_enc_stages=args.lora_enc_stages,
        lora_dec_stages=args.lora_dec_stages,
        lora_lr_scale=args.lora_lr_scale,
        assembly=args.assembly,
    )

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch, args.epochs)

        if not args.frozen:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, f'epoch_{epoch}.pth'), epoch, loss)
            if loss < best_loss:
                best_loss = loss
                trainer.save_checkpoint(
                    os.path.join(args.save_dir, 'best.pth'), epoch, loss)

    if not args.frozen:
        trainer.save_checkpoint(
            os.path.join(args.save_dir, 'final.pth'), args.epochs - 1, loss)
    print("Done!")


if __name__ == '__main__':
    main()
