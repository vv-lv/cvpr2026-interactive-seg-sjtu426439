"""
Decoder Attention 训练脚本 — 在 24³ 分辨率注入 click 信息。

关键改动 vs bottleneck 版本：
- Attention 插入到 decoder stage 1 输出（24³, 256ch）而非 bottleneck (6³, 320ch)
- 更高的分辨率让不同 object 的 click 可被空间区分

Usage:
    python -m training.run_decoder_attn --num_files 300 --epochs 15 --gpu 0
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
    DecoderAttentionModule, wrap_decoder_stage, count_parameters,
    compute_token_pos,
)
from training.bottleneck_attention import (
    normalize_pos, ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG,
)
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.lora import LoRAConv3d
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


def build_token_info(self_clicks, other_clicks):
    """Sparse-only token_info，用于 decoder attention。"""
    return {'clicks': self_clicks + other_clicks}


class DecoderAttentionTrainer:

    def __init__(self, gpu: int = 0, lr: float = 3e-4,
                 num_rounds: int = 4, frozen: bool = False,
                 stage_idx: int = 1, internal_dim: int = 128,
                 num_layers: int = 1, num_heads: int = 4,
                 lora_rank: int = 0, lora_stages: str = '2,3',
                 lora_lr_scale: float = 0.1, assembly: str = 'lastwins',
                 use_tanh_gate: bool = False,
                 use_softmax_competition: bool = False,
                 use_relative_pos: bool = False,
                 use_token_gate: bool = False,
                 use_voxel_gate: bool = False,
                 use_learnable_scale: bool = False,
                 use_lora_scale: bool = False,
                 overlap_lambda: float = 0.0,
                 max_objects: int = 8,
                 lora_wd: float = 1e-4,
                 lora_stage_lrs: dict = None,
                 lora_norm_cap: float = 0.0,
                 lora_dropout: float = 0.0,
                 use_rslora: bool = False,
                 skel_recall_weight: float = 0.0):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.frozen = frozen
        self.stage_idx = stage_idx
        self.use_relative_pos = use_relative_pos
        self.use_token_gate = use_token_gate
        self.overlap_lambda = overlap_lambda
        self.max_objects = max_objects
        self.lora_rank = lora_rank
        self.lora_lr_scale = lora_lr_scale
        self.lora_norm_cap = lora_norm_cap
        self.lora_stage_lrs = lora_stage_lrs or {}
        self.lora_wd = lora_wd
        self.skel_recall_weight = skel_recall_weight
        self.lora_dropout = lora_dropout
        self.use_rslora = use_rslora
        assert assembly in ('lastwins', 'maxprob'), f"unknown assembly: {assembly}"
        self.assembly = assembly
        print(f"Between-round assembly: {assembly}")

        if skel_recall_weight > 0:
            from training.skeleton_recall_loss import SoftSkeletonRecallLoss
            self.skel_loss = SoftSkeletonRecallLoss()
            print(f"Skeleton Recall Loss: weight={skel_recall_weight}")

        # 1. 网络（encoder + decoder 全冻结）
        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        # 2. 确定 stage_idx 对应的 input_dim 和 spatial_size
        # Decoder stages: 0=12³×320, 1=24³×256, 2=48³×128, 3=96³×64, 4=192³×32
        stage_configs = {
            0: (320, 12),
            1: (256, 24),
            2: (128, 48),
            3: (64, 96),
            4: (32, 192),
        }
        input_dim, spatial_size = stage_configs[stage_idx]
        print(f"Insert at decoder stage {stage_idx}: ({input_dim}ch, {spatial_size}³)")

        # 3. Attention module
        self.attention = DecoderAttentionModule(
            input_dim=input_dim,
            spatial_size=spatial_size,
            internal_dim=internal_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bg_tokens=1,
            use_tanh_gate=use_tanh_gate,
            use_softmax_competition=use_softmax_competition,
            use_relative_pos=use_relative_pos,
            use_token_gate=use_token_gate,
            use_voxel_gate=use_voxel_gate,
            use_learnable_scale=use_learnable_scale,
            use_lora_scale=use_lora_scale,
        ).to(self.device)

        n_params = count_parameters(self.attention)
        print(f"Attention module: {n_params:,} trainable params")
        print(f"Training rounds: {num_rounds}")

        # 4. Wrap decoder stage
        self.stage_wrapper = wrap_decoder_stage(
            self.network.decoder, stage_idx=stage_idx, attention=self.attention)

        # 5. LoRA on decoder stages AFTER attention injection (Bug D 同期修复 + 联合训练)
        # 默认 stage 2, 3 — attention 在 stage 1，stage 2 是第一个收到 modified feature 的层
        self.lora_params = []
        self._set_lora_bypass = lambda b: None
        if lora_rank > 0:
            from training.lora import apply_lora_to_decoder, get_lora_params, set_lora_bypass
            self._set_lora_bypass = lambda b: set_lora_bypass(self.network, b)
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
                  f"rank={lora_rank}, lr={lr * lora_lr_scale:.1e}")

        if frozen:
            self.attention.eval()
            for p in self.attention.parameters():
                p.requires_grad_(False)
            for p in self.lora_params:
                p.requires_grad_(False)
            print("FROZEN mode")

        # 6. Loss
        self.criterion = build_loss(deep_supervision=True).to(self.device)

        # 7. Optimizer — attention + LoRA 分组 lr (支持 per-stage LoRA LR)
        self._lora_modules = {}  # stage_idx → LoRAConv3d (for norm clamp)
        if not frozen:
            param_groups = [
                {'params': list(self.attention.parameters()),
                 'lr': lr, 'weight_decay': 1e-4},
            ]
            if self.lora_params and lora_rank > 0:
                from training.lora import LoRAConv3d
                # Collect per-stage LoRA params
                stage_params = {}
                for name, module in self.network.named_modules():
                    if isinstance(module, LoRAConv3d):
                        stage_num = int(name.split('.stages.')[1].split('.')[0])
                        stage_params.setdefault(stage_num, [])
                        stage_params[stage_num].extend(
                            [module.lora_A.weight, module.lora_B.weight])
                        self._lora_modules[stage_num] = module
                default_lora_lr = lr * lora_lr_scale
                for s, params in sorted(stage_params.items()):
                    s_lr = self.lora_stage_lrs.get(s, default_lora_lr)
                    param_groups.append({
                        'params': params,
                        'lr': s_lr, 'weight_decay': lora_wd,
                        'stage': s,
                    })
                    print(f"  LoRA stage {s}: lr={s_lr:.1e}, wd={lora_wd}")
            self.optimizer = torch.optim.AdamW(param_groups)
            self.scaler = GradScaler()

            # Cache frozen conv weight norms for norm clamp
            self._frozen_conv_norms = {}
            if lora_norm_cap > 0 and self._lora_modules:
                for s, lora_mod in self._lora_modules.items():
                    w_norm = lora_mod.original_conv.weight.float().norm().item()
                    self._frozen_conv_norms[s] = w_norm
                    print(f"  Norm clamp stage {s}: ||W_frozen||={w_norm:.4f}, "
                          f"cap={lora_norm_cap*100:.1f}%")

    def train_epoch(self, files: list, epoch: int, total_epochs: int):
        if not self.frozen:
            self.attention.train()
        else:
            self.attention.eval()

        self._clamp_counts = {}

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

            spacing_dhw = None
            if self.use_relative_pos:
                try:
                    _npz = np.load(fpath, allow_pickle=True)
                    sp = _npz['spacing'].tolist() if 'spacing' in _npz else [1., 1., 1.]
                    spacing_dhw = [sp[2], sp[1], sp[0]]
                except Exception:
                    spacing_dhw = [1., 1., 1.]

            step_loss = self._train_step(image_patch, gt_patch, labels,
                                         spacing_dhw=spacing_dhw)
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

        # Log LoRA norm ratios
        if self._lora_modules and self._frozen_conv_norms:
            for s, lora_mod in sorted(self._lora_modules.items()):
                A = lora_mod.lora_A.weight.detach()
                B = lora_mod.lora_B.weight.detach()
                A_2d = A.reshape(A.shape[0], -1)
                B_2d = B.reshape(B.shape[0], B.shape[1])
                ba_norm = (B_2d @ A_2d).norm().item() * (lora_mod.alpha / lora_mod.rank)
                w_norm = self._frozen_conv_norms[s]
                ratio = ba_norm / w_norm * 100
                clamps = self._clamp_counts.get(s, 0)
                print(f"  LoRA stage {s}: ||ΔW||/||W||={ratio:.2f}%, "
                      f"clamp_count={clamps}")

        return mean_loss

    def _train_step(self, image_crop, gt_crop, labels, spacing_dhw=None):
        """Per-object patches + assembly between rounds (Bug B/C/D/E/F 修复).

        Assembly mode:
          - lastwins: 默认，简单按 sorted label 顺序覆盖。
          - maxprob: 捕获每个 object 的 fg 概率，overlap 取 argmax — 与 eval 一致。
        """
        device = self.device
        full_shape = image_crop.shape
        patch_shape = (PATCH_SIZE,) * 3

        if len(labels) > self.max_objects:
            labels = sorted(random.sample(labels, self.max_objects))
        K = len(labels)

        labels_sorted = sorted(labels)
        click_hist_image = defaultdict(list)
        assembled_per_obj = {k: None for k in labels}
        gt_binaries_full = {k: (gt_crop == k).astype(np.float32) for k in labels}

        skel_binaries_full = {}
        if self.skel_recall_weight > 0:
            from training.skeleton_recall_loss import precompute_multiclass_skeleton
            skel_full = precompute_multiclass_skeleton(gt_crop, labels)
            for k in labels:
                sk = (skel_full == k).astype(np.float32)
                skel_binaries_full[k] = sk if sk.sum() > 0 else None

        # Step 1.1: per-object full-resolution prob buffer + per-round snapshots
        full_masks = {k: np.zeros(full_shape, dtype=np.float16) for k in labels}
        mask_snapshots = {k: [] for k in labels}

        if not self.frozen:
            self.optimizer.zero_grad()

        total_loss_val = 0.0
        n_fwd = 0
        n_trained = 0
        n_bypassed = 0

        for round_idx in range(self.num_rounds):
            round_preds = {}
            round_probs = {}  # 仅 maxprob 模式使用
            skel_patches = {}  # per-round skeleton patches

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

                if self.skel_recall_weight > 0:
                    sk = skel_binaries_full.get(k)
                    if sk is not None:
                        skel_patches[k] = _extract_patch_single(sk, patch_center)
                    else:
                        skel_patches[k] = None

                interactions = np.zeros((7, *patch_shape), dtype=np.float32)

                # prev_pred from assembled view (Bug D 修复)
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

                # Token info — self clicks 保留全部（patch 以最新 self click 为中心）
                # other clicks 只保留落在 patch 内的（过滤 patch 外噪声）
                # 使用 compute_token_pos 保证训练-eval 一致
                use_rel = self.use_relative_pos
                self_tokens = []
                for c in click_hist_image[k]:
                    if use_rel:
                        pos = compute_token_pos(
                            c['pos_image'], patch_center, PATCH_SIZE,
                            use_relative=True, spacing_dhw=spacing_dhw)
                    else:
                        pos = compute_token_pos(
                            c['pos_image'], patch_center, PATCH_SIZE,
                            use_relative=False).clamp(0, 1)
                    self_tokens.append({
                        'pos': pos,
                        'role': ROLE_SELF_FG if c['is_fg'] else ROLE_SELF_BG,
                        'round': c['round'],
                    })
                other_tokens = []
                for j in labels:
                    if j == k:
                        continue
                    for c in click_hist_image[j]:
                        pos_abs = compute_token_pos(
                            c['pos_image'], patch_center, PATCH_SIZE,
                            use_relative=False)
                        if not all(0.0 <= x <= 1.0 for x in pos_abs.tolist()):
                            continue
                        if use_rel:
                            pos = compute_token_pos(
                                c['pos_image'], patch_center, PATCH_SIZE,
                                use_relative=True, spacing_dhw=spacing_dhw)
                        else:
                            pos = pos_abs
                        other_tokens.append({
                            'pos': pos,
                            'role': ROLE_OTHER_FG if c['is_fg'] else ROLE_OTHER_BG,
                            'round': c['round'],
                        })
                token_info = build_token_info(self_tokens, other_tokens)

                # 检查是否有 other click 在 patch 内
                has_other_in_patch = len(other_tokens) > 0

                if has_other_in_patch:
                    # 有 other click → attention + LoRA 激活，计算 loss + backward
                    self.stage_wrapper._bypass = False
                    self._set_lora_bypass(False)
                    if self.attention.use_lora_scale:
                        lora_s = torch.sigmoid(self.attention.lora_scale_param) * 2
                        for m in self.network.modules():
                            if isinstance(m, LoRAConv3d):
                                m._external_scale = lora_s
                    self.stage_wrapper.set_token_info(token_info)

                    with autocast_ctx():
                        with torch.no_grad():
                            skips = self.network.encoder(input_8ch)
                        skips = [s.detach() for s in skips]
                        outputs = self.network.decoder(skips)

                        gt_t = torch.from_numpy(gt_k_patch_np[None, None]).float().to(device)
                        targets = downsample_target_for_ds(gt_t)
                        loss = self.criterion(outputs, targets)

                        if self.skel_recall_weight > 0:
                            skel_k = skel_patches.get(k)
                            if skel_k is not None and skel_k.sum() > 0:
                                skel_t = torch.from_numpy(
                                    skel_k[None, None]).float().to(device)
                                pred_sm = torch.softmax(outputs[0].float(), dim=1)
                                skel_l = self.skel_loss(pred_sm, skel_t)
                                loss = loss + self.skel_recall_weight * skel_l

                        if self.overlap_lambda > 0:
                            other_gt_patch = np.zeros(patch_shape, dtype=np.float32)
                            for j in labels:
                                if j == k:
                                    continue
                                other_gt_patch = np.maximum(
                                    other_gt_patch,
                                    _extract_patch_single(gt_binaries_full[j], patch_center))
                            if other_gt_patch.sum() > 0:
                                omask = torch.from_numpy(other_gt_patch[None, None]).to(device)
                                ce_high = nn.functional.cross_entropy(
                                    outputs[0].float(),
                                    gt_t[:, 0].long(),
                                    reduction='none')
                                n_total = ce_high.numel()
                                ds_w0 = 16.0 / 31.0
                                overlap_extra = ds_w0 * (ce_high * omask[:, 0]).sum() / n_total
                                loss = loss + self.overlap_lambda * overlap_extra

                    n_fwd += 1
                    n_trained += 1

                    if not self.frozen:
                        self.scaler.scale(loss / (K * self.num_rounds)).backward()

                    total_loss_val += loss.item()
                else:
                    # 无 other click → bypass attention + LoRA，纯 baseline 推理
                    self.stage_wrapper._bypass = True
                    self._set_lora_bypass(True)
                    n_bypassed += 1
                    with torch.no_grad():
                        with autocast_ctx():
                            skips = self.network.encoder(input_8ch)
                            outputs = self.network.decoder(skips)

                with torch.no_grad():
                    # outputs[0] has shape (B=1, C=2, D, H, W) at full patch res.
                    # Index by [0] to drop batch dim explicitly (vs squeeze, which
                    # would also strip a coincident singleton C dim if C==1).
                    out_full = outputs[0][0]  # (C, D, H, W)
                    pred_patch = (out_full.argmax(0) == 1).cpu().numpy().astype(np.uint8)
                    # Bug E 修复：保留 prev_pred 在 patch 外的值（匹配 eval）
                    if assembled_per_obj[k] is not None:
                        pred_obj_buffer = assembled_per_obj[k].copy()
                    else:
                        pred_obj_buffer = np.zeros(full_shape, dtype=np.uint8)
                    _paste_patch_into_buffer(pred_obj_buffer, pred_patch, patch_center)
                    round_preds[k] = pred_obj_buffer

                    if self.assembly == 'maxprob':
                        # 捕获 FG 概率（softmax over channel dim, take class 1）
                        prob_patch = torch.softmax(
                            out_full.float(), dim=0)[1].cpu().numpy().astype(np.float32)
                        prob_obj_buffer = np.zeros(full_shape, dtype=np.float32)
                        _paste_patch_into_buffer(prob_obj_buffer, prob_patch, patch_center)
                        round_probs[k] = prob_obj_buffer

                    # Step 1.1: paste prob into full_mask for snapshot
                    snap_prob = torch.softmax(
                        out_full.float(), dim=0)[1].cpu().numpy().astype(np.float16)
                    _paste_patch_into_buffer(full_masks[k], snap_prob, patch_center)

            # Bug D 修复：本轮所有 forward 完成后做 assembly
            if round_preds:
                if self.assembly == 'maxprob':
                    assembled = assemble_max_prob(
                        round_preds, round_probs, full_shape, labels_sorted)
                else:
                    assembled = assemble_last_wins(round_preds, full_shape, labels_sorted)
                for k in labels:
                    assembled_per_obj[k] = (assembled == k).astype(np.uint8)

            # Step 1.1: save mask snapshots for this round
            for k in labels:
                mask_snapshots[k].append(full_masks[k].copy())
            assert all(len(mask_snapshots[k]) == round_idx + 1 for k in labels)

        if not self.frozen and n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            all_trainable = list(self.attention.parameters()) + self.lora_params
            nn.utils.clip_grad_norm_(
                [p for p in all_trainable if p.requires_grad], max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # LoRA norm clamp
            if self.lora_norm_cap > 0 and self._lora_modules:
                for s, lora_mod in self._lora_modules.items():
                    A = lora_mod.lora_A.weight
                    B = lora_mod.lora_B.weight
                    A_2d = A.detach().reshape(A.shape[0], -1)
                    B_2d = B.detach().reshape(B.shape[0], B.shape[1])
                    ba_norm = (B_2d @ A_2d).norm().item() * (lora_mod.alpha / lora_mod.rank)
                    w_norm = self._frozen_conv_norms[s]
                    ratio = ba_norm / w_norm
                    if ratio > self.lora_norm_cap:
                        scale = (self.lora_norm_cap / ratio) ** 0.5
                        A.data.mul_(scale)
                        B.data.mul_(scale)
                        self._clamp_counts[s] = self._clamp_counts.get(s, 0) + 1

        if n_bypassed > 0:
            print(f"    [bypass] trained={n_trained} bypassed={n_bypassed}")
        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from training.lora import LoRAConv3d
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()
        save_dict = {
            'attention_state_dict': self.attention.state_dict(),
            'lora_state_dict': lora_state,
            'stage_idx': self.stage_idx,
            'lora_rank': self.lora_rank,
            'use_tanh_gate': self.attention.use_tanh_gate,
            'use_softmax_competition': self.attention.use_softmax_competition,
            'use_relative_pos': self.attention.use_relative_pos,
            'use_token_gate': self.attention.use_token_gate,
            'use_voxel_gate': self.attention.use_voxel_gate,
            'use_learnable_scale': self.attention.use_learnable_scale,
            'use_lora_scale': self.attention.use_lora_scale,
            'internal_dim': self.attention.internal_dim,
            'num_layers': self.attention.num_layers,
            'num_heads': self.attention.num_heads,
            'num_bg_tokens': self.attention.bg_tokens.shape[0],
            'epoch': epoch,
            'loss': loss,
        }
        if not self.frozen:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            save_dict['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(save_dict, path)
        print(f"Saved: {path} (attn + {len(lora_state)} lora tensors)")

    def resume_from(self, path: str):
        """从 checkpoint 恢复 attention + LoRA + optimizer + scaler 状态。"""
        from training.lora import LoRAConv3d
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.attention.load_state_dict(ckpt['attention_state_dict'])
        lora_state = ckpt.get('lora_state_dict', {})
        if lora_state:
            for name, module in self.network.named_modules():
                if isinstance(module, LoRAConv3d):
                    for key, val in lora_state.items():
                        if name in key:
                            if 'lora_A' in key:
                                module.lora_A.weight.data.copy_(val.to(self.device))
                            elif 'lora_B' in key:
                                module.lora_B.weight.data.copy_(val.to(self.device))
        if not self.frozen and 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scaler_state_dict' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            except ValueError:
                print("  WARNING: optimizer state mismatch (param groups changed), "
                      "starting fresh optimizer")
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {path} (epoch {ckpt['epoch']}, "
              f"loss={ckpt['loss']:.4f}), continuing from epoch {start_epoch}")
        return start_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--stage_idx', type=int, default=1,
                        help='Decoder stage to insert attention (1=24³)')
    parser.add_argument('--internal_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=0,
                        help='LoRA rank (0 = disabled)')
    parser.add_argument('--lora_stages', default='2,3',
                        help='Decoder stages for LoRA (after attention injection)')
    parser.add_argument('--lora_lr_scale', type=float, default=0.1,
                        help='LoRA lr = main lr * scale')
    parser.add_argument('--assembly', choices=['lastwins', 'maxprob'],
                        default='lastwins',
                        help='Between-round assembly: lastwins (legacy) or '
                             'maxprob (matches max_prob eval)')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--include_acdc', action='store_true',
                        help='Include MR_Heart_ACDC in training data')
    parser.add_argument('--include_extended', action='store_true',
                        help='Include all dist<96 multi-object datasets (CT/MRI/Microscopy)')
    parser.add_argument('--use_tanh_gate', action='store_true',
                        help='Use Flamingo-style tanh gating instead of zero-init proj_out')
    parser.add_argument('--use_softmax_competition', action='store_true',
                        help='v5: BG+click tokens with softmax competition per spatial position')
    parser.add_argument('--candidates', default=None,
                        help='Pre-filtered candidates.json (from prefilter_candidates.py). '
                             'If set, --data_root is ignored.')
    parser.add_argument('--train_json', default=None,
                        help='Balanced split JSON (from build_balanced_splits.py). '
                             'Overrides --data_root and --candidates.')
    parser.add_argument('--use_relative_pos', action='store_true',
                        help='Use relative displacement + physical distance as token PE')
    parser.add_argument('--use_token_gate', action='store_true',
                        help='Per-token relation gate on K/V (sigmoid MLP, bias=3.0)')
    parser.add_argument('--use_voxel_gate', action='store_true',
                        help='Per-voxel conflict gate on attention delta (conv, bias=3.0)')
    parser.add_argument('--use_learnable_scale', action='store_true',
                        help='Learnable attention scale parameter (attn_scale only)')
    parser.add_argument('--use_lora_scale', action='store_true',
                        help='Learnable LoRA scale parameter')
    parser.add_argument('--overlap_lambda', type=float, default=0.0,
                        help='Overlap-aware loss weight (0=disabled, 1.0=2x penalty)')
    parser.add_argument('--max_objects', type=int, default=8,
                        help='Max objects per training step (random subset if exceeded)')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--lora_wd', type=float, default=1e-4,
                        help='LoRA weight decay (default 1e-4)')
    parser.add_argument('--lora_norm_cap', type=float, default=0.0,
                        help='LoRA Frobenius norm cap as fraction of ||W_frozen|| (0=disabled, 0.05=5%%)')
    parser.add_argument('--lora_stage3_lr', type=float, default=0.0,
                        help='Override LR for LoRA stage 3 (0=use default lora lr)')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='Dropout rate for LoRA layers (0=disabled)')
    parser.add_argument('--use_rslora', action='store_true',
                        help='Use rsLoRA scaling (alpha/sqrt(r) instead of alpha/r)')
    parser.add_argument('--skel_recall_weight', type=float, default=0.0,
                        help='Weight for skeleton recall loss (0=disabled)')
    parser.add_argument('--max_per_dataset', type=int, default=100)
    parser.add_argument('--max_brats', type=int, default=400)
    parser.add_argument('--save_dir', default='experiments/decoder_attn')
    args = parser.parse_args()

    if args.train_json and os.path.exists(args.train_json):
        import json
        with open(args.train_json) as f:
            train_data = json.load(f)
        files = [f['path'] for f in train_data['files']]
        random.shuffle(files)
        sampled_ds = {}
        for f_entry in train_data['files']:
            sampled_ds[f_entry['dataset']] = sampled_ds.get(f_entry['dataset'], 0) + 1
        print(f"Loaded {len(files)} files from {args.train_json} "
              f"({len(sampled_ds)} datasets)")
        for ds, n in sorted(sampled_ds.items(), key=lambda x: -x[1])[:10]:
            print(f"  {ds}: {n}")
    elif args.candidates and os.path.exists(args.candidates):
        import json
        with open(args.candidates) as f:
            cand_data = json.load(f)
        candidates = cand_data['candidates']
        print(f"Loaded {len(candidates)} candidates (threshold={cand_data['threshold']})")
        # Balanced sampling
        by_dataset = {}
        for c in candidates:
            by_dataset.setdefault(c['dataset'], []).append(c['path'])
        files = []
        brats_total = 0
        for ds, paths in sorted(by_dataset.items()):
            random.shuffle(paths)
            if 'BraTS' in ds:
                remaining = args.max_brats - brats_total
                if remaining <= 0:
                    continue
                n = min(len(paths), remaining)
                files.extend(paths[:n])
                brats_total += n
            else:
                n = min(len(paths), args.max_per_dataset)
                files.extend(paths[:n])
        random.shuffle(files)
        if args.num_files and len(files) > args.num_files:
            files = files[:args.num_files]
        # Print summary
        sampled_ds = {}
        for p in files:
            ds = p.split('/')[-2]
            sampled_ds[ds] = sampled_ds.get(ds, 0) + 1
        print(f"Sampled {len(files)} files from {len(sampled_ds)} datasets")
        for ds, n in sorted(sampled_ds.items(), key=lambda x: -x[1])[:10]:
            print(f"  {ds}: {n}")
    else:
        files = find_brats_files(args.data_root, max_files=args.num_files,
                                include_acdc=args.include_acdc,
                                include_extended=args.include_extended)
    if not files:
        print("No files found!")
        return

    lora_stage_lrs = {}
    if args.lora_stage3_lr > 0:
        lora_stage_lrs[3] = args.lora_stage3_lr

    trainer = DecoderAttentionTrainer(
        gpu=args.gpu, lr=args.lr,
        num_rounds=args.num_rounds, frozen=args.frozen,
        stage_idx=args.stage_idx,
        internal_dim=args.internal_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lora_rank=args.lora_rank,
        lora_stages=args.lora_stages,
        lora_lr_scale=args.lora_lr_scale,
        assembly=args.assembly,
        use_tanh_gate=args.use_tanh_gate,
        use_softmax_competition=args.use_softmax_competition,
        use_relative_pos=args.use_relative_pos,
        use_token_gate=args.use_token_gate,
        use_voxel_gate=args.use_voxel_gate,
        use_learnable_scale=args.use_learnable_scale,
        use_lora_scale=args.use_lora_scale,
        overlap_lambda=args.overlap_lambda,
        max_objects=args.max_objects,
        lora_wd=args.lora_wd,
        lora_stage_lrs=lora_stage_lrs,
        lora_norm_cap=args.lora_norm_cap,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        skel_recall_weight=args.skel_recall_weight,
    )

    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.resume_from(args.resume)

    for epoch in range(start_epoch, args.epochs):
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
