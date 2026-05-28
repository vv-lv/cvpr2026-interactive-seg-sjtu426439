"""
Sparse-Only Bottleneck Interaction Attention 训练脚本。

Step 1 最小可行验证：
- 只有 click tokens + BG embeddings（无 dense mask tokens）
- 无 gate, 无 LoRA
- 训练 4 轮交互（覆盖 temporal_emb round 0-3，减少 train-eval gap）
- 300 BraTS files, 15-20 epochs

Usage:
    python -m training.run_bottleneck_attn --num_files 300 --epochs 15 --gpu 1
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

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

from training.bottleneck_attention import (
    BottleneckInteractionAttention, build_token_info, normalize_pos,
    count_parameters, ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG,
)
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.interaction_sim import (
    InteractionManager, generate_point_blob, sample_point_from_error_region,
    POINT_RADIUS, INTERACTION_DECAY,
)
INTERACTION_DECAY_TRAIN = INTERACTION_DECAY  # 0.98
from training.dataset import _normalize_like_inference, preprocess_like_inference, augment_patch, augment_full

PATCH_SIZE = 192
CHECKPOINT_PATH = (
    '/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models'
    '/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth'
)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════════════════════════════

EXTENDED_DATASETS = {
    'MRI': [
        'MR_Heart_ACDC', 'MR_EMIDEC', 'MR_ISLES_ADC', 'MR_ISLES_DWI',
        'MR_HNTS-MRG_HeadTumor', 'MR_HVSMR', 'MR_CHAOS-T1', 'MR_CHAOS-T2',
        'MR_T1c_crossMoDA_Tumor_Cochlea', 'MR_WMH_FLAIR', 'MR_WMH_T1',
    ],
    'CT': [
        'CT_Aortic-Dissection', 'CT_TotalSeg_organs',
        'CT_TotalSeg_cardiac', 'CT_LymphNode',
    ],
    'Microscopy': ['Microscopy_nucmm'],
}


def find_brats_files(data_root: str, max_files: int = None,
                     include_acdc: bool = False,
                     include_extended: bool = False) -> list:
    """Find multi-object training files.

    Args:
        include_acdc: add Heart_ACDC only
        include_extended: add all dist<96 multi-object datasets (CT, MRI, Microscopy)
                          includes ACDC automatically
    """
    dirs = [
        os.path.join(data_root, 'MRI', d)
        for d in ['MR_BraTS-T1c', 'MR_BraTS-T1n', 'MR_BraTS-T2f', 'MR_BraTS-T2w']
    ]
    if include_extended:
        for modality, datasets in EXTENDED_DATASETS.items():
            for ds in datasets:
                dirs.append(os.path.join(data_root, modality, ds))
    elif include_acdc:
        dirs.append(os.path.join(data_root, 'MRI', 'MR_Heart_ACDC'))

    files = []
    for d in dirs:
        if os.path.isdir(d):
            files.extend(
                os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.npz'))
    random.shuffle(files)
    if max_files and len(files) > max_files:
        files = files[:max_files]
    # 统计各数据集
    ds_counts = {}
    for f in files:
        ds = f.split('/')[-2]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    desc = ', '.join(f'{d}:{n}' for d, n in sorted(ds_counts.items(), key=lambda x: -x[1])[:10])
    if len(ds_counts) > 10:
        desc += f', ... +{len(ds_counts)-10} more'
    print(f"Found {len(files)} files from {len(ds_counts)} datasets ({desc})")
    return files


def load_and_prepare(npz_path: str, augment: bool = True):
    """加载并预处理 BraTS 文件 — 返回裁切后的全图（不提取 patch）。

    流程：
    1. 裁切到非零 bbox（等价于 session preprocessing）
    2. 在裁切区域上归一化
    3. 可选增强（在裁切的全图上做增强，但增强后的 shape 可能变化）

    返回的 image_crop, gt_crop 已经预处理好，patch 提取在 _train_step 中
    针对每个 object 单独完成（匹配 eval 的 per-object 处理）。
    """
    data = np.load(npz_path, allow_pickle=True)
    image = data['imgs'].astype(np.float32)
    gt = data['gts'].astype(np.uint8)

    labels = [l for l in np.unique(gt) if l > 0]
    if len(labels) < 2:  # 至少 2 个 object 才能跨 object 训练
        return None, None, []

    # 关键修复 1：先裁切到非零 bbox 再归一化（匹配推理 session）
    image_crop, gt_crop, bbox_min = preprocess_like_inference(image, gt)

    fg_coords = np.argwhere(gt_crop > 0)
    if len(fg_coords) == 0:
        return None, None, []

    # Bug B 修复：启用 shape-preserving augmentation
    if augment:
        image_crop, gt_crop = augment_full(image_crop, gt_crop)

    crop_labels = [l for l in np.unique(gt_crop) if l > 0]
    return image_crop, gt_crop, crop_labels


def _extract_patch(image, gt, center):
    """Extract 192³ patch from image and gt, centered on `center`. Pads with 0."""
    P = PATCH_SIZE
    shape = image.shape
    slices_src, slices_dst = [], []
    for d in range(3):
        lo = center[d] - P // 2
        hi = lo + P
        src_lo, src_hi = max(0, lo), min(shape[d], hi)
        dst_lo = src_lo - lo
        slices_src.append(slice(src_lo, src_hi))
        slices_dst.append(slice(dst_lo, dst_lo + src_hi - src_lo))

    img_patch = np.zeros((P, P, P), dtype=image.dtype)
    gt_patch = np.zeros((P, P, P), dtype=gt.dtype)
    img_patch[tuple(slices_dst)] = image[tuple(slices_src)]
    gt_patch[tuple(slices_dst)] = gt[tuple(slices_src)]
    return img_patch, gt_patch


def _extract_patch_single(arr, center):
    """Extract 192³ patch from a single (D,H,W) array."""
    P = PATCH_SIZE
    shape = arr.shape
    slices_src, slices_dst = [], []
    for d in range(3):
        lo = center[d] - P // 2
        hi = lo + P
        src_lo, src_hi = max(0, lo), min(shape[d], hi)
        dst_lo = src_lo - lo
        slices_src.append(slice(src_lo, src_hi))
        slices_dst.append(slice(dst_lo, dst_lo + src_hi - src_lo))

    patch = np.zeros((P, P, P), dtype=arr.dtype)
    patch[tuple(slices_dst)] = arr[tuple(slices_src)]
    return patch


def _paste_patch_to_full(patch, full_shape, center):
    """Paste 192³ patch back into a full-image-sized buffer.
    `center` is in full-image coordinates."""
    P = PATCH_SIZE
    full = np.zeros(full_shape, dtype=patch.dtype)
    slices_src, slices_dst = [], []
    for d in range(3):
        lo = center[d] - P // 2
        hi = lo + P
        # Source range within patch (handling negative bounds)
        src_lo = max(0, -lo)
        src_hi = P - max(0, hi - full_shape[d])
        # Destination range in full image
        dst_lo = max(0, lo)
        dst_hi = min(full_shape[d], hi)
        slices_src.append(slice(src_lo, src_hi))
        slices_dst.append(slice(dst_lo, dst_hi))

    full[tuple(slices_dst)] = patch[tuple(slices_src)]
    return full


def _paste_patch_into_buffer(buffer, patch, center):
    """In-place paste patch into existing buffer at center.

    Used for Bug E fix: matches eval's paste_tensor behavior where
    the patch region is OVERWRITTEN with new pred, but outside the patch
    keeps its existing values (from prev_pred).
    """
    P = PATCH_SIZE
    full_shape = buffer.shape
    slices_src, slices_dst = [], []
    for d in range(3):
        lo = center[d] - P // 2
        hi = lo + P
        src_lo = max(0, -lo)
        src_hi = P - max(0, hi - full_shape[d])
        dst_lo = max(0, lo)
        dst_hi = min(full_shape[d], hi)
        slices_src.append(slice(src_lo, src_hi))
        slices_dst.append(slice(dst_lo, dst_hi))
    buffer[tuple(slices_dst)] = patch[tuple(slices_src)]


# ═══════════════════════════════════════════════════════════════════════════════
# 交互模拟
# ═══════════════════════════════════════════════════════════════════════════════

def generate_initial_click(gt_binary: np.ndarray) -> tuple:
    from scipy.ndimage import distance_transform_edt
    coords = np.argwhere(gt_binary > 0)
    if len(coords) == 0:
        return None
    dt = distance_transform_edt(gt_binary)
    dt_vals = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
    weights = dt_vals ** 8
    total = weights.sum()
    if total > 0:
        weights /= total
        idx = np.random.choice(len(coords), p=weights)
    else:
        idx = random.randint(0, len(coords) - 1)
    return tuple(coords[idx])


def generate_followup_click(pred_binary, gt_binary):
    fn_region = (gt_binary > 0) & (pred_binary == 0)
    fp_region = (gt_binary == 0) & (pred_binary > 0)
    fn_count, fp_count = fn_region.sum(), fp_region.sum()
    if fn_count == 0 and fp_count == 0:
        return None, None
    if fn_count == 0:
        error_region, is_fg = fp_region, False
    elif fp_count == 0:
        error_region, is_fg = fn_region, True
    else:
        is_fg = random.random() < fn_count / (fn_count + fp_count)
        error_region = fn_region if is_fg else fp_region
    center = sample_point_from_error_region(error_region, center_biased=True)
    return center, is_fg


# ═══════════════════════════════════════════════════════════════════════════════
# Bug C 修复：使用 eval 完全相同的 click 生成算法
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_edt_safe_eval_style(error_component):
    """匹配 scripts/eval_bottleneck_attn.py 的 compute_edt_safe."""
    from scipy.ndimage import distance_transform_edt
    coords = np.argwhere(error_component)
    if len(coords) == 0:
        return np.zeros_like(error_component, dtype=np.float64)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0) + 1
    crop_shape = max_c - min_c
    padding = np.maximum((crop_shape * 0.25).astype(int), 1)
    padded_shape = crop_shape + 2 * padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)
    s = tuple(slice(padding[d], padding[d] + crop_shape[d]) for d in range(3))
    center_crop[s] = error_component[
        min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    edt = distance_transform_edt(center_crop)
    dist_cropped = edt[s]
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]] = dist_cropped
    return dist_full


def _sample_coord_eval_style(edt):
    """匹配 eval 的 sample_coord — 取 EDT max。

    Bug F 修复：去掉 np.random.choice 的随机性，
    总是取第一个 max coord（确定性，与 eval 完全一致需要 eval 也一样）。
    """
    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)
    return tuple(max_coords[0])


def generate_click_eval_style(per_class_seg, per_class_gt):
    """完全匹配 eval generate_click 的算法。

    Returns: (center, is_fg) or (None, None) if no error
    """
    import cc3d
    error_mask = (per_class_seg != per_class_gt).astype(np.uint8)
    if error_mask.sum() == 0:
        return None, None

    errors = cc3d.connected_components(error_mask, connectivity=26)
    component_sizes = np.bincount(errors.flat)
    component_sizes[0] = 0
    if component_sizes.max() == 0:
        return None, None

    largest_component = (errors == np.argmax(component_sizes))
    edt = _compute_edt_safe_eval_style(largest_component)
    edt = edt * largest_component
    if edt.sum() == 0:
        edt = largest_component.astype(np.float64)
    center = _sample_coord_eval_style(edt)

    # 根据 GT 在该位置的值决定 fg/bg
    is_fg = per_class_gt[center] > 0
    return center, is_fg


def assemble_last_wins(pred_dict, full_shape, label_order):
    """Last-wins assembly：按 label 升序处理，后面的 label 覆盖前面的。

    Args:
        pred_dict: {label: binary_pred (D, H, W)}
        full_shape: image_crop shape
        label_order: 处理顺序（从前到后），最后一个 label 胜出

    Returns:
        assembled: (D, H, W) uint8, 值是 label
    """
    assembled = np.zeros(full_shape, dtype=np.uint8)
    for k in label_order:
        if k in pred_dict and pred_dict[k] is not None:
            assembled[pred_dict[k] > 0] = k
    return assembled


def assemble_max_prob(pred_dict, prob_dict, full_shape, label_order):
    """Max-prob assembly：单一 object 直接保留；overlap 区域取 fg 概率最高的 object。

    Vectorized: stacks per-object FG probability into (K, *shape) with -inf where
    the object's binary mask is 0, then argmax over K. 比 per-voxel 循环快很多.

    Args:
        pred_dict: {label: binary_pred (D, H, W) uint8}
        prob_dict: {label: float prob (D, H, W) float32}, same keys as pred_dict
        full_shape: image_crop shape
        label_order: 标签顺序（用于 ki 索引和稳定 tie-breaking）

    Returns:
        assembled: (D, H, W) uint8
    """
    K = len(label_order)
    if K == 0:
        return np.zeros(full_shape, dtype=np.uint8)

    probs_stacked = np.full((K, *full_shape), -np.inf, dtype=np.float32)
    has_any = np.zeros(full_shape, dtype=bool)
    for ki, k in enumerate(label_order):
        if k not in pred_dict or pred_dict[k] is None:
            continue
        mask = pred_dict[k] > 0
        if k in prob_dict and prob_dict[k] is not None:
            probs_stacked[ki][mask] = prob_dict[k][mask]
        else:
            probs_stacked[ki][mask] = 0.5
        has_any |= mask

    winner = probs_stacked.argmax(axis=0)
    assembled = np.zeros(full_shape, dtype=np.uint8)
    for ki, k in enumerate(label_order):
        assembled[(winner == ki) & has_any] = k
    return assembled


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class BottleneckAttentionTrainer:

    def __init__(self, gpu: int = 0, lr: float = 3e-4,
                 num_attn_layers: int = 2, num_heads: int = 8,
                 num_rounds: int = 4, frozen: bool = False):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds
        self.frozen = frozen

        # 1. 网络
        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        # 2. Attention module (sparse-only)
        self.attention = BottleneckInteractionAttention(
            feat_dim=320, num_layers=num_attn_layers,
            num_heads=num_heads, num_bg_tokens=4,
        ).to(self.device)

        n_params = count_parameters(self.attention)
        print(f"Attention module: {n_params:,} trainable params (sparse-only)")
        print(f"Training rounds: {num_rounds}")

        if frozen:
            self.attention.eval()
            for p in self.attention.parameters():
                p.requires_grad_(False)
            print("FROZEN mode")

        # 3. Loss
        self.criterion = build_loss(deep_supervision=True).to(self.device)

        # 4. Optimizer
        if not frozen:
            self.optimizer = torch.optim.AdamW(
                self.attention.parameters(), lr=lr, weight_decay=1e-4)
            self.scaler = GradScaler()

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
        """Per-object patches 版本：每个 object 用自己的 click 作为 patch 中心。

        匹配 eval 流程（修复后）：
        1. 在裁切后的 image space (image_crop) 中维护 click 历史
        2. 每个 object 每轮独立提取 192³ patch（centered on its own latest click）
        3. 在 patch 内构造 8ch input（image patch + interactions placed in patch coords）
        4. Forward + loss
        5. Pred 拼回 image_crop full shape
        6. **Bug D 修复**：每轮所有 object forward 完成后做 last-wins assembly，
           下轮的 prev_pred 和 click error source 都从 assembled 中提取
        """
        device = self.device
        K = len(labels)
        full_shape = image_crop.shape  # e.g., (70, 166, 134)
        patch_shape = (PATCH_SIZE,) * 3

        # Bug D 修复：assembly 顺序按 label 升序，最大 label 胜出（匹配 eval）
        labels_sorted = sorted(labels)

        # Per-object state（在 image_crop 全图坐标中）
        click_hist_image = defaultdict(list)
        # **assembled_for_this_obj**: 每个 object 当前的"assembled view"
        # （上一轮 assembly 后提取的该 object 的 binary）
        assembled_per_obj = {k: None for k in labels}

        gt_binaries_full = {k: (gt_crop == k).astype(np.float32) for k in labels}

        if not self.frozen:
            self.optimizer.zero_grad()

        total_loss_val = 0.0
        n_fwd = 0

        for round_idx in range(self.num_rounds):
            # 本轮 forward 输出（用于结束时做 assembly）
            round_preds = {}

            # ── 生成 click（基于 assembled 视图，匹配 eval）──
            for k in labels:
                gt_k_full = gt_binaries_full[k]
                if round_idx == 0:
                    # Round 0: 从 GT 生成初始 click（用 eval 的算法 — EDT max）
                    edt = _compute_edt_safe_eval_style(gt_k_full > 0)
                    if edt.max() > 0:
                        center = _sample_coord_eval_style(edt * (gt_k_full > 0))
                    else:
                        coords = np.argwhere(gt_k_full > 0)
                        if len(coords) == 0:
                            continue
                        center = tuple(coords[len(coords) // 2])
                    click_hist_image[k].append({
                        'pos_image': center,
                        'is_fg': True, 'round': round_idx,
                    })
                else:
                    # Round 1+: 从 assembled 视图算 error，用 eval 的算法
                    if assembled_per_obj[k] is None:
                        continue
                    per_class_seg = assembled_per_obj[k]
                    per_class_gt = (gt_k_full > 0).astype(np.uint8)
                    center, is_fg = generate_click_eval_style(per_class_seg, per_class_gt)
                    if center is not None:
                        click_hist_image[k].append({
                            'pos_image': center,
                            'is_fg': bool(is_fg), 'round': round_idx,
                        })

            # ── Forward + loss per object ──
            for k in labels:
                if not click_hist_image[k]:
                    continue

                # 1. Patch 中心 = 该 object 最新 click 位置
                patch_center = click_hist_image[k][-1]['pos_image']

                # 2. 提取 image patch 和 GT patch
                image_patch_np = _extract_patch_single(image_crop, patch_center)
                gt_k_patch_np = _extract_patch_single(
                    gt_binaries_full[k], patch_center)

                # 3. 构造 interactions in patch 坐标
                interactions = np.zeros((7, *patch_shape), dtype=np.float32)
                patch_start = [patch_center[d] - PATCH_SIZE // 2 for d in range(3)]

                # Prev pred from previous round (extracted from assembled view, Bug D 修复)
                if assembled_per_obj[k] is not None:
                    interactions[0] = _extract_patch_single(
                        assembled_per_obj[k].astype(np.float32), patch_center)

                # Add all accumulated clicks for this object
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

                # 4. 8ch input
                input_8ch_np = np.concatenate(
                    [image_patch_np[None], interactions], axis=0)[None]
                input_8ch = torch.from_numpy(input_8ch_np).to(device)

                # 5. 构造 token_info（self + other clicks 在 patch 坐标 [0,1]）
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
                        cp_norm = [(c['pos_image'][d] - patch_start[d]) / PATCH_SIZE
                                   for d in range(3)]
                        cp_norm = [max(0.0, min(1.0, x)) for x in cp_norm]
                        other_tokens.append({
                            'pos': torch.tensor(cp_norm, dtype=torch.float32),
                            'role': ROLE_OTHER_FG if c['is_fg'] else ROLE_OTHER_BG,
                            'round': c['round'],
                        })

                token_info = build_token_info(self_tokens, other_tokens)

                # 6. Forward
                with autocast_ctx():
                    with torch.no_grad():
                        skips = self.network.encoder(input_8ch)
                    skips = list(skips)
                    skips[-1] = self.attention(skips[-1], token_info)
                    outputs = self.network.decoder(skips)

                    gt_t = torch.from_numpy(gt_k_patch_np[None, None]).float().to(device)
                    targets = downsample_target_for_ds(gt_t)
                    loss = self.criterion(outputs, targets)

                n_fwd += 1

                if not self.frozen:
                    self.scaler.scale(loss / (K * self.num_rounds)).backward()

                total_loss_val += loss.item()

                # 7. 拼 pred 回 full shape，**Bug E 修复**：保留 prev_pred 在 patch 外的值
                with torch.no_grad():
                    pred_patch = (outputs[0].argmax(1).squeeze(0) == 1
                                  ).cpu().numpy().astype(np.uint8)
                    # 初始化 buffer 为该 object 的 prev_pred (assembled view from prev round)
                    if assembled_per_obj[k] is not None:
                        pred_obj_buffer = assembled_per_obj[k].copy()
                    else:
                        pred_obj_buffer = np.zeros(full_shape, dtype=np.uint8)
                    # In-place 覆盖 patch 区域（patch 外保留 prev_pred 值，匹配 eval）
                    _paste_patch_into_buffer(pred_obj_buffer, pred_patch, patch_center)
                    round_preds[k] = pred_obj_buffer

            # Bug D 修复：本轮所有 object forward 完成后，做 last-wins assembly
            # 然后为每个 object 提取 assembled view 作为下轮的 prev_pred 来源
            if round_preds:
                assembled = assemble_last_wins(round_preds, full_shape, labels_sorted)
                for k in labels:
                    assembled_per_obj[k] = (assembled == k).astype(np.uint8)

        if not self.frozen and n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.attention.parameters(), max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'attention_state_dict': self.attention.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }, path)
        print(f"Saved checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--save_dir', default='experiments/bottleneck_attn')
    args = parser.parse_args()

    files = find_brats_files(args.data_root, max_files=args.num_files)
    if not files:
        print("No BraTS files found!")
        return

    trainer = BottleneckAttentionTrainer(
        gpu=args.gpu, lr=args.lr,
        num_attn_layers=args.num_layers, num_heads=args.num_heads,
        num_rounds=args.num_rounds, frozen=args.frozen,
    )

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch, args.epochs)

        if not args.frozen and loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(
                os.path.join(args.save_dir, 'best.pth'), epoch, loss)

        if not args.frozen and (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, f'epoch_{epoch}.pth'), epoch, loss)

    if not args.frozen:
        trainer.save_checkpoint(
            os.path.join(args.save_dir, 'final.pth'), args.epochs - 1, loss)

    print("Done!")


if __name__ == '__main__':
    main()
