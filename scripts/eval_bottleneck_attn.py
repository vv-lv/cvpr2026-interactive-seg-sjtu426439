#!/usr/bin/env python3
"""
Bottleneck Interaction Attention 评估脚本。

通过 monkey-patch session.network 的方式集成 attention module，
最小化对 nnInteractive 推理流程的改动。

Usage:
    python scripts/eval_bottleneck_attn.py \
        --attn_ckpt experiments/bottleneck_attn_v1/best.pth \
        --cases 10 --gpu 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import cc3d
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
EVAL_DIR = PROJECT_ROOT / "evaluation" / "CVPR-MedSegFMCompetition"
sys.path.insert(0, str(EVAL_DIR))

# ── numpy 兼容性 ──
try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from SurfaceDice import compute_dice_coefficient
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache

from training.bottleneck_attention import (
    BottleneckInteractionAttention, build_token_info, normalize_pos,
    ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG, PATCH_SIZE,
)
from training.trainer import autocast_ctx

# ── 默认路径 ──
DEFAULT_CHECKPOINT = Path(
    "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models"
    "/nnInteractive_v1.0_fold_all"
)
DEFAULT_VAL_NPZ = PROJECT_ROOT / "data" / "3D_val_npz"
DEFAULT_VAL_GT = PROJECT_ROOT / "data" / "3D_val_gt" / "3D_val_gt_interactive"

N_CLICKS = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]:
        dsc.append(compute_dice_coefficient(gt == i, seg == i))
    return np.mean(dsc) if dsc else 0.0


def compute_auc(dsc_list):
    """DSC AUC: sum(DSC_i) - DSC_first/2 - DSC_last/2"""
    if len(dsc_list) < 2:
        return sum(dsc_list)
    return sum(dsc_list) - dsc_list[0] / 2 - dsc_list[-1] / 2


# ═══════════════════════════════════════════════════════════════════════════════
# Click Generation (匹配竞赛评估协议)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_coord(edt):
    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)
    # Bug F 修复：去掉随机性，取第一个 max coord（与训练一致）
    return tuple(max_coords[0])


def compute_edt_safe(error_component):
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


def generate_click(segs, gts, clicks_cls, clicks_order):
    """生成一轮 click（匹配竞赛评估协议 CVPR25_iter_eval.py）。"""
    unique_gts = np.sort(pd.unique(gts.ravel()))
    for ind, cls in enumerate(sorted(unique_gts[1:])):
        segs_cls = (segs == cls).astype(np.uint8)
        gts_cls = (gts == cls).astype(np.uint8)
        error_mask = (segs_cls != gts_cls).astype(np.uint8)
        if np.sum(error_mask) > 0:
            errors = cc3d.connected_components(error_mask, connectivity=26)
            sizes = np.bincount(errors.flat)
            sizes[0] = 0
            largest = (errors == np.argmax(sizes))
            edt = compute_edt_safe(largest)
            edt *= largest
            if np.sum(edt) == 0:
                edt = largest.astype(np.float64)
            center = sample_coord(edt)
            if gts_cls[center] == 0:
                clicks_cls[ind]['bg'].append(list(center))
                clicks_order[ind].append('bg')
            else:
                clicks_cls[ind]['fg'].append(list(center))
                clicks_order[ind].append('fg')
        else:
            clicks_order[ind].append(None)


# ═══════════════════════════════════════════════════════════════════════════════
# Network Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionNetworkWrapper:
    """替换 session.network，注入 bottleneck attention。

    在每次 forward 时：
      encoder(8ch) → attention(bottleneck, token_info) → decoder → logits

    关键：将 click 坐标从 image 空间映射到当前 192³ crop 空间，
    匹配训练时的坐标系（clicks 在 patch [0,1] 空间）。
    """

    def __init__(self, network, attention_module, session, device, delta_scale=1.0):
        self.network = network
        self.attention = attention_module
        self.session = session  # 用于获取 crop 信息
        self.device = device
        self.delta_scale = delta_scale  # 人工缩放 delta（调试用）
        # 原始 image 空间的 click 列表（不含坐标归一化）
        self._raw_clicks = []  # [{'pos_image': (z,y,x), 'role': int, 'round': int}]
        self._other_raw_clicks = []

        # 确保 decoder 不做 deep supervision
        if hasattr(self.network, 'decoder'):
            self.network.decoder.deep_supervision = False

    def set_raw_clicks(self, self_clicks, other_clicks):
        """设置 image 空间的 raw click 位置。坐标映射在 __call__ 中完成。"""
        self._raw_clicks = self_clicks
        self._other_raw_clicks = other_clicks

    def _map_clicks_to_crop(self, raw_clicks, center, bbox_start):
        """将 image 空间 clicks 映射到当前 crop 的 [0,1] 空间。

        关键：nnInteractive session 的 patch 总是以 click 为中心，
        通过 crop_and_pad_nd 处理 padding。所以 patch 的 src start 可能是负数。

        Args:
            raw_clicks: [{'pos_image': (z,y,x), 'role': int, 'round': int}]
            center: crop center in preprocessed image coords
            bbox_start: [z0, y0, x0] of preprocessed bbox in original image
        """
        mapped = []
        for c in raw_clicks:
            pos_img = c['pos_image']
            # image → preprocessed: subtract bbox start
            pos_pre = [pos_img[d] - bbox_start[d] for d in range(3)]
            # preprocessed → patch: subtract crop start (signed, can be negative)
            # The session always centers patch on click, so click at center → patch[96]
            crop_start = [center[d] - PATCH_SIZE // 2 for d in range(3)]
            pos_crop = [(pos_pre[d] - crop_start[d]) / PATCH_SIZE for d in range(3)]
            pos_norm = torch.tensor(pos_crop, dtype=torch.float32).clamp(0, 1)
            mapped.append({
                'pos': pos_norm,
                'role': c['role'],
                'round': c['round'],
            })
        return mapped

    def __call__(self, x):
        # 获取当前 crop 信息
        center = None
        if (hasattr(self.session, 'new_interaction_centers') and
                self.session.new_interaction_centers):
            center = self.session.new_interaction_centers[-1]

        bbox_start = [0, 0, 0]
        if hasattr(self.session, 'preprocessed_props'):
            bbox = self.session.preprocessed_props.get('bbox_used_for_cropping', None)
            if bbox is not None:
                bbox_start = [int(b[0]) for b in bbox]

        # 构建正确坐标的 token_info
        if center is not None and (self._raw_clicks or self._other_raw_clicks):
            mapped_self = self._map_clicks_to_crop(
                self._raw_clicks, center, bbox_start)
            mapped_other = self._map_clicks_to_crop(
                self._other_raw_clicks, center, bbox_start)
            token_info = build_token_info(mapped_self, mapped_other)
        else:
            token_info = {'clicks': []}

        with torch.no_grad():
            skips = self.network.encoder(x)
        skips = list(skips)
        with torch.no_grad():
            attn_out = self.attention(skips[-1], token_info)
            if self.delta_scale != 1.0:
                # 缩放 delta: new_out = input + scale * (output - input)
                delta = attn_out - skips[-1]
                attn_out = skips[-1] + self.delta_scale * delta
            skips[-1] = attn_out
        with torch.no_grad():
            out = self.network.decoder(skips)
        return out

    def __getattr__(self, name):
        return getattr(self.network, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_case(npz_path, gt_path, attn_ckpt, checkpoint_dir, device,
                  use_attention=True):
    """评估一个 case：baseline vs attention，各跑完整 6 轮交互。"""
    np.random.seed(42)

    data = np.load(npz_path, allow_pickle=True)
    image = data['imgs']
    gt = np.load(gt_path)['gts']
    has_bbox = 'boxes' in data and data['boxes'].shape[0] > 0

    unique_labels = np.sort(pd.unique(gt.ravel()))
    num_classes = len(unique_labels) - 1
    if num_classes == 0:
        return None

    results = {}

    for condition in (['baseline', 'attention'] if use_attention else ['baseline']):
        np.random.seed(42)

        # Session 初始化
        session = nnInteractiveInferenceSession(
            device=device, use_torch_compile=False, verbose=False,
            do_autozoom=True, use_pinned_memory=True,
        )
        session.initialize_from_trained_model_folder(
            model_training_output_dir=str(checkpoint_dir), use_fold='all')

        if condition == 'attention' and attn_ckpt is not None:
            attn_module = BottleneckInteractionAttention(
                feat_dim=320, num_layers=2, num_heads=8, num_bg_tokens=4)
            ckpt = torch.load(attn_ckpt, map_location=device, weights_only=False)
            missing, unexpected = attn_module.load_state_dict(
                ckpt['attention_state_dict'], strict=False)
            if missing:
                print(f"  [warn] Missing attn keys: {missing}")
            attn_module.to(device).eval()

            # 如果 checkpoint 包含 LoRA 权重，也加载
            if 'lora_state_dict' in ckpt and ckpt['lora_state_dict']:
                from training.lora import apply_lora_to_decoder, LoRAConv3d
                apply_lora_to_decoder(session.network.decoder,
                                      target_stages=[3, 4], rank=4)
                session.network.decoder.to(device)
                # 加载 LoRA 权重
                lora_state = ckpt['lora_state_dict']
                for name, module in session.network.named_modules():
                    if isinstance(module, LoRAConv3d):
                        for key, val in lora_state.items():
                            if name in key:
                                if 'lora_A' in key:
                                    module.lora_A.weight.data.copy_(val.to(device))
                                elif 'lora_B' in key:
                                    module.lora_B.weight.data.copy_(val.to(device))
                print(f"  LoRA loaded: {len(lora_state)} tensors")

            wrapper = AttentionNetworkWrapper(
                session.network, attn_module, session, device,
                delta_scale=getattr(evaluate_case, '_delta_scale', 1.0))
            session.network = wrapper
        else:
            wrapper = None

        session.set_image(image[None].astype(np.float32))
        target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
        session.set_target_buffer(target_buffer)

        clicks_cls = [{'fg': [], 'bg': []} for _ in range(num_classes)]
        clicks_order = [[] for _ in range(num_classes)]
        click_positions = {}  # {oid: [{'pos', 'role', 'round'}, ...]}
        prev_masks = {}       # {oid: [mask_array, ...]}
        # GT label 列表（sorted non-zero），oid→label 映射
        gt_labels = sorted(unique_labels[1:].tolist())

        dsc_list = []

        # Round 0: initial interaction
        for oid in range(1, num_classes + 1):
            click_positions[oid] = []
            prev_masks[oid] = []
            session.reset_interactions()

            if has_bbox:
                boxes = data['boxes']
                box = boxes[oid - 1] if oid - 1 < len(boxes) else None
                if box is not None:
                    session.add_bbox_interaction(
                        [box[0], box[2], box[4]],
                        [box[1], box[3], box[5]],
                        include_interaction=True, run_prediction=False)
            # No bbox → will get first click in round 1

            if wrapper is not None:
                other_raw = _gather_other_clicks(click_positions, oid)
                wrapper.set_raw_clicks(click_positions[oid], other_raw)

            if has_bbox:
                session._predict()

        if has_bbox:
            # Round 0 assembly: 所有 object 已跑完，target_buffer 有最后 object 的 binary
            # 需要重新 assemble — 但 round 0 每个 oid 的 binary mask 没有存储
            # 简化：round 0 直接用 target_buffer（假设 bbox 场景的 assembly 基本正确）
            # TODO: 如果需要精确评估，需要在 round 0 也做 per-object assembly
            seg_r0 = target_buffer.numpy().copy()
            # 简单映射：如果只有 0/1 值，赋 gt_labels[0]
            if seg_r0.max() == 1 and num_classes > 1:
                # 对 bbox round 0 也做 per-object assembly
                assembled_r0 = np.zeros_like(seg_r0)
                for oid in range(1, num_classes + 1):
                    gt_label = gt_labels[oid - 1]
                    target_buffer.zero_()
                    session.reset_interactions()
                    boxes = data['boxes']
                    box = boxes[oid - 1] if oid - 1 < len(boxes) else None
                    if box is not None:
                        session.add_bbox_interaction(
                            [box[0], box[2], box[4]], [box[1], box[3], box[5]],
                            include_interaction=True, run_prediction=True)
                    obj_pred = (target_buffer.numpy() > 0)
                    assembled_r0[obj_pred] = gt_label
                seg_r0 = assembled_r0
            dsc_r0 = compute_multi_class_dsc(gt, seg_r0)
            dsc_list.append(dsc_r0)
            prev_pred = seg_r0
        else:
            prev_pred = None

        # Rounds 1-5: click refinement
        for round_idx in range(N_CLICKS):
            if prev_pred is not None:
                generate_click(prev_pred, gt, clicks_cls, clicks_order)
            else:
                # No bbox, no prev: generate initial click from GT
                for ind, cls in enumerate(sorted(unique_labels[1:])):
                    gt_cls = (gt == cls).astype(np.uint8)
                    coords = np.argwhere(gt_cls > 0)
                    if len(coords) > 0:
                        edt = compute_edt_safe(gt_cls)
                        center = sample_coord(edt)
                        clicks_cls[ind]['fg'].append(list(center))
                        clicks_order[ind].append('fg')

            # 多类别 assembly 结果
            assembled = np.zeros(image.shape, dtype=np.uint8)
            # Bug G fix: pre-build click_positions for ALL oids before the
            # forward loop, so each oid sees the same "other clicks" set as
            # the training pipeline.
            click_positions = _build_all_click_positions(
                clicks_cls, clicks_order, num_classes)

            for oid in range(1, num_classes + 1):
                gt_label = gt_labels[oid - 1]
                target_buffer.zero_()

                if prev_pred is not None:
                    session.add_initial_seg_interaction(
                        (prev_pred == gt_label).astype(np.uint8),
                        run_prediction=False)
                else:
                    session.reset_interactions()

                # Replay ALL clicks for this object via session API.
                clicks_here = clicks_cls[oid - 1]
                order_here = clicks_order[oid - 1]
                fg_ptr = bg_ptr = 0
                for ri, kind in enumerate(order_here):
                    if kind is None:
                        continue
                    if kind == 'fg':
                        click = clicks_here['fg'][fg_ptr]
                        fg_ptr += 1
                    else:
                        click = clicks_here['bg'][bg_ptr]
                        bg_ptr += 1
                    session.add_point_interaction(
                        click, include_interaction=(kind == 'fg'),
                        run_prediction=False)

                if wrapper is not None:
                    other_raw = _gather_other_clicks(click_positions, oid)
                    wrapper.set_raw_clicks(click_positions[oid], other_raw)

                session.new_interaction_centers = [
                    session.new_interaction_centers[-1]]
                session.new_interaction_zoom_out_factors = [
                    session.new_interaction_zoom_out_factors[-1]]
                session._predict()

                # Capture binary prediction → last-wins assembly
                obj_pred = (target_buffer.numpy() > 0)
                assembled[obj_pred] = gt_label  # 使用 GT label 而非 oid

                # Store prediction mask for this object
                obj_mask = obj_pred.astype(np.float32)
                if len(prev_masks[oid]) >= 3:
                    prev_masks[oid] = prev_masks[oid][-2:]
                prev_masks[oid].append(obj_mask)

            dsc = compute_multi_class_dsc(gt, assembled)
            dsc_list.append(dsc)
            prev_pred = assembled.copy()

        auc = compute_auc(dsc_list)
        results[condition] = {
            'DSC_AUC': auc,
            'DSC_Final': dsc_list[-1],
            'per_round': dsc_list,
        }

        del session
        empty_cache(torch.device(device))

    return results


def _gather_other_clicks(click_positions, oid):
    """收集其他 objects 的 clicks，转换 role 为 OTHER_*。保留 pos_image。"""
    other = []
    for j, clicks in click_positions.items():
        if j == oid:
            continue
        for c in clicks:
            other.append({
                'pos_image': c['pos_image'],
                'role': ROLE_OTHER_FG if c['role'] == ROLE_SELF_FG else ROLE_OTHER_BG,
                'round': c['round'],
            })
    return other


def _build_all_click_positions(clicks_cls, clicks_order, num_classes):
    """Pre-build click_positions for ALL oids at the start of a round.

    Bug G fix: gathering "other clicks" inside the per-oid forward loop sees a
    stale picture (other oids' clicks from the previous round only). Training
    has every object's current-round click before any forward pass, so eval
    must mirror that.
    """
    click_positions = {}
    for oid in range(1, num_classes + 1):
        clicks_here = clicks_cls[oid - 1]
        order_here = clicks_order[oid - 1]
        fg_ptr = bg_ptr = 0
        positions = []
        for ri, kind in enumerate(order_here):
            if kind is None:
                continue
            if kind == 'fg':
                click = clicks_here['fg'][fg_ptr]
                fg_ptr += 1
                role = ROLE_SELF_FG
            else:
                click = clicks_here['bg'][bg_ptr]
                bg_ptr += 1
                role = ROLE_SELF_BG
            positions.append({
                'pos_image': tuple(click), 'role': role, 'round': ri,
            })
        click_positions[oid] = positions
    return click_positions


def _gather_other_masks(prev_masks, oid, round_idx):
    """收集其他 objects 的最新 mask（最多 1 轮）。"""
    masks = []
    for j, ms in prev_masks.items():
        if j == oid or not ms:
            continue
        m = ms[-1]
        masks.append((
            torch.from_numpy(m[None, None].astype(np.float32)),
            max(0, round_idx)
        ))
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_ckpt',
                        default='experiments/bottleneck_attn_v1/best.pth')
    parser.add_argument('--model_dir', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--val_npz', default=str(DEFAULT_VAL_NPZ))
    parser.add_argument('--val_gt', default=str(DEFAULT_VAL_GT))
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--cases', type=int, default=10,
                        help='Number of BraTS cases to evaluate')
    parser.add_argument('--delta_scale', type=float, default=1.0,
                        help='Scale factor for attention delta (debug)')
    parser.add_argument('--output', default='experiments/bottleneck_attn_eval')
    args = parser.parse_args()

    evaluate_case._delta_scale = args.delta_scale
    print(f"Delta scale: {args.delta_scale}")

    device = torch.device(f'cuda:{args.gpu}')
    os.makedirs(args.output, exist_ok=True)

    # 找 BraTS 验证 cases
    val_files = sorted([
        f for f in os.listdir(args.val_npz) if 'BraTS' in f and f.endswith('.npz')
    ])[:args.cases]

    print(f"Evaluating {len(val_files)} BraTS cases on {device}")
    print(f"Attention checkpoint: {args.attn_ckpt}")

    rows = []
    for i, fname in enumerate(val_files):
        npz_path = os.path.join(args.val_npz, fname)
        gt_path = os.path.join(args.val_gt, fname)
        if not os.path.exists(gt_path):
            print(f"  [{i+1}] {fname}: GT not found, skip")
            continue

        t0 = time.time()
        results = evaluate_case(
            npz_path, gt_path, args.attn_ckpt, args.model_dir, device)
        elapsed = time.time() - t0

        if results is None:
            continue

        for cond, r in results.items():
            row = {'case': fname, 'condition': cond,
                   'DSC_AUC': r['DSC_AUC'], 'DSC_Final': r['DSC_Final']}
            rows.append(row)

        bl = results['baseline']
        at = results['attention']
        delta = at['DSC_AUC'] - bl['DSC_AUC']
        print(f"  [{i+1}/{len(val_files)}] {fname}: "
              f"baseline={bl['DSC_AUC']:.3f} attn={at['DSC_AUC']:.3f} "
              f"Δ={delta:+.3f} ({elapsed:.1f}s)")

    # Save results
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output, 'results.csv')
        df.to_csv(csv_path, index=False)

        # Summary
        bl_df = df[df.condition == 'baseline']
        at_df = df[df.condition == 'attention']
        if len(bl_df) > 0 and len(at_df) > 0:
            merged = bl_df.merge(at_df, on='case', suffixes=('_bl', '_at'))
            n = len(merged)
            wins = (merged.DSC_AUC_at > merged.DSC_AUC_bl).sum()
            losses = (merged.DSC_AUC_at < merged.DSC_AUC_bl).sum()
            ties = n - wins - losses
            mean_delta = (merged.DSC_AUC_at - merged.DSC_AUC_bl).mean()
            print(f"\n=== Summary ({n} cases) ===")
            print(f"Mean AUC: baseline={merged.DSC_AUC_bl.mean():.3f} "
                  f"attention={merged.DSC_AUC_at.mean():.3f} "
                  f"Δ={mean_delta:+.3f}")
            print(f"W/L/T: {wins}/{losses}/{ties}")
            print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
