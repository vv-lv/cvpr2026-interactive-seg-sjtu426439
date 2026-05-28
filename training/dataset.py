"""
Layer 2 数据集：NPZ 加载 → patch 提取 → 交互通道生成 → 8ch 网络输入。

设计原则：
- 增强只应用到 image + GT，交互在增强后的坐标系里生成
- 每个 sample 随机选一个 object 做 binary segmentation
- 输出 8ch input + binary GT，可直接喂入网络
"""
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from scipy.ndimage import rotate as ndimage_rotate
from torch.utils.data import Dataset

# ── 常量 ──
PATCH_SIZE = (192, 192, 192)
INTERACTION_DECAY = 0.98  # 匹配推理实际值（inference_session_class.json 无此 key → 默认 0.98）
POINT_RADIUS = 4


def _normalize_like_inference(image: np.ndarray) -> np.ndarray:
    """Z-score 归一化（仅统计 bbox，但保持原图大小）。

    ⚠️ 注意：此函数保持原图大小，零值像素被归一化为非零值。
    推理时 session 会裁切到 bbox 并 padding=0，两者在 patch 边界有差异。
    若需与推理完全一致，请用 `preprocess_like_inference()` 代替。
    """
    nonzero_idx = np.where(image != 0)
    if len(nonzero_idx[0]) == 0:
        return image

    bbox_slices = tuple(slice(idx.min(), idx.max() + 1) for idx in nonzero_idx)
    cropped = image[bbox_slices]

    mean_val = cropped.mean()
    std_val = cropped.std()
    if std_val > 0:
        image = (image - mean_val) / std_val
    else:
        image = image - mean_val
    return image


def augment_full(image: np.ndarray, gt: np.ndarray, p_flip: float = 0.5,
                 p_noise: float = 0.15, p_brightness: float = 0.15,
                 p_gamma: float = 0.3, p_contrast: float = 0.15,
                 p_blur: float = 0.1, p_cutout: float = 0.1) -> tuple:
    """对裁切后的全图做 shape-preserving 增强。

    跳过会改 shape 的操作（rotation, scaling）。
    """
    import random as _random

    for axis in range(3):
        if _random.random() < p_flip:
            image = np.flip(image, axis=axis).copy()
            gt = np.flip(gt, axis=axis).copy()

    if _random.random() < p_noise:
        sigma = _random.uniform(0, 0.15)
        image = image + np.random.normal(0, sigma, image.shape).astype(np.float32)

    if _random.random() < p_brightness:
        factor = _random.uniform(0.7, 1.3)
        image = image * factor

    if _random.random() < p_gamma:
        gamma = _random.uniform(0.7, 1.5)
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normed = (image - img_min) / (img_max - img_min)
            normed = np.clip(normed, 0, 1) ** gamma
            image = normed * (img_max - img_min) + img_min

    if _random.random() < p_contrast:
        mean = image.mean()
        factor = _random.uniform(0.7, 1.3)
        image = (image - mean) * factor + mean

    if _random.random() < p_blur:
        from scipy.ndimage import gaussian_filter
        sigma_blur = _random.uniform(0.5, 1.5)
        image = gaussian_filter(image, sigma=sigma_blur).astype(np.float32)

    if _random.random() < p_cutout:
        D, H, W = image.shape
        cd = _random.randint(1, max(1, D // 6))
        ch = _random.randint(1, max(1, H // 6))
        cw = _random.randint(1, max(1, W // 6))
        d0 = _random.randint(0, D - cd)
        h0 = _random.randint(0, H - ch)
        w0 = _random.randint(0, W - cw)
        image[d0:d0+cd, h0:h0+ch, w0:w0+cw] = 0.0

    return image.astype(np.float32), gt


def preprocess_like_inference(image: np.ndarray, gt: np.ndarray = None):
    """**精确**匹配推理 session 的预处理：裁切到非零 bbox + 归一化。

    推理流程 (inference_session._background_set_image):
      1. 找到所有非零体素的 bounding box
      2. 裁剪图像到该 bbox
      3. 在裁剪后的图像上计算 mean/std
      4. 归一化
      5. 后续提取 patch 时在 bbox 外 padding=0

    Args:
        image: (D, H, W) float32
        gt: (D, H, W) uint8 (optional), 如果提供则同步裁切

    Returns:
        image_cropped: (D', H', W') normalized, 已裁切到 nonzero bbox
        gt_cropped: (D', H', W') uint8, 同步裁切 (if gt is not None)
        bbox_min: [z0, y0, x0] bbox 在原图中的起点（用于坐标转换）
    """
    image = image.astype(np.float32)
    nonzero_idx = np.where(image != 0)
    if len(nonzero_idx[0]) == 0:
        bbox_min = [0, 0, 0]
        return image, (gt if gt is not None else None), bbox_min

    bbox_min = [int(idx.min()) for idx in nonzero_idx]
    bbox_max = [int(idx.max() + 1) for idx in nonzero_idx]
    bbox_slices = tuple(slice(bbox_min[d], bbox_max[d]) for d in range(3))

    image_cropped = image[bbox_slices].copy()
    mean_val = image_cropped.mean()
    std_val = image_cropped.std()
    if std_val > 0:
        image_cropped = (image_cropped - mean_val) / std_val
    else:
        image_cropped = image_cropped - mean_val

    if gt is not None:
        gt_cropped = gt[bbox_slices].copy()
        return image_cropped, gt_cropped, bbox_min

    return image_cropped, None, bbox_min


# ═══════════════════════════════════════════════════════════════════════════════
# 交互通道生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_bbox_channel(gt_binary: np.ndarray, jitter_frac: float = 0.05) -> np.ndarray:
    """从 binary GT 生成 bbox 交互通道（ch2, bbox_fg）。

    Args:
        gt_binary: (D, H, W) binary mask of one object
        jitter_frac: bbox 边界抖动比例（论文 ±5%）

    Returns:
        bbox_channel: (D, H, W) float32, bbox 内=1, 外=0
    """
    coords = np.argwhere(gt_binary > 0)
    if len(coords) == 0:
        return np.zeros_like(gt_binary, dtype=np.float32)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    sizes = maxs - mins + 1

    bbox_channel = np.zeros_like(gt_binary, dtype=np.float32)
    slices = []
    for d in range(3):
        jitter = int(sizes[d] * jitter_frac)
        lo = max(0, mins[d] - random.randint(0, max(jitter, 1)))
        hi = min(gt_binary.shape[d], maxs[d] + 1 + random.randint(0, max(jitter, 1)))
        slices.append(slice(lo, hi))
    bbox_channel[tuple(slices)] = 1.0
    return bbox_channel


def generate_point_blob(shape: tuple, center: tuple, radius: int = POINT_RADIUS) -> np.ndarray:
    """生成距离变换 blob（复现论文的 soft sphere）。

    与 nnInteractive/interaction/point.py 的 build_point() 一致：
    ball() → distance_transform_edt() → 归一化 [0,1]。

    Args:
        shape: (D, H, W) 输出 shape
        center: (z, y, x) 中心坐标
        radius: 球半径

    Returns:
        blob: (D, H, W) float32, 中心=1 向外衰减到 0
    """
    blob = np.zeros(shape, dtype=np.float32)

    # 计算 bbox，clip 到边界
    slices_blob = []
    slices_out = []
    for d in range(3):
        lo_blob = max(0, center[d] - radius)
        hi_blob = min(shape[d], center[d] + radius + 1)
        lo_out = lo_blob - (center[d] - radius)
        hi_out = lo_out + (hi_blob - lo_blob)
        slices_blob.append(slice(lo_blob, hi_blob))
        slices_out.append(slice(lo_out, hi_out))

    # 构建球体
    ball_size = 2 * radius + 1
    ball = np.zeros((ball_size, ball_size, ball_size), dtype=np.uint8)
    zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    ball[(zz**2 + yy**2 + xx**2) <= radius**2] = 1

    # 距离变换 → 归一化
    if ball.sum() > 0:
        dt = distance_transform_edt(ball)
        dt = dt / dt.max() if dt.max() > 0 else dt
    else:
        dt = ball.astype(np.float32)

    # Crop 并放置
    crop = dt[tuple(slices_out)]
    existing = blob[tuple(slices_blob)]
    blob[tuple(slices_blob)] = np.maximum(existing, crop)

    return blob


def sample_point_from_error(gt_binary: np.ndarray, pred_binary: np.ndarray,
                            center_biased: bool = True) -> Optional[Tuple[tuple, bool]]:
    """从预测错误区域采样一个 click 点。

    Args:
        gt_binary: (D, H, W) GT binary mask
        pred_binary: (D, H, W) predicted binary mask
        center_biased: 是否用 EDT center-biased 采样（论文 alpha=8）

    Returns:
        (center, is_fg): center=(z,y,x), is_fg=True 表示 fg click（FN 区域），False 表示 bg click（FP 区域）
        如果无错误区域返回 None
    """
    fn_region = (gt_binary > 0) & (pred_binary == 0)  # False Negative → fg click
    fp_region = (gt_binary == 0) & (pred_binary > 0)  # False Positive → bg click

    fn_count = fn_region.sum()
    fp_count = fp_region.sum()

    if fn_count == 0 and fp_count == 0:
        return None

    # 按面积比例选 FN 或 FP（论文做法）
    if fn_count == 0:
        error_region, is_fg = fp_region, False
    elif fp_count == 0:
        error_region, is_fg = fn_region, True
    else:
        is_fg = random.random() < fn_count / (fn_count + fp_count)
        error_region = fn_region if is_fg else fp_region

    coords = np.argwhere(error_region)
    if len(coords) == 0:
        return None

    if center_biased and len(coords) > 1:
        # EDT center-biased sampling（论文 alpha=8）
        dt = distance_transform_edt(error_region)
        dt_vals = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
        alpha = 8
        weights = dt_vals ** alpha
        total = weights.sum()
        if total > 0:
            weights /= total
            idx = np.random.choice(len(coords), p=weights)
        else:
            idx = random.randint(0, len(coords) - 1)
    else:
        idx = random.randint(0, len(coords) - 1)

    center = tuple(coords[idx])
    return center, is_fg


def simulate_imperfect_pred(gt_binary: np.ndarray) -> np.ndarray:
    """模拟不完美的预测（用于生成 follow-up 交互的错误区域）。

    随机选择腐蚀或膨胀 GT，制造 FN 或 FP 错误。
    """
    if gt_binary.sum() == 0:
        return gt_binary.copy()

    # 随机腐蚀/膨胀 1-3 次
    iterations = random.randint(1, 3)
    if random.random() < 0.5:
        pred = binary_erosion(gt_binary, iterations=iterations).astype(np.uint8)
    else:
        pred = binary_dilation(gt_binary, iterations=iterations).astype(np.uint8)

    return pred


def build_interaction_channels(gt_binary: np.ndarray, shape: tuple,
                               include_bbox: bool = True,
                               include_followup: bool = True) -> np.ndarray:
    """构建 7 通道交互 tensor。

    通道布局（对应网络输入 ch1-ch7）：
      [0] prev_pred    [1] bbox_fg    [2] bbox_bg
      [3] point_fg     [4] point_bg   [5] scribble_fg  [6] scribble_bg

    Args:
        gt_binary: (D, H, W) 当前 object 的 binary GT
        shape: (D, H, W) patch shape
        include_bbox: 是否生成 bbox 初始 prompt
        include_followup: 是否生成 follow-up point click

    Returns:
        interactions: (7, D, H, W) float32
    """
    interactions = np.zeros((7, *shape), dtype=np.float32)

    if gt_binary.sum() == 0:
        return interactions

    # ch1: bbox_fg（初始 prompt）
    if include_bbox:
        interactions[1] = generate_bbox_channel(gt_binary, jitter_frac=0.05)

    # Follow-up: 模拟一次预测错误 → 采样 click
    if include_followup:
        fake_pred = simulate_imperfect_pred(gt_binary)

        # prev_pred = fake_pred（模拟上一轮的不完美预测）
        interactions[0] = fake_pred.astype(np.float32)

        # bbox 衰减（因为这是第二轮了）
        interactions[1] *= INTERACTION_DECAY

        # 从错误区域采样 point
        result = sample_point_from_error(gt_binary, fake_pred)
        if result is not None:
            center, is_fg = result
            blob = generate_point_blob(shape, center, radius=POINT_RADIUS)
            if is_fg:
                interactions[3] = blob  # point_fg
            else:
                interactions[4] = blob  # point_bg

    return interactions


# ═══════════════════════════════════════════════════════════════════════════════
# Patch 提取
# ═══════════════════════════════════════════════════════════════════════════════

def extract_patch(image: np.ndarray, gt: np.ndarray, patch_size: tuple,
                  target_label: int) -> Tuple[np.ndarray, np.ndarray]:
    """从 image/gt 中提取 patch，中心偏向 target_label 的前景。

    Args:
        image: (D, H, W) float32 原始图像
        gt: (D, H, W) uint8 多类别标签
        patch_size: (pD, pH, pW)
        target_label: 目标类别 ID

    Returns:
        image_patch: (pD, pH, pW) float32
        gt_patch: (pD, pH, pW) uint8 (binary: target_label → 1, else → 0)
    """
    D, H, W = image.shape
    pD, pH, pW = patch_size

    # 找目标前景体素
    fg_coords = np.argwhere(gt == target_label)
    if len(fg_coords) > 0:
        # Center-biased：随机选一个前景体素作为中心
        idx = random.randint(0, len(fg_coords) - 1)
        center = fg_coords[idx]
    else:
        # 无前景，随机中心
        center = np.array([D // 2, H // 2, W // 2])

    # 计算 patch 范围，clip 到边界
    starts = []
    ends = []
    for d in range(3):
        half = patch_size[d] // 2
        s = max(0, min(center[d] - half, image.shape[d] - patch_size[d]))
        e = s + patch_size[d]
        if e > image.shape[d]:
            e = image.shape[d]
            s = max(0, e - patch_size[d])
        starts.append(s)
        ends.append(e)

    slices = tuple(slice(s, e) for s, e in zip(starts, ends))
    img_patch = image[slices]
    gt_patch = gt[slices]

    # Pad if image smaller than patch
    pad_widths = [(0, max(0, patch_size[d] - img_patch.shape[d])) for d in range(3)]
    if any(pw[1] > 0 for pw in pad_widths):
        img_patch = np.pad(img_patch, pad_widths, mode='constant', constant_values=0)
        gt_patch = np.pad(gt_patch, pad_widths, mode='constant', constant_values=0)

    # Binary: target_label → 1
    gt_binary = (gt_patch == target_label).astype(np.uint8)

    return img_patch, gt_binary


# ═══════════════════════════════════════════════════════════════════════════════
# 数据增强（应用于 image + GT，交互在增强后生成）
# ═══════════════════════════════════════════════════════════════════════════════

def _zoom_array(arr, zoom_factors, order):
    """Zoom a 3D array with given factors per axis."""
    from scipy.ndimage import zoom as ndimage_zoom
    return ndimage_zoom(arr, zoom_factors, order=order, mode='constant', cval=0)


def augment_patch(image: np.ndarray, gt_binary: np.ndarray,
                  layer3: bool = False) -> tuple:
    """对 image 和 GT 做数据增强。

    空间变换同步应用到 image 和 GT。
    强度变换只应用到 image。
    layer3=True 启用论文扩展增强。

    论文增强（arXiv 2503.08373 Appendix A1）：
    - Rotation ±30° (p=0.2)
    - Scaling [0.5, 2.0], 独立轴缩放 p=0.6 (p=0.3)  [论文扩展]
    - Transpose 随机轴转置 (p=0.5)  [论文扩展]
    - Intensity inversion (p=0.1)  [论文扩展]
    - Mirror 三轴翻转 (always)
    - Gaussian noise (p=0.1)
    - Gaussian blur σ=0.5-1.0 (p=0.2)
    - Brightness ×0.75-1.25 (p=0.15)
    - Low-resolution sim zoom 0.5-1.0 (p=0.25)
    - Gamma 0.7-1.5 (p=0.3)
    """
    from scipy.ndimage import gaussian_filter

    target_shape = image.shape  # remember for resize-back after scaling

    # --- 空间变换 ---

    # 随机翻转（三轴独立, always）
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            gt_binary = np.flip(gt_binary, axis=axis).copy()

    # [论文扩展] 随机轴转置 (p=0.5)
    if layer3 and random.random() < 0.5:
        axes = list(range(3))
        random.shuffle(axes)
        image = np.transpose(image, axes).copy()
        gt_binary = np.transpose(gt_binary, axes).copy()

    # 随机旋转 (p=0.2)
    if random.random() < 0.2:
        angle = random.uniform(-30, 30)
        axes_pairs = [(0, 1), (0, 2), (1, 2)]
        axes = axes_pairs[random.randint(0, 2)]
        image = ndimage_rotate(image, angle, axes=axes, reshape=False,
                               order=3, mode='constant', cval=0)
        gt_binary = ndimage_rotate(gt_binary.astype(np.float32), angle, axes=axes,
                                    reshape=False, order=0, mode='constant', cval=0)
        gt_binary = np.round(gt_binary).astype(np.uint8)

    # [论文扩展] Scaling [0.5, 2.0] (p=0.3)
    if layer3 and random.random() < 0.3:
        if random.random() < 0.6:
            # 独立轴缩放
            zoom_factors = [random.uniform(0.5, 2.0) for _ in range(3)]
        else:
            # 等比缩放
            s = random.uniform(0.5, 2.0)
            zoom_factors = [s, s, s]
        image = _zoom_array(image, zoom_factors, order=3)
        gt_binary = _zoom_array(gt_binary.astype(np.float32), zoom_factors, order=0)
        gt_binary = np.round(gt_binary).astype(np.uint8)
        # Crop or pad back to target_shape
        result_img = np.zeros(target_shape, dtype=np.float32)
        result_gt = np.zeros(target_shape, dtype=np.uint8)
        slices_src = []
        slices_dst = []
        for d in range(3):
            src_size = image.shape[d]
            dst_size = target_shape[d]
            if src_size >= dst_size:
                start = (src_size - dst_size) // 2
                slices_src.append(slice(start, start + dst_size))
                slices_dst.append(slice(0, dst_size))
            else:
                start = (dst_size - src_size) // 2
                slices_src.append(slice(0, src_size))
                slices_dst.append(slice(start, start + src_size))
        result_img[tuple(slices_dst)] = image[tuple(slices_src)]
        result_gt[tuple(slices_dst)] = gt_binary[tuple(slices_src)]
        image = result_img
        gt_binary = result_gt

    # --- 强度变换（只对 image）---

    # [论文扩展] Intensity inversion (p=0.1)
    if layer3 and random.random() < 0.1:
        image = -image

    # 高斯噪声 (p=0.1)
    if random.random() < 0.1:
        image = image + np.random.normal(0, 0.1, image.shape).astype(np.float32)

    # Gaussian blur (p=0.2, σ=0.5-1.0)
    if random.random() < 0.2:
        sigma = random.uniform(0.5, 1.0)
        image = gaussian_filter(image, sigma=sigma).astype(np.float32)

    # 亮度（乘性）(p=0.15)
    if random.random() < 0.15:
        image = image * random.uniform(0.75, 1.25)

    # Low-resolution simulation (p=0.25)
    if random.random() < 0.25:
        zoom_factor = random.uniform(0.5, 1.0)
        low_shape = [max(1, int(s * zoom_factor)) for s in image.shape]
        from scipy.ndimage import zoom as ndimage_zoom
        low = ndimage_zoom(image, [l / s for l, s in zip(low_shape, image.shape)],
                           order=3, mode='constant')
        image = ndimage_zoom(low, [s / l for s, l in zip(image.shape, low.shape)],
                             order=3, mode='constant').astype(np.float32)

    # Gamma 校正 (p=0.3)
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.5)
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_01 = (image - img_min) / (img_max - img_min)
            image_01 = np.clip(image_01, 1e-6, 1.0)
            image = np.power(image_01, gamma) * (img_max - img_min) + img_min

    return image, gt_binary


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class NNInteractiveTrainDataset(Dataset):
    """Layer 2 训练数据集。

    每个 __getitem__ 返回:
        input_8ch: (8, pD, pH, pW) float32 — 网络输入
        gt_binary: (1, pD, pH, pW) float32 — binary GT
        meta: dict — 文件名、target_label 等
    """

    def __init__(self, data_dir: str, patch_size: tuple = PATCH_SIZE,
                 max_files: int = 0, include_followup: bool = True,
                 augment: bool = True):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.include_followup = include_followup
        self.augment = augment
        self.files = sorted(self.data_dir.rglob("*.npz"))
        if max_files > 0:
            random.shuffle(self.files)  # shuffle 确保模态多样性
            self.files = self.files[:max_files]
        assert len(self.files) > 0, f"No NPZ files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        image = data['imgs'].astype(np.float32)  # (D, H, W), [0, 255]
        gt = data['gts'].astype(np.uint8)         # (D, H, W)

        # Image-level Z-score 归一化（精确匹配推理代码 set_image()）
        # 推理: crop to nonzero bbox → z-score on ENTIRE crop (包含 crop 内的零值)
        # 训练: 同样 crop to nonzero bbox → z-score on entire crop → 应用到全图
        image = _normalize_like_inference(image)

        # 随机选一个非背景 object
        labels = np.unique(gt)
        labels = labels[labels > 0]
        if len(labels) == 0:
            target_label = 1
        else:
            target_label = int(np.random.choice(labels))

        # 提取 patch（center-biased 在目标 object 上）
        for _attempt in range(5):
            img_patch, gt_binary = extract_patch(image, gt, self.patch_size, target_label)
            if gt_binary.sum() > 0:
                break
            if len(labels) > 1:
                target_label = int(np.random.choice(labels))

        # 数据增强（只对 image + GT，交互在增强后生成）
        if self.augment:
            img_patch, gt_binary = augment_patch(img_patch, gt_binary)

        # 生成 7 通道交互（在增强后的坐标系里）
        interactions = build_interaction_channels(
            gt_binary, self.patch_size,
            include_bbox=True,
            include_followup=self.include_followup
        )

        # 拼接 8 通道输入: [image(1), interactions(7)]
        input_8ch = np.concatenate([
            img_patch[np.newaxis],  # (1, D, H, W)
            interactions,            # (7, D, H, W)
        ], axis=0)  # (8, D, H, W)

        return {
            'input': torch.from_numpy(input_8ch),
            'target': torch.from_numpy(gt_binary[np.newaxis].astype(np.float32)),
            'name': self.files[idx].stem,
            'label': target_label,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 Dataset（返回 image + gt，不含交互）
# ═══════════════════════════════════════════════════════════════════════════════

class Layer3Dataset(Dataset):
    """Layer 3 数据集：返回 image patch + binary GT，交互在训练循环中生成。

    与 NNInteractiveTrainDataset 的区别：
    - 不生成交互通道（交互基于模型预测，需要在训练循环中生成）
    - 返回 image (1ch) 和 gt (1ch)，不是 8ch input
    - 使用论文扩展增强
    """

    def __init__(self, data_dir: str, patch_size: tuple = PATCH_SIZE,
                 max_files: int = 0, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.files = sorted(self.data_dir.rglob("*.npz"))
        if max_files > 0:
            random.shuffle(self.files)
            self.files = self.files[:max_files]
        assert len(self.files) > 0, f"No NPZ files found in {data_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        image = data['imgs'].astype(np.float32)
        gt = data['gts'].astype(np.uint8)

        # Image-level Z-score（精确匹配推理 set_image()）
        image = _normalize_like_inference(image)

        # 随机选 object
        labels = np.unique(gt)
        labels = labels[labels > 0]
        if len(labels) == 0:
            target_label = 1
        else:
            target_label = int(np.random.choice(labels))

        # 提取 patch（重试确保有前景）
        for _attempt in range(5):
            img_patch, gt_binary = extract_patch(image, gt, self.patch_size, target_label)
            if gt_binary.sum() > 0:
                break
            if len(labels) > 1:
                target_label = int(np.random.choice(labels))

        # 增强（论文扩展版）
        if self.augment:
            img_patch, gt_binary = augment_patch(img_patch, gt_binary, layer3=True)

        return {
            'image': torch.from_numpy(img_patch[np.newaxis].copy()),   # (1, D, H, W)
            'target': torch.from_numpy(gt_binary[np.newaxis].astype(np.float32)),  # (1, D, H, W)
            'name': self.files[idx].stem,
            'label': target_label,
        }
