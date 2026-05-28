"""
Layer 3 交互模拟器：基于模型实际预测生成 follow-up 交互。

通道布局（网络输入 ch1-ch7，对应 interactions[0]-[6]）:
  [0] prev_pred    [1] bbox_fg    [2] bbox_bg
  [3] point_fg     [4] point_bg   [5] scribble_fg  [6] scribble_bg

核心改进（vs Layer 2）:
- follow-up 交互基于模型实际预测的 FP/FN，不是 GT 腐蚀/膨胀
- 错误区域做连通域分析，按面积比例选最大组件
- 支持多轮交互，每轮衰减 ×0.98（匹配推理 session 默认值）
"""
import random
from typing import Optional, Tuple

import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt

# ── 常量 ──
INTERACTION_DECAY = 0.98  # 匹配推理实际值（inference_session_class.json 无此 key → 默认 0.98）
POINT_RADIUS = 4
POINT_ALPHA = 8           # center-biased 采样指数


def generate_point_blob(shape: tuple, center: tuple, radius: int = POINT_RADIUS) -> np.ndarray:
    """生成距离变换 soft sphere blob。"""
    blob = np.zeros(shape, dtype=np.float32)
    slices_blob = []
    slices_ball = []
    for d in range(3):
        lo = max(0, center[d] - radius)
        hi = min(shape[d], center[d] + radius + 1)
        lo_ball = lo - (center[d] - radius)
        hi_ball = lo_ball + (hi - lo)
        slices_blob.append(slice(lo, hi))
        slices_ball.append(slice(lo_ball, hi_ball))

    ball_size = 2 * radius + 1
    zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    ball = ((zz**2 + yy**2 + xx**2) <= radius**2).astype(np.uint8)

    if ball.sum() > 0:
        dt = distance_transform_edt(ball)
        dt = dt / dt.max() if dt.max() > 0 else dt
    else:
        dt = ball.astype(np.float32)

    crop = dt[tuple(slices_ball)]
    existing = blob[tuple(slices_blob)]
    blob[tuple(slices_blob)] = np.maximum(existing, crop)
    return blob


def generate_bbox_channel(gt_binary: np.ndarray, jitter_frac: float = 0.05) -> np.ndarray:
    """从 binary GT 生成 bbox 通道（值=1 within bbox）。"""
    coords = np.argwhere(gt_binary > 0)
    if len(coords) == 0:
        return np.zeros_like(gt_binary, dtype=np.float32)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    sizes = maxs - mins + 1

    bbox_ch = np.zeros_like(gt_binary, dtype=np.float32)
    slices = []
    for d in range(3):
        jitter = int(sizes[d] * jitter_frac)
        lo = max(0, mins[d] - random.randint(0, max(jitter, 1)))
        hi = min(gt_binary.shape[d], maxs[d] + 1 + random.randint(0, max(jitter, 1)))
        slices.append(slice(lo, hi))
    bbox_ch[tuple(slices)] = 1.0
    return bbox_ch


def sample_point_from_error_region(error_region: np.ndarray,
                                    center_biased: bool = True,
                                    alpha: int = POINT_ALPHA) -> Optional[tuple]:
    """从错误区域中采样一个点。

    论文做法：对错误区域做连通域分析 → 按面积选最大组件 → EDT center-biased 采样。
    """
    if error_region.sum() == 0:
        return None

    # 连通域分析，按面积选最大组件
    labels = cc3d.connected_components(error_region.astype(np.uint8), connectivity=26)
    component_ids = np.unique(labels)
    component_ids = component_ids[component_ids > 0]

    if len(component_ids) == 0:
        return None

    # 按面积比例随机选一个组件（论文：probability proportional to size）
    sizes = np.array([(labels == cid).sum() for cid in component_ids])
    probs = sizes / sizes.sum()
    chosen_id = np.random.choice(component_ids, p=probs)
    component_mask = (labels == chosen_id)

    coords = np.argwhere(component_mask)

    if center_biased and len(coords) > 1:
        # EDT center-biased 采样 (论文 alpha=8)
        dt = distance_transform_edt(component_mask)
        dt_vals = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
        weights = dt_vals ** alpha
        total = weights.sum()
        if total > 0:
            weights /= total
            idx = np.random.choice(len(coords), p=weights)
        else:
            idx = random.randint(0, len(coords) - 1)
    else:
        idx = random.randint(0, len(coords) - 1)

    return tuple(coords[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# InteractionManager
# ═══════════════════════════════════════════════════════════════════════════════

class InteractionManager:
    """管理 7 通道交互状态，支持多轮交互。

    用法:
        mgr = InteractionManager(shape=(192, 192, 192))
        mgr.set_initial_bbox(gt_binary)          # Round 0
        interactions_r0 = mgr.get_tensor()        # → torch tensor

        mgr.add_followup(pred_binary, gt_binary)  # Round 1 (基于模型预测)
        interactions_r1 = mgr.get_tensor()
    """

    def __init__(self, shape: tuple, decay: float = INTERACTION_DECAY):
        self.shape = shape
        self.decay = decay
        self.interactions = np.zeros((7, *shape), dtype=np.float32)

    def reset(self):
        self.interactions.fill(0)

    def set_initial_bbox(self, gt_binary: np.ndarray, jitter: float = 0.05):
        """生成初始 bbox prompt → ch1 (bbox_fg)。"""
        self.interactions[1] = generate_bbox_channel(gt_binary, jitter_frac=jitter)

    def set_initial_point(self, gt_binary: np.ndarray, is_fg: bool = True):
        """生成初始 point prompt（无 bbox 的 case）。"""
        coords = np.argwhere(gt_binary > 0) if is_fg else np.argwhere(gt_binary == 0)
        if len(coords) == 0:
            return

        # EDT center-biased 采样
        dt = distance_transform_edt(gt_binary if is_fg else (1 - gt_binary))
        dt_vals = dt[coords[:, 0], coords[:, 1], coords[:, 2]]
        weights = dt_vals ** POINT_ALPHA
        total = weights.sum()
        if total > 0:
            weights /= total
            idx = np.random.choice(len(coords), p=weights)
        else:
            idx = random.randint(0, len(coords) - 1)

        center = tuple(coords[idx])
        blob = generate_point_blob(self.shape, center, radius=POINT_RADIUS)
        ch = 3 if is_fg else 4  # point_fg or point_bg
        self.interactions[ch] = np.maximum(self.interactions[ch], blob)

    def set_prev_pred(self, pred_binary: np.ndarray):
        """写入 prev_pred → ch0。"""
        self.interactions[0] = pred_binary.astype(np.float32)

    def add_followup(self, pred_binary: np.ndarray, gt_binary: np.ndarray):
        """基于模型预测误差生成 follow-up 交互。

        这是 Layer 3 的核心：使用模型实际预测（不是 GT 腐蚀/膨胀）。

        流程（匹配推理 session 的 add_point_interaction）：
        1. 写入 prev_pred = 模型预测
        2. 衰减旧 point 交互 ch[3:5] ×0.98（不衰减 bbox ch[1:3]）
        3. 计算 FP/FN 错误区域
        4. 连通域分析 → 面积比例采样 → EDT center-biased 采样
        5. 生成 point blob 写入 ch3/ch4
        """
        # 1. prev_pred = 模型预测
        self.set_prev_pred(pred_binary)

        # 2. 衰减旧 point 交互（只衰减 ch3,ch4）
        # 推理中 add_point_interaction 只衰减 interactions[-4:-2] (point 通道)
        # bbox 通道不受 point 添加影响
        self.interactions[3:5] *= self.decay

        # 3. 计算错误区域
        fn_region = (gt_binary > 0) & (pred_binary == 0)  # FN → fg click
        fp_region = (gt_binary == 0) & (pred_binary > 0)  # FP → bg click

        fn_count = fn_region.sum()
        fp_count = fp_region.sum()

        if fn_count == 0 and fp_count == 0:
            return  # 完美预测，无需 follow-up

        # 按面积比例选 FN 或 FP
        if fn_count == 0:
            error_region, is_fg = fp_region, False
        elif fp_count == 0:
            error_region, is_fg = fn_region, True
        else:
            is_fg = random.random() < fn_count / (fn_count + fp_count)
            error_region = fn_region if is_fg else fp_region

        # 4. 连通域 + EDT 采样
        center = sample_point_from_error_region(error_region, center_biased=True)
        if center is None:
            return

        # 5. 生成 blob 写入通道
        blob = generate_point_blob(self.shape, center, radius=POINT_RADIUS)
        ch = 3 if is_fg else 4
        self.interactions[ch] = np.maximum(self.interactions[ch], blob)

    def get_numpy(self) -> np.ndarray:
        """返回 (7, D, H, W) numpy array。"""
        return self.interactions.copy()
