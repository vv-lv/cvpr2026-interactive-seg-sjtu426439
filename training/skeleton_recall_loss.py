"""
Skeleton Recall Loss for tubular structure segmentation.

Based on MIC-DKFZ's Skeleton Recall Loss (ECCV 2024).
Precompute tubed skeleton from GT on CPU, then use soft recall loss on GPU.

Reference: https://github.com/MIC-DKFZ/Skeleton-Recall
"""
import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import skeletonize, dilation, ball


def precompute_tubed_skeleton_3d(gt_binary: np.ndarray, num_dilations: int = 2) -> np.ndarray:
    """Precompute tubed skeleton from a binary GT mask.

    Args:
        gt_binary: (D, H, W) binary numpy array (uint8 or bool)
        num_dilations: number of dilation iterations (default 2, ~2-voxel tube)

    Returns:
        skel: (D, H, W) float32 array (tubed skeleton, 0 or 1)
    """
    if gt_binary.sum() < 10:
        return np.zeros_like(gt_binary, dtype=np.float32)

    skel = skeletonize(gt_binary.astype(bool))

    struct = ball(1)
    for _ in range(num_dilations):
        skel = dilation(skel, footprint=struct)

    skel = skel & gt_binary.astype(bool)
    return skel.astype(np.float32)


def precompute_multiclass_skeleton(gt: np.ndarray, labels: list,
                                   num_dilations: int = 2,
                                   fill_threshold: float = 0.13,
                                   min_extent: int = 15) -> np.ndarray:
    """Precompute tubed skeleton for multiple classes.

    Only computes skeleton for labels detected as tubular (low fill ratio).
    Non-tubular labels get zero skeleton (skeleton loss is skipped for them).

    Args:
        gt: (D, H, W) multi-class segmentation
        labels: list of foreground label values
        num_dilations: dilation iterations
        fill_threshold: max fill ratio to be considered tubular
        min_extent: min extent in all dims to be considered tubular

    Returns:
        skel: (D, H, W) float32 array with skeleton voxels labeled by class
    """
    skel_map = np.zeros_like(gt, dtype=np.float32)

    for label in labels:
        gt_cls = (gt == label).astype(np.uint8)
        vol = gt_cls.sum()
        if vol < 100:
            continue

        coords = np.argwhere(gt_cls > 0)
        extent = coords.max(0) - coords.min(0) + 1
        fill = vol / (extent[0] * extent[1] * extent[2])
        min_ext = min(extent)

        is_tubular = (fill < fill_threshold) and (min_ext > min_extent)

        if is_tubular:
            skel = precompute_tubed_skeleton_3d(gt_cls, num_dilations)
            skel_map[skel > 0] = label

    return skel_map


class SoftSkeletonRecallLoss(nn.Module):
    """Soft skeleton recall loss.

    Computes recall of the network prediction on the precomputed GT skeleton.
    Only penalizes missed skeleton voxels — forces the model to maintain
    connectivity along centerlines.

    L = -mean( sum(pred_fg * skel_gt) / sum(skel_gt) )  per class
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_softmax: torch.Tensor, skel_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_softmax: (B, C, D, H, W) after softmax, C=2 (bg, fg)
            skel_target: (B, 1, D, H, W) binary tubed skeleton for this object

        Returns:
            Scalar loss (negative recall, to minimize)
        """
        pred_fg = pred_softmax[:, 1:2]

        axes = list(range(2, pred_fg.ndim))
        inter = (pred_fg * skel_target).sum(axes)
        gt_sum = skel_target.sum(axes).clamp(min=self.smooth)

        recall = (inter + self.smooth) / (gt_sum + self.smooth)
        return -recall.mean()
