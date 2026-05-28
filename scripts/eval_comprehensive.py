#!/usr/bin/env python3
"""
Comprehensive evaluation: maxprob assembly + decoupled TTA + v9 attention.

Matches Docker predict.py behavior (maxprob assembly with probability capture)
while supporting all TTA and v9 attention variants.

Variants:
  baseline_lw   — no TTA, no v9, last-wins assembly (reference)
  baseline      — no TTA, no v9, maxprob assembly
  v9            — v9 attention only, maxprob
  tta           — flip TTA + bbox_vol gating, maxprob
  v9_tta        — v9 + TTA, maxprob
  v9_tta_dc     — v9 + decoupled TTA, maxprob (best candidate)
"""
from __future__ import annotations
import os, sys, time, json, argparse, types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import interpolate
from scipy import integrate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "docker"))
EVAL_DIR = PROJECT_ROOT / "evaluation" / "CVPR-MedSegFMCompetition"
sys.path.insert(0, str(EVAL_DIR))

try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

try:
    import scipy.ndimage
    if not hasattr(scipy.ndimage, 'filters'):
        class _F:
            correlate = staticmethod(scipy.ndimage.correlate)
        scipy.ndimage.filters = _F()
    if not hasattr(scipy.ndimage, 'morphology'):
        class _M:
            distance_transform_edt = staticmethod(scipy.ndimage.distance_transform_edt)
        scipy.ndimage.morphology = _M()
except Exception:
    pass
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

from SurfaceDice import compute_dice_coefficient, compute_surface_distances, compute_surface_dice_at_tolerance
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.utils.crop import paste_tensor, crop_and_pad_into_buffer
from nnInteractive.utils.bboxes import generate_bounding_boxes
import cc3d
from scipy.ndimage import distance_transform_edt

from attention_inference import (
    setup_attention, build_token_info_for_object, set_lora_bypass,
)

DEFAULT_CHECKPOINT = Path(
    "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models"
    "/nnInteractive_v1.0_fold_all"
)
V9_CHECKPOINT = PROJECT_ROOT / "experiments" / "v9_no_scale" / "epoch_2.pth"
GT_DIR = Path("/media/ssd/jz/CVPR-BiomedSegFM/3D_val_gt/3D_val_gt_interactive")
N_CLICKS = 5


# ══════════════════════════════════════════════════════════════════════
#  Network TTA Wrapper (with probability capture)
# ══════════════════════════════════════════════════════════════════════

class NetworkTTAWrapper:
    """Wrap network forward with flip TTA. Captures fg probability.

    agree_thresh: if > 0, only use TTA when mean Dice agreement between
        original and flipped predictions exceeds this threshold. When flips
        disagree, TTA is actively moving boundaries → butterfly effect risk.
    """

    def __init__(self, network, flip_axes_list=None, agree_thresh=0.0,
                 orig_weight=0.5):
        self._network = network
        self.flip_axes_list = flip_axes_list or [(0,), (1,), (2,)]
        self.tta_enabled = False
        self.decoupled = False
        self.agree_thresh = agree_thresh
        self.orig_weight = orig_weight  # 0.5 = equal, 0.7 = original-biased
        self._attn_wrappers = self._find_attn_wrappers()

    def _find_attn_wrappers(self):
        wrappers = []
        for module in self._network.modules():
            if hasattr(module, '_token_info') and hasattr(module, '_bypass'):
                wrappers.append(module)
        return wrappers

    def _flip_token_info(self, token_info, flip_axes):
        if not token_info.get('clicks'):
            return token_info
        flipped = {'clicks': []}
        for tok in token_info['clicks']:
            new_tok = dict(tok)
            pos = tok['pos'].clone()
            if len(pos) == 3:  # absolute
                for a in flip_axes:
                    pos[a] = 1.0 - pos[a]
            elif len(pos) >= 4:  # relative
                for a in flip_axes:
                    pos[a] = -pos[a]
            new_tok['pos'] = pos
            flipped['clicks'].append(new_tok)
        return flipped

    @torch.inference_mode()
    def __call__(self, x):
        orig_logits = self._network(x)
        if not self.tta_enabled or not self.flip_axes_list:
            return orig_logits

        orig_f = orig_logits.float()
        flip_results = []
        for flip_axes in self.flip_axes_list:
            saved_infos = []
            for aw in self._attn_wrappers:
                saved_infos.append(aw._token_info)
                aw._token_info = self._flip_token_info(aw._token_info, flip_axes)

            flip_dims = [a + 2 for a in flip_axes]
            fl = self._network(torch.flip(x, dims=flip_dims))
            flip_results.append(torch.flip(fl.float(), dims=flip_dims))
            del fl

            for aw, saved in zip(self._attn_wrappers, saved_infos):
                aw._token_info = saved

        if self.agree_thresh > 0:
            orig_bin = (orig_f[0].argmax(0) > 0)
            agreements = []
            for fl in flip_results:
                fl_bin = (fl[0].argmax(0) > 0)
                inter = (orig_bin & fl_bin).sum().float()
                union = orig_bin.sum() + fl_bin.sum()
                dice = (2 * inter / (union + 1e-8)).item() if union > 0 else 1.0
                agreements.append(dice)
            if sum(agreements) / len(agreements) < self.agree_thresh:
                del flip_results
                return orig_logits

        if self.orig_weight != 0.5 and len(self.flip_axes_list) > 0:
            flip_w = (1.0 - self.orig_weight) / len(self.flip_axes_list)
            result = orig_f * self.orig_weight
            for fl in flip_results:
                result = result + fl * flip_w
            del flip_results
            return result
        else:
            logits_sum = orig_f
            for fl in flip_results:
                logits_sum = logits_sum + fl
            del flip_results
            return logits_sum / (1 + len(self.flip_axes_list))

    def eval(self):
        self._network.eval()
        return self

    def __getattr__(self, name):
        if name in ('_network', 'flip_axes_list', 'tta_enabled', 'decoupled',
                     '_attn_wrappers'):
            raise AttributeError
        return getattr(self._network, name)


# ══════════════════════════════════════════════════════════════════════
#  Monkey-patched _predict with probability capture + decoupled TTA
# ══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def _predict_prob_tta(self):
    """_predict replacement: AutoZoom w/o TTA, final pred with TTA, prob capture.

    Writes to:
      self.target_buffer    — output mask (TTA or baseline)
      self._prob_buffer     — output fg probability
      self._bl_buffer       — baseline mask (decoupled only)
      self._bl_prob_buffer  — baseline fg probability (decoupled only)
    """
    assert self.pad_mode_data == 'constant'
    if len(self.new_interaction_centers) == 0:
        return

    prediction_center = self.new_interaction_centers[-1]
    zoom_out_factor = min(4, self.new_interaction_zoom_out_factors[-1])

    wrapper = self.network
    want_tta = wrapper.tta_enabled
    decoupled = want_tta and wrapper.decoupled

    autoctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
    with autoctx:
        # ── Phase 1: AutoZoom — TTA OFF ──
        wrapper.tta_enabled = False

        input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(
            prediction_center, zoom_out_factor)
        pred = self.network(input_for_predict[None])[0].argmax(0).detach()
        del input_for_predict

        previous_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox)
        if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
            previous_prediction = interpolate(
                previous_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]
        has_change = self._detect_change_at_border(pred, previous_prediction)
        del previous_prediction

        zoom_out_growth_factor = 1.5
        while has_change and self.do_autozoom:
            if zoom_out_factor >= 4:
                break
            zoom_out_factor *= zoom_out_growth_factor
            zoom_out_factor = min(4, zoom_out_factor)

            input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(
                prediction_center, zoom_out_factor)
            pred = self.network(input_for_predict[None])[0].argmax(0).detach()
            del input_for_predict

            previous_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox)
            if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
                previous_prediction = interpolate(
                    previous_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]
            has_change = self._detect_change_at_border(pred, previous_prediction)
        del pred

        # ── Phase 2: Final prediction at settled zoom ──
        input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(
            prediction_center, zoom_out_factor)

        if decoupled:
            # Baseline forward → state + baseline buffers
            wrapper.tta_enabled = False
            logits_bl = self.network(input_for_predict[None])[0].detach()
            pred_bl = logits_bl.argmax(0)
            prob_bl = torch.softmax(logits_bl.float(), dim=0)[1]
            del logits_bl
            # TTA forward → output buffers
            wrapper.tta_enabled = True
            logits_tta = self.network(input_for_predict[None])[0].detach()
            pred_tta = logits_tta.argmax(0)
            prob_tta = torch.softmax(logits_tta.float(), dim=0)[1]
            del logits_tta, input_for_predict
        else:
            wrapper.tta_enabled = want_tta
            logits = self.network(input_for_predict[None])[0].detach()
            pred_bl = logits.argmax(0)
            prob_bl = torch.softmax(logits.float(), dim=0)[1]
            del logits, input_for_predict
            pred_tta = pred_bl
            prob_tta = prob_bl

        if zoom_out_factor == 1:
            # No autozoom — direct paste
            paste_tensor(self.interactions[0], pred_bl.half(), scaled_bbox)
            bbox_orig = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                         zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self.target_buffer, pred_tta.to(self.target_buffer.device), bbox_orig)
            paste_tensor(self._prob_buffer, prob_tta.cpu(), bbox_orig)
            if decoupled:
                paste_tensor(self._bl_buffer, pred_bl.to(self._bl_buffer.device), bbox_orig)
                paste_tensor(self._bl_prob_buffer, prob_bl.cpu(), bbox_orig)
        else:
            # Autozoom — need refinement. Upscale coarse predictions.
            needs_upscale = not all([i == j for i, j in zip(pred_bl.shape, scaled_patch_size)])
            if needs_upscale:
                pred_bl = (interpolate(pred_bl[None, None].to(float),
                    scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                prob_bl = interpolate(prob_bl[None, None],
                    scaled_patch_size, mode='trilinear')[0, 0]
            if decoupled and needs_upscale:
                pred_tta = (interpolate(pred_tta[None, None].to(float),
                    scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                prob_tta = interpolate(prob_tta[None, None],
                    scaled_patch_size, mode='trilinear')[0, 0]
            elif not decoupled:
                # Non-decoupled: tta == bl (keep in sync after upscale)
                pred_tta = pred_bl
                prob_tta = prob_bl

            diff_map, has_diff = self._compute_diff_map(
                pred_bl, self.interactions[0], scaled_bbox, scaled_patch_size)
            paste_tensor(self.interactions[0], pred_bl, scaled_bbox)
            bbox_orig = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                         zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self._prob_buffer, prob_tta.cpu(), bbox_orig)
            if decoupled:
                paste_tensor(self._bl_buffer, pred_bl.to(self._bl_buffer.device), bbox_orig)
                paste_tensor(self._bl_prob_buffer, prob_bl.cpu(), bbox_orig)

            if decoupled:
                _refine_prob_decoupled(self, diff_map, self.interactions[0])
            else:
                _refine_prob(self, diff_map, self.interactions[0])
            del pred_bl, prob_bl, pred_tta, prob_tta

    self.new_interaction_centers = []
    self.new_interaction_zoom_out_factors = []
    empty_cache(self.device)


def _refine_prob(session, diff_map, prediction_with_coarse):
    """Refinement with probability capture (non-decoupled)."""
    if session.has_positive_bbox:
        session.interactions[-6][(~(prediction_with_coarse > 0.5))] = 0
        session.has_positive_bbox = False

    bboxes_ordered = generate_bounding_boxes(
        diff_map, session.configuration_manager.patch_size, stride='auto',
        margin=(10, 10, 10), max_depth=3)
    if len(bboxes_ordered) == 0:
        center = session.new_interaction_centers[-1] if session.new_interaction_centers \
            else [s // 2 for s in session.interactions[0].shape]
        bboxes_ordered = [[[ci - pi // 2, ci - pi // 2 + pi]
                           for ci, pi in zip(center, session.configuration_manager.patch_size)]]
    del diff_map
    empty_cache(session.device)

    preallocated_input = torch.zeros(
        (8, *session.configuration_manager.patch_size),
        device=session.device, dtype=torch.float)

    for refinement_bbox in bboxes_ordered:
        crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox,
                                 session.preprocessed_image[0])
        crop_and_pad_into_buffer(preallocated_input[1], refinement_bbox,
                                 prediction_with_coarse)
        crop_and_pad_into_buffer(preallocated_input[2:], refinement_bbox,
                                 session.interactions[1:])

        logits = session.network(preallocated_input[None])[0].detach()
        pred = logits.argmax(0)
        prob_fg = torch.softmax(logits.float(), dim=0)[1]
        del logits

        paste_tensor(session.interactions[0], pred, refinement_bbox)
        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                zip(refinement_bbox, session.preprocessed_props['bbox_used_for_cropping'])]
        paste_tensor(session.target_buffer, pred.to(session.target_buffer.device), bbox)
        paste_tensor(session._prob_buffer, prob_fg.cpu(), bbox)
        del pred, prob_fg
        preallocated_input.zero_()

    del preallocated_input
    empty_cache(session.device)


def _refine_prob_decoupled(session, diff_map, prediction_with_coarse):
    """Decoupled refinement: baseline → interactions[0] + _bl_buffer,
    TTA → target_buffer + _prob_buffer."""
    if session.has_positive_bbox:
        session.interactions[-6][(~(prediction_with_coarse > 0.5))] = 0
        session.has_positive_bbox = False

    bboxes_ordered = generate_bounding_boxes(
        diff_map, session.configuration_manager.patch_size, stride='auto',
        margin=(10, 10, 10), max_depth=3)
    if len(bboxes_ordered) == 0:
        center = session.new_interaction_centers[-1] if session.new_interaction_centers \
            else [s // 2 for s in session.interactions[0].shape]
        bboxes_ordered = [[[ci - pi // 2, ci - pi // 2 + pi]
                           for ci, pi in zip(center, session.configuration_manager.patch_size)]]
    del diff_map
    empty_cache(session.device)

    wrapper = session.network
    preallocated_input = torch.zeros(
        (8, *session.configuration_manager.patch_size),
        device=session.device, dtype=torch.float)

    for refinement_bbox in bboxes_ordered:
        crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox,
                                 session.preprocessed_image[0])
        crop_and_pad_into_buffer(preallocated_input[1], refinement_bbox,
                                 prediction_with_coarse)
        crop_and_pad_into_buffer(preallocated_input[2:], refinement_bbox,
                                 session.interactions[1:])

        # Baseline → state
        wrapper.tta_enabled = False
        logits_bl = wrapper(preallocated_input[None])[0].detach()
        pred_bl = logits_bl.argmax(0)
        prob_bl = torch.softmax(logits_bl.float(), dim=0)[1]
        del logits_bl
        paste_tensor(session.interactions[0], pred_bl, refinement_bbox)
        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                zip(refinement_bbox, session.preprocessed_props['bbox_used_for_cropping'])]
        paste_tensor(session._bl_buffer, pred_bl.to(session._bl_buffer.device), bbox)
        paste_tensor(session._bl_prob_buffer, prob_bl.cpu(), bbox)

        # TTA → output
        wrapper.tta_enabled = True
        logits_tta = wrapper(preallocated_input[None])[0].detach()
        pred_tta = logits_tta.argmax(0)
        prob_tta = torch.softmax(logits_tta.float(), dim=0)[1]
        del logits_tta
        paste_tensor(session.target_buffer, pred_tta.to(session.target_buffer.device), bbox)
        paste_tensor(session._prob_buffer, prob_tta.cpu(), bbox)

        del pred_bl, prob_bl, pred_tta, prob_tta
        preallocated_input.zero_()

    del preallocated_input
    empty_cache(session.device)


# ══════════════════════════════════════════════════════════════════════
#  Maxprob assembly
# ══════════════════════════════════════════════════════════════════════

def maxprob_assemble(per_obj_masks, per_obj_probs, classes, alpha=0.5):
    """Maxprob assembly with optional size-boost/Bayesian and safety net.

    alpha=0   → raw maxprob (original)
    alpha>0   → size-boosted maxprob (small objects get probability boost)
    alpha=-1  → Bayesian: score = p_k(x) / |M_k| (zero-parameter)
    """
    MIN_VOXELS = 5
    if not classes:
        return np.zeros(1, dtype=np.uint8)
    shape = per_obj_masks[classes[0]].shape
    result = np.zeros(shape, dtype=np.uint8)
    for cls in classes:
        result[per_obj_masks[cls] > 0] = cls

    stacked = sum(per_obj_masks[c].astype(np.int32) for c in classes)
    overlap_mask = stacked > 1
    if overlap_mask.any():
        pred_sizes = {c: max(1, int(per_obj_masks[c].sum())) for c in classes}
        if alpha == -1:
            # Bayesian: score = p / V_hard
            idx = np.argwhere(overlap_mask)
            for voxel in idx:
                v = tuple(voxel)
                claiming = [c for c in classes if per_obj_masks[c][v] > 0]
                best = max(claiming,
                           key=lambda c: per_obj_probs[c][v] / pred_sizes[c])
                result[v] = best
        else:
            boost = {c: 1.0 + alpha / np.log2(pred_sizes[c] + 2) for c in classes}
            idx = np.argwhere(overlap_mask)
            for voxel in idx:
                v = tuple(voxel)
                claiming = [c for c in classes if per_obj_masks[c][v] > 0]
                best = max(claiming,
                           key=lambda c: per_obj_probs[c][v] * boost[c])
                result[v] = best

    for cls in classes:
        if (result == cls).sum() < MIN_VOXELS and per_obj_masks[cls].any():
            raw_probs = per_obj_probs[cls].copy()
            raw_probs[per_obj_masks[cls] == 0] = -1
            flat_idx = np.argsort(raw_probs.ravel())[::-1][:MIN_VOXELS]
            coords = np.unravel_index(flat_idx, shape)
            for i in range(len(coords[0])):
                result[coords[0][i], coords[1][i], coords[2][i]] = cls

    return result


# ══════════════════════════════════════════════════════════════════════
#  Eval infrastructure
# ══════════════════════════════════════════════════════════════════════

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]:
        dsc.append(compute_dice_coefficient(gt == i, seg == i))
    return np.mean(dsc) if dsc else 0.0


from SurfaceDice import neighbour_code_to_normals as _NC2N
_NC2N_ARR = [np.array(n) for n in _NC2N]


def _build_area_lut(spacing_mm):
    lut = np.zeros(256)
    s0, s1, s2 = spacing_mm
    for code in range(256):
        normals = _NC2N_ARR[code]
        s = 0.0
        for i in range(normals.shape[0]):
            n = np.array([normals[i, 0] * s1 * s2,
                          normals[i, 1] * s0 * s2,
                          normals[i, 2] * s0 * s1])
            s += np.linalg.norm(n)
        lut[code] = s
    return lut


def _fast_nsd_one_class(mask_gt, mask_pred, spacing_mm, tolerance):
    """Fast NSD for one class — skips sorted() since NSD only needs threshold check."""
    mask_all = mask_gt | mask_pred
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx0 = np.nonzero(proj_0)[0]
    if len(idx0) == 0:
        return 0.0
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx1 = np.nonzero(proj_1)[0]
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx2 = np.nonzero(proj_2)[0]
    crop_gt = np.zeros((idx0.max()-idx0.min()+2, idx1.max()-idx1.min()+2, idx2.max()-idx2.min()+2), np.uint8)
    crop_pred = np.zeros_like(crop_gt)
    cs = (slice(0, idx0.max()-idx0.min()+1), slice(0, idx1.max()-idx1.min()+1), slice(0, idx2.max()-idx2.min()+1))
    crop_gt[cs] = mask_gt[idx0.min():idx0.max()+1, idx1.min():idx1.max()+1, idx2.min():idx2.max()+1]
    crop_pred[cs] = mask_pred[idx0.min():idx0.max()+1, idx1.min():idx1.max()+1, idx2.min():idx2.max()+1]
    kernel = np.array([[[128,64],[32,16]],[[8,4],[2,1]]])
    nc_gt = scipy.ndimage.correlate(crop_gt, kernel, mode="constant", cval=0)
    nc_pred = scipy.ndimage.correlate(crop_pred, kernel, mode="constant", cval=0)
    borders_gt = (nc_gt != 0) & (nc_gt != 255)
    borders_pred = (nc_pred != 0) & (nc_pred != 255)
    if not borders_gt.any() or not borders_pred.any():
        return 0.0
    area_lut = _build_area_lut(spacing_mm)
    area_gt = area_lut[nc_gt]
    area_pred = area_lut[nc_pred]
    dt_gt = distance_transform_edt(~borders_gt, sampling=spacing_mm)
    dt_pred = distance_transform_edt(~borders_pred, sampling=spacing_mm)
    overlap_gt = area_gt[borders_gt & (dt_pred <= tolerance)].sum()
    overlap_pred = area_pred[borders_pred & (dt_gt <= tolerance)].sum()
    total = area_gt[borders_gt].sum() + area_pred[borders_pred].sum()
    return (overlap_gt + overlap_pred) / total if total > 0 else 0.0


def compute_multi_class_nsd(gt, seg, spacing_mm, tolerance=2.0):
    nsd = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]:
        gt_i = gt == i
        seg_i = seg == i
        if not seg_i.any():
            nsd.append(0.0)
            continue
        nsd.append(_fast_nsd_one_class(gt_i, seg_i, spacing_mm, tolerance))
    return np.mean(nsd) if nsd else 0.0


def compute_edt(error_component):
    """Copied from official CVPR25_iter_eval.py — identical behavior."""
    coords = np.argwhere(error_component)
    if len(coords) == 0:
        return np.zeros_like(error_component, dtype=np.float64)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    crop_shape = max_coords - min_coords
    padding = np.maximum((crop_shape * 0.25).astype(int), 1)
    padded_shape = crop_shape + 2 * padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)
    center_crop[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ] = error_component[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]
    large_roi = False
    if center_crop.shape[0] * center_crop.shape[1] * center_crop.shape[2] > 60000000:
        from skimage.measure import block_reduce
        center_crop = block_reduce(center_crop, block_size=(2, 2, 2), func=np.max)
        large_roi = True
    # Official eval uses cupy/cucim GPU EDT when available; we use scipy CPU EDT.
    # Both compute exact EDT; results differ only at floating-point tie-breaking.
    edt = distance_transform_edt(center_crop)
    if large_roi:
        edt = edt.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    dist_cropped = edt[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ]
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ] = dist_cropped
    return dist_full


def sample_coord(edt):
    """Copied from official CVPR25_iter_eval.py — identical behavior."""
    np.random.seed(42)
    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)
    chosen_index = max_coords[np.random.choice(len(max_coords))]
    center = tuple(chosen_index)
    return center


def generate_click_official(segs, gts, unique_classes, clicks_cls, clicks_order):
    for ind, cls in enumerate(unique_classes):
        segs_cls = (segs == cls).astype(np.uint8)
        gts_cls = (gts == cls).astype(np.uint8)
        error_mask = (segs_cls != gts_cls).astype(np.uint8)
        if np.sum(error_mask) > 0:
            errors = cc3d.connected_components(error_mask, connectivity=26)
            component_sizes = np.bincount(errors.flat)
            component_sizes[0] = 0
            largest_component = (errors == np.argmax(component_sizes))
            edt = compute_edt(largest_component)
            edt *= largest_component
            if np.sum(edt) == 0:
                edt = largest_component.astype(np.float64)
            center = sample_coord(edt)
            if gts_cls[center] == 0:
                clicks_cls[ind]['bg'].append(list(center))
                clicks_order[ind].append('bg')
            else:
                clicks_cls[ind]['fg'].append(list(center))
                clicks_order[ind].append('fg')
        else:
            clicks_order[ind].append(None)


def get_object_bbox_vol(boxes, oid_idx, prev_pred, cls):
    if boxes is not None and oid_idx < len(boxes):
        bb = boxes[oid_idx]
        dz = int(bb['z_max']) - int(bb['z_min']) + 1
        dy = int(bb['z_mid_y_max']) - int(bb['z_mid_y_min']) + 1
        dx = int(bb['z_mid_x_max']) - int(bb['z_mid_x_min']) + 1
        return dz * dy * dx
    if prev_pred is not None:
        vol = int((prev_pred == cls).sum())
        if vol > 0:
            return vol
    return 999999


# ══════════════════════════════════════════════════════════════════════
#  Core: run one case
# ══════════════════════════════════════════════════════════════════════

def run_one_case(npz_path, gt_path, device, checkpoint_dir, max_obj=8,
                 use_v9=False, attn_ckpt_path=None,
                 flip_axes_list=None, bbox_vol_thresh=0,
                 decoupled=False, use_maxprob=True,
                 max_tta_labels=0, agree_thresh=0.0,
                 orig_weight=0.5, tta_rounds=None,
                 assembly_alpha=0.0, compute_nsd=False):
    """Run one case with all features.

    Args:
        use_maxprob: True → maxprob assembly, False → last-wins (for baseline_lw reference)
        decoupled: True → non-TTA for prev_pred, TTA for output
        max_tta_labels: if > 0, disable TTA for cases with more labels (case-level gating)
        agree_thresh: if > 0, only use TTA when flip agreement > threshold (per-forward gating)
    """
    data = np.load(npz_path, allow_pickle=True)
    image = data['imgs']
    gt = np.load(gt_path, allow_pickle=True)['gts']
    boxes = data['boxes'] if 'boxes' in data else None
    spacing = data['spacing'] if 'spacing' in data else None

    unique_labels = np.sort(pd.unique(gt.ravel()))
    gt_classes = sorted([int(l) for l in unique_labels if l > 0])
    if max_obj > 0 and len(gt_classes) > max_obj:
        gt_classes = gt_classes[:max_obj]
        gt = gt * np.isin(gt, gt_classes).astype(gt.dtype)
    if len(gt_classes) == 0:
        return None

    # ── Session setup helper (called each round, matching Docker lifecycle) ──
    use_tta = flip_axes_list is not None
    if use_tta and max_tta_labels > 0 and len(gt_classes) > max_tta_labels:
        use_tta = False

    sp = list(spacing) if spacing is not None else [1., 1., 1.]
    spacing_dhw = [sp[2], sp[1], sp[0]] if len(sp) == 3 else [1., 1., 1.]
    image_f32 = image[None].astype(np.float32)

    has_bbox = boxes is not None
    clicks_cls = [{'fg': [], 'bg': []} for _ in gt_classes]
    clicks_order = [[] for _ in gt_classes]
    prev_pred = None
    dscs = []
    nsds = []

    for it in range(N_CLICKS + 1):
        if it == 0:
            if not has_bbox:
                dscs.append(0)
                if compute_nsd:
                    nsds.append(0)
                continue
        else:
            if prev_pred is None:
                prev_pred = np.zeros_like(gt, dtype=np.uint8)
            generate_click_official(prev_pred, gt, gt_classes,
                                    clicks_cls, clicks_order)

        # ── Fresh session each round (matching Docker: each call is independent) ──
        session = nnInteractiveInferenceSession(
            device=device, use_torch_compile=False, verbose=False,
            torch_n_threads=os.cpu_count(),
            do_autozoom=True, use_pinned_memory=True,
        )
        session.initialize_from_trained_model_folder(
            model_training_output_dir=str(checkpoint_dir), use_fold='all')

        # v9 attention
        attn_wrapper = None
        use_relative_pos_local = False
        if use_v9 and attn_ckpt_path and os.path.exists(str(attn_ckpt_path)):
            attn_wrapper, use_relative_pos_local = setup_attention(
                session, str(attn_ckpt_path), device)

        # TTA wrapper (with round-level gating)
        round_tta = use_tta and (tta_rounds is None or it in tta_rounds)
        tta_wrapper = None
        if round_tta:
            tta_wrapper = NetworkTTAWrapper(session.network, flip_axes_list,
                                            agree_thresh=agree_thresh,
                                            orig_weight=orig_weight)
            tta_wrapper.decoupled = decoupled
            session.network = tta_wrapper
            session._predict = types.MethodType(_predict_prob_tta, session)
        elif use_maxprob:
            tta_wrapper = NetworkTTAWrapper(session.network, [])
            tta_wrapper.decoupled = False
            session.network = tta_wrapper
            session._predict = types.MethodType(_predict_prob_tta, session)

        session.set_image(image_f32)
        target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
        session.set_target_buffer(target_buffer)
        session._prob_buffer = torch.zeros(image.shape, dtype=torch.float32, device='cpu')
        session._bl_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
        session._bl_prob_buffer = torch.zeros(image.shape, dtype=torch.float32, device='cpu')

        # Bbox cases → bypass attention for ALL rounds
        if attn_wrapper is not None and has_bbox:
            attn_wrapper._bypass = True
            set_lora_bypass(session.network, True)

        per_obj_masks = {}
        per_obj_probs = {}
        per_obj_bl_masks = {}
        per_obj_bl_probs = {}

        for oid_idx, cls in enumerate(gt_classes):
            target_buffer.zero_()
            session._prob_buffer.zero_()
            session._bl_buffer.zero_()
            session._bl_prob_buffer.zero_()

            # Per-object TTA gating
            if tta_wrapper is not None:
                if bbox_vol_thresh > 0:
                    vol = get_object_bbox_vol(boxes, oid_idx, prev_pred, cls)
                    tta_wrapper.tta_enabled = use_tta and (vol >= bbox_vol_thresh)
                else:
                    tta_wrapper.tta_enabled = use_tta

            # Add prev_pred or reset
            if prev_pred is not None:
                session.add_initial_seg_interaction(
                    (prev_pred == cls).astype(np.uint8), run_prediction=False)
            else:
                session.reset_interactions()

            # Add bbox (every round, matching Docker — bbox is cumulative)
            if has_bbox and oid_idx < len(boxes):
                bb = boxes[oid_idx]
                session.add_bbox_interaction(
                    [[int(bb['z_min']), int(bb['z_max']) + 1],
                     [int(bb['z_mid_y_min']), int(bb['z_mid_y_max']) + 1],
                     [int(bb['z_mid_x_min']), int(bb['z_mid_x_max']) + 1]],
                    include_interaction=True, run_prediction=False)

            # Add clicks
            if clicks_cls[oid_idx]['fg'] or clicks_cls[oid_idx]['bg']:
                fg_ptr = bg_ptr = 0
                for kind in clicks_order[oid_idx]:
                    if kind is None:
                        continue
                    if kind == 'fg':
                        click = clicks_cls[oid_idx]['fg'][fg_ptr]; fg_ptr += 1
                    else:
                        click = clicks_cls[oid_idx]['bg'][bg_ptr]; bg_ptr += 1
                    session.add_point_interaction(
                        click, include_interaction=(kind == 'fg'),
                        run_prediction=False)

            if not session.new_interaction_centers:
                per_obj_masks[cls] = np.zeros(image.shape, dtype=np.uint8)
                per_obj_probs[cls] = np.zeros(image.shape, dtype=np.float32)
                if decoupled:
                    per_obj_bl_masks[cls] = np.zeros(image.shape, dtype=np.uint8)
                    per_obj_bl_probs[cls] = np.zeros(image.shape, dtype=np.float32)
                continue

            # Set attention token info (no-bbox cases only, matching Docker)
            if attn_wrapper is not None and not has_bbox:
                build_token_info_for_object(
                    session, oid_idx + 1, len(gt_classes),
                    clicks_cls, clicks_order,
                    attn_wrapper, use_relative_pos=use_relative_pos_local,
                    spacing_dhw=spacing_dhw)

            session.new_interaction_centers = [session.new_interaction_centers[-1]]
            session.new_interaction_zoom_out_factors = [
                session.new_interaction_zoom_out_factors[-1]]
            session._predict()

            per_obj_masks[cls] = (target_buffer.numpy() > 0).astype(np.uint8).copy()
            per_obj_probs[cls] = session._prob_buffer.numpy().copy()
            if decoupled:
                if tta_wrapper is not None and tta_wrapper.tta_enabled:
                    per_obj_bl_masks[cls] = (session._bl_buffer.numpy() > 0).astype(np.uint8).copy()
                    per_obj_bl_probs[cls] = session._bl_prob_buffer.numpy().copy()
                else:
                    per_obj_bl_masks[cls] = per_obj_masks[cls].copy()
                    per_obj_bl_probs[cls] = per_obj_probs[cls].copy()

        del session
        empty_cache(torch.device(device))

        # ── Assemble ──
        if use_maxprob and len(gt_classes) > 1:
            result = maxprob_assemble(per_obj_masks, per_obj_probs, gt_classes, alpha=assembly_alpha)
        elif use_maxprob:
            cls0 = gt_classes[0]
            result = np.zeros(image.shape, dtype=np.uint8)
            result[per_obj_masks.get(cls0, np.zeros(image.shape, dtype=np.uint8)) > 0] = cls0
        else:
            result = np.zeros(image.shape, dtype=np.uint8)
            for cls in gt_classes:
                if cls in per_obj_masks:
                    result[per_obj_masks[cls] > 0] = cls

        # ── Prev_pred for next round ──
        if decoupled and per_obj_bl_masks:
            if len(gt_classes) > 1:
                prev_pred = maxprob_assemble(per_obj_bl_masks, per_obj_bl_probs, gt_classes, alpha=assembly_alpha)
            else:
                cls0 = gt_classes[0]
                prev_pred = np.zeros(image.shape, dtype=np.uint8)
                if cls0 in per_obj_bl_masks:
                    prev_pred[per_obj_bl_masks[cls0] > 0] = cls0
        else:
            prev_pred = result.copy()

        dsc = compute_multi_class_dsc(gt, result)
        dscs.append(dsc)

        if compute_nsd:
            if dsc > 0.2:
                nsd = compute_multi_class_nsd(gt, result, spacing_dhw)
            else:
                nsd = 0.0
            nsds.append(nsd)

    click_dscs = np.array(dscs[1:])
    auc = integrate.cumulative_trapezoid(
        click_dscs, np.arange(len(click_dscs)))[-1] if len(click_dscs) >= 2 else 0
    ret = {'DSC_AUC': auc, 'DSC_Final': dscs[-1], 'DSC_bbox': dscs[0],
           'per_round': dscs}
    if compute_nsd:
        click_nsds = np.array(nsds[1:])
        nsd_auc = integrate.cumulative_trapezoid(
            click_nsds, np.arange(len(click_nsds)))[-1] if len(click_nsds) >= 2 else 0
        ret.update({'NSD_AUC': nsd_auc, 'NSD_Final': nsds[-1],
                    'per_round_nsd': nsds})
    return ret


# ══════════════════════════════════════════════════════════════════════
#  Variant runner
# ══════════════════════════════════════════════════════════════════════

VARIANT_CONFIGS = {
    'baseline_lw': dict(use_v9=False, flip_axes_list=None, bbox_vol_thresh=0,
                        decoupled=False, use_maxprob=False),
    'baseline_mp': dict(use_v9=False, flip_axes_list=None, bbox_vol_thresh=0,
                        decoupled=False, use_maxprob=True, assembly_alpha=0.0),
    'baseline': dict(use_v9=False, flip_axes_list=None, bbox_vol_thresh=0,
                     decoupled=False, use_maxprob=True),
    'baseline_bayesian': dict(use_v9=False, flip_axes_list=None, bbox_vol_thresh=0,
                              decoupled=False, use_maxprob=True, assembly_alpha=-1.0),
    'v9_lw': dict(use_v9=True, flip_axes_list=None, bbox_vol_thresh=0,
                  decoupled=False, use_maxprob=False),
    'v9_mp': dict(use_v9=True, flip_axes_list=None, bbox_vol_thresh=0,
                  decoupled=False, use_maxprob=True, assembly_alpha=0.0),
    'v9': dict(use_v9=True, flip_axes_list=None, bbox_vol_thresh=0,
               decoupled=False, use_maxprob=True),
    'v9_bayesian': dict(use_v9=True, flip_axes_list=None, bbox_vol_thresh=0,
                        decoupled=False, use_maxprob=True, assembly_alpha=-1.0),
    'tta': dict(use_v9=False, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                decoupled=False, use_maxprob=True),
    'v9_tta': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                   decoupled=False, use_maxprob=True),
    'v9_tta_dc': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                      decoupled=True, use_maxprob=True),
    # Agreement-gated TTA variants
    'v9_tta_ag90': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                        decoupled=False, use_maxprob=True, agree_thresh=0.90),
    'v9_tta_ag95': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                        decoupled=False, use_maxprob=True, agree_thresh=0.95),
    'v9_tta_ag98': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                        decoupled=False, use_maxprob=True, agree_thresh=0.98),
    # Label-count gated TTA
    'v9_tta_l5': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                      decoupled=False, use_maxprob=True, max_tta_labels=5),
    # Combined: label gating + agreement gating
    'v9_tta_l5_ag95': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                           decoupled=False, use_maxprob=True,
                           max_tta_labels=5, agree_thresh=0.95),
    # Round-specific: TTA only at R5 (no cascade possible)
    'v9_tta_r5': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                      decoupled=False, use_maxprob=True, tta_rounds={5}),
    # Round-specific + agreement gating
    'v9_tta_r5_ag90': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                           decoupled=False, use_maxprob=True,
                           tta_rounds={5}, agree_thresh=0.90, assembly_alpha=0.0),
    # Original-biased weighting (70/30)
    'v9_tta_w70': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                       decoupled=False, use_maxprob=True, orig_weight=0.7),
    # Original-biased + agreement gating
    'v9_tta_w70_ag90': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                            decoupled=False, use_maxprob=True,
                            orig_weight=0.7, agree_thresh=0.90),
    # R4-R5 only (late rounds, less cascade risk)
    'v9_tta_r45': dict(use_v9=True, flip_axes_list=[(2,)], bbox_vol_thresh=5000,
                       decoupled=False, use_maxprob=True, tta_rounds={4, 5}),
}


def parse_resume_log(log_path, variant_name):
    """Parse completed cases from a previous log file for resuming."""
    done = {}
    if not log_path or not os.path.exists(log_path):
        return done
    in_variant = False
    with open(log_path) as f:
        for line in f:
            if line.strip().startswith('VARIANT:'):
                in_variant = (variant_name in line)
            if in_variant and 'AUC=' in line and '] ' in line:
                fname = line.split('] ')[1].split(' (')[0].strip()
                try:
                    auc = float(line.split('AUC=')[1].split()[0])
                    final = float(line.split('Final=')[1].split()[0])
                    t = float(line.rsplit('(', 1)[1].split('s)')[0])
                    done[fname] = {'DSC_AUC': auc, 'DSC_Final': final, 'time_s': t}
                except (ValueError, IndexError):
                    pass
    return done


def run_variant(variant_name, val_files, device, max_obj, attn_ckpt_path,
                resume_log=None, compute_nsd=False):
    cfg = VARIANT_CONFIGS[variant_name]
    resumed = parse_resume_log(resume_log, variant_name)
    print(f"\n{'='*70}", flush=True)
    print(f"VARIANT: {variant_name}", flush=True)
    print(f"  v9={cfg['use_v9']}, TTA={cfg['flip_axes_list']}, "
          f"decoupled={cfg['decoupled']}, maxprob={cfg['use_maxprob']}, "
          f"bbox_vol_thresh={cfg['bbox_vol_thresh']}, "
          f"max_tta_labels={cfg.get('max_tta_labels',0)}, "
          f"agree_thresh={cfg.get('agree_thresh',0)}", flush=True)
    print(f"  {len(val_files)} cases ({len(resumed)} resumed)", flush=True)
    print(f"{'='*70}", flush=True)

    rows = []
    for i, vf in enumerate(val_files):
        if vf['fname'] in resumed:
            r = resumed[vf['fname']]
            print(f"  [{i+1}/{len(val_files)}] {vf['fname'][:50]}: RESUMED "
                  f"AUC={r['DSC_AUC']:.4f} Final={r['DSC_Final']:.4f}", flush=True)
            rows.append({
                'case': vf['fname'], 'dataset': vf['dataset'],
                'has_bbox': vf.get('has_bbox', True),
                'DSC_AUC': r['DSC_AUC'], 'DSC_Final': r['DSC_Final'],
                'DSC_bbox': 0, 'time_s': r['time_s'],
                'variant': variant_name,
            })
            continue
        t0 = time.time()
        try:
            r = run_one_case(
                vf['npz'], vf['gt'], device, DEFAULT_CHECKPOINT,
                max_obj=max_obj,
                use_v9=cfg['use_v9'],
                attn_ckpt_path=attn_ckpt_path,
                flip_axes_list=cfg['flip_axes_list'],
                bbox_vol_thresh=cfg['bbox_vol_thresh'],
                decoupled=cfg['decoupled'],
                use_maxprob=cfg['use_maxprob'],
                max_tta_labels=cfg.get('max_tta_labels', 0),
                agree_thresh=cfg.get('agree_thresh', 0.0),
                orig_weight=cfg.get('orig_weight', 0.5),
                tta_rounds=cfg.get('tta_rounds', None),
                assembly_alpha=cfg.get('assembly_alpha', 0.5),
                compute_nsd=compute_nsd)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i+1}] {vf['fname'][:50]}: OOM", flush=True)
            empty_cache(torch.device(device))
            continue
        except Exception as e:
            import traceback
            print(f"  [{i+1}] {vf['fname'][:50]}: ERROR {e}", flush=True)
            traceback.print_exc()
            empty_cache(torch.device(device))
            continue

        elapsed = time.time() - t0
        if r is None:
            continue

        rounds_str = ' '.join(f'{d:.3f}' for d in r['per_round'])
        bbox_tag = 'bbox' if vf.get('has_bbox', True) else 'nobbox'
        nsd_str = ''
        if compute_nsd and 'NSD_AUC' in r:
            nsd_str = f" NSD_AUC={r['NSD_AUC']:.4f} NSD_F={r['NSD_Final']:.4f}"
        print(f"  [{i+1}/{len(val_files)}] {vf['fname'][:50]} ({bbox_tag}): "
              f"AUC={r['DSC_AUC']:.4f} Final={r['DSC_Final']:.4f}{nsd_str} ({elapsed:.0f}s)",
              flush=True)
        print(f"    rounds=[{rounds_str}]", flush=True)

        row = {
            'case': vf['fname'], 'dataset': vf['dataset'],
            'has_bbox': vf.get('has_bbox', True),
            'DSC_AUC': r['DSC_AUC'], 'DSC_Final': r['DSC_Final'],
            'DSC_bbox': r['DSC_bbox'], 'time_s': elapsed,
            'variant': variant_name,
        }
        if compute_nsd and 'NSD_AUC' in r:
            row['NSD_AUC'] = r['NSD_AUC']
            row['NSD_Final'] = r['NSD_Final']
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n  --- {variant_name} Summary ({len(df)} cases) ---", flush=True)
        nsd_summary = ''
        if 'NSD_AUC' in df.columns:
            nsd_summary = (f"  NSD_AUC={df.NSD_AUC.mean():.4f}  "
                           f"NSD_Final={df.NSD_Final.mean():.4f}")
        print(f"  mean AUC={df.DSC_AUC.mean():.4f}  "
              f"Final={df.DSC_Final.mean():.4f}  "
              f"time={df.time_s.mean():.1f}s/case", flush=True)
        if nsd_summary:
            print(f"  {nsd_summary}", flush=True)
        for has_bb in [True, False]:
            sub = df[df.has_bbox == has_bb]
            if len(sub) > 0:
                tag = 'bbox' if has_bb else 'no-bbox'
                print(f"    {tag:10s} n={len(sub):3d}  "
                      f"AUC={sub.DSC_AUC.mean():.4f}  "
                      f"Final={sub.DSC_Final.mean():.4f}", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive eval: maxprob + decoupled TTA + v9 attention")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_obj', type=int, default=0,
                        help='Max objects per case (0=no limit, matching official eval)')
    parser.add_argument('--val_json', required=True)
    parser.add_argument('--n_cases', type=int, default=0)
    parser.add_argument('--variants', default='baseline_lw,baseline,v9_tta_dc',
                        help='Comma-separated variant names or "all"')
    parser.add_argument('--v9_ckpt', default=str(V9_CHECKPOINT))
    parser.add_argument('--resume_log', default=None,
                        help='Path to previous log file to resume from')
    parser.add_argument('--compute_nsd', action='store_true',
                        help='Also compute NSD (slower due to surface distance)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    with open(args.val_json) as f:
        val_data = json.load(f)
    val_files = []
    for entry in val_data['files']:
        npz_path = entry['path']
        fname = os.path.basename(npz_path)
        gt_path = entry.get('gt_path', str(GT_DIR / fname))
        if os.path.exists(npz_path) and os.path.exists(gt_path):
            val_files.append({
                'npz': npz_path, 'gt': gt_path, 'fname': fname,
                'dataset': entry.get('dataset', ''),
                'has_bbox': entry.get('has_bbox', True),
            })
    if args.n_cases > 0:
        val_files = val_files[:args.n_cases]

    print(f"Loaded {len(val_files)} cases from {args.val_json}", flush=True)
    print(f"v9 checkpoint: {args.v9_ckpt}", flush=True)

    if args.variants == 'all':
        variants = list(VARIANT_CONFIGS.keys())
    else:
        variants = [v.strip() for v in args.variants.split(',')]
        for v in variants:
            if v not in VARIANT_CONFIGS:
                sys.exit(f"Unknown variant: {v}. Choose from: {list(VARIANT_CONFIGS.keys())}")

    all_rows = []
    for variant in variants:
        rows = run_variant(variant, val_files, device, args.max_obj, args.v9_ckpt,
                           resume_log=args.resume_log,
                           compute_nsd=args.compute_nsd)
        all_rows.extend(rows)

    # ── Final comparison ──
    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f"\n{'='*70}", flush=True)
        print("FINAL COMPARISON", flush=True)
        print(f"{'='*70}", flush=True)

        header = f"{'variant':20s} {'n':>3s} {'AUC':>7s} {'Final':>7s} {'time':>6s}"
        print(header, flush=True)
        print('-' * len(header), flush=True)
        for variant in variants:
            sub = df[df.variant == variant]
            if len(sub) == 0:
                continue
            print(f"{variant:20s} {len(sub):3d} "
                  f"{sub.DSC_AUC.mean():7.4f} {sub.DSC_Final.mean():7.4f} "
                  f"{sub.time_s.mean():5.1f}s", flush=True)

        # Per-case delta vs baseline_lw (if available)
        ref = 'baseline_lw' if 'baseline_lw' in variants else variants[0]
        ref_df = df[df.variant == ref].set_index('case')
        for variant in variants:
            if variant == ref:
                continue
            var_df = df[df.variant == variant].set_index('case')
            common = ref_df.index.intersection(var_df.index)
            if len(common) == 0:
                continue
            deltas = var_df.loc[common, 'DSC_AUC'] - ref_df.loc[common, 'DSC_AUC']
            n_up = (deltas > 0.001).sum()
            n_down = (deltas < -0.001).sum()
            print(f"\n  {variant} vs {ref}: mean dAUC={deltas.mean():+.4f}, "
                  f"{n_up} improved, {n_down} regressed", flush=True)

        out_csv = PROJECT_ROOT / "experiments" / "comprehensive_eval.csv"
        os.makedirs(out_csv.parent, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}", flush=True)


if __name__ == '__main__':
    main()
