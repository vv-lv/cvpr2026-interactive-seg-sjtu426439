#!/usr/bin/env python3
"""
Speed-optimized predict.py for CVPR 2026 Interactive Track.

Optimizations:
  1. JIT fast loading: skip network build + pickle, use torch.jit.load (~2.3s/call saved)
  2. Manual session init: bypass initialize_from_trained_model_folder
  3. Mock trainer import: skip 816ms of unnecessary batchgeneratorsv2 imports
  4. CUDA Graphs: cache compiled forward graph, ~6% forward speedup
  5. Same output as baseline (last-wins assembly)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.interaction.point import PointInteraction_stub
from nnunetv2.utilities.helpers import empty_cache
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from torch.nn.functional import interpolate
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.utils.crop import paste_tensor, crop_and_pad_into_buffer
from nnInteractive.utils.bboxes import generate_bounding_boxes

from attention_inference import (
    setup_attention, build_token_info_for_object, set_lora_bypass,
)

CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', 'checkpoint_folder')
ATTN_CKPT_PATH = os.environ.get('ATTN_CKPT', 'attention_checkpoint.pth')
TTA_AGREE_THRESH = 0.90
TTA_FLIP_AXES = [(2,)]
N_CLICK_ROUNDS = 5


class NetworkTTAWrapper:
    """Flip TTA with agreement gating for R5-only use."""

    def __init__(self, network, flip_axes_list, agree_thresh=0.90):
        self._network = network
        self.flip_axes_list = flip_axes_list
        self.agree_thresh = agree_thresh
        self.tta_enabled = False
        self._attn_wrappers = [
            m for m in network.modules()
            if hasattr(m, '_token_info') and hasattr(m, '_bypass')
        ]

    def _flip_token_info(self, token_info, flip_axes):
        if not token_info.get('clicks'):
            return token_info
        flipped = {'clicks': []}
        for tok in token_info['clicks']:
            new_tok = dict(tok)
            pos = tok['pos'].clone()
            if len(pos) == 3:
                for a in flip_axes:
                    pos[a] = 1.0 - pos[a]
            elif len(pos) >= 4:
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
            saved = [aw._token_info for aw in self._attn_wrappers]
            for aw in self._attn_wrappers:
                aw._token_info = self._flip_token_info(aw._token_info, flip_axes)
            flip_dims = [a + 2 for a in flip_axes]
            fl = self._network(torch.flip(x, dims=flip_dims))
            flip_results.append(torch.flip(fl.float(), dims=flip_dims))
            del fl
            for aw, s in zip(self._attn_wrappers, saved):
                aw._token_info = s
        if self.agree_thresh > 0:
            orig_bin = (orig_f[0].argmax(0) > 0)
            for fl in flip_results:
                fl_bin = (fl[0].argmax(0) > 0)
                inter = (orig_bin & fl_bin).sum().float()
                union = orig_bin.sum() + fl_bin.sum()
                dice = (2 * inter / (union + 1e-8)).item() if union > 0 else 1.0
                if dice < self.agree_thresh:
                    del flip_results
                    return orig_logits
        logits_sum = orig_f
        for fl in flip_results:
            logits_sum = logits_sum + fl
        del flip_results
        return logits_sum / (1 + len(self.flip_axes_list))

    def eval(self):
        self._network.eval()
        return self

    def __getattr__(self, name):
        if name in ('_network', 'flip_axes_list', 'tta_enabled',
                     '_attn_wrappers', 'agree_thresh'):
            raise AttributeError
        return getattr(self._network, name)
JIT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model_traced.pt')
CONFIGURATION_NAME = '3d_fullres_ps192'


# ── Monkey-patch _predict to also capture fg probability ──

def _predict_with_prob(self):
    """Modified _predict that writes fg probability to self._prob_buffer."""
    assert self.pad_mode_data == 'constant'
    if len(self.new_interaction_centers) == 0:
        return

    prediction_center = self.new_interaction_centers[-1]
    zoom_out_factor = min(4, self.new_interaction_zoom_out_factors[-1])

    from nnunetv2.utilities.helpers import dummy_context
    autoctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
    with autoctx:
        input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(
            prediction_center, zoom_out_factor)
        logits = self.network(input_for_predict[None])[0].detach()
        pred = logits.argmax(0)
        prob_fg = torch.softmax(logits.float(), dim=0)[1]
        del input_for_predict, logits

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
            logits = self.network(input_for_predict[None])[0].detach()
            pred = logits.argmax(0)
            prob_fg = torch.softmax(logits.float(), dim=0)[1]
            del input_for_predict, logits
            previous_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox)
            if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
                previous_prediction = interpolate(
                    previous_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]
            has_change = self._detect_change_at_border(pred, previous_prediction)

        if zoom_out_factor == 1:
            paste_tensor(self.interactions[0], pred.half(), scaled_bbox)
            bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                    zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self.target_buffer, pred.to(self.target_buffer.device), bbox)
            paste_tensor(self._prob_buffer, prob_fg.cpu(), bbox)
        else:
            prediction_with_coarse = self.interactions[0]
            if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                pred = (interpolate(pred[None, None].to(float), scaled_patch_size,
                                    mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                prob_fg = interpolate(prob_fg[None, None], scaled_patch_size,
                                      mode='trilinear')[0, 0]
            diff_map, has_diff = self._compute_diff_map(
                pred, self.interactions[0], scaled_bbox, scaled_patch_size)
            paste_tensor(prediction_with_coarse, pred, scaled_bbox)
            bbox_orig = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                         zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self._prob_buffer, prob_fg.cpu(), bbox_orig)
            self._refine_coarse_with_prob(diff_map, prediction_with_coarse)
            del prediction_with_coarse

    self.new_interaction_centers = []
    self.new_interaction_zoom_out_factors = []
    empty_cache(self.device)


def _refine_coarse_with_prob(self, diff_map, prediction_with_coarse):
    """Refinement that also captures prob into _prob_buffer."""
    if self.has_positive_bbox:
        self.interactions[-6][(~(prediction_with_coarse > 0.5))] = 0
        self.has_positive_bbox = False
    bboxes_ordered = generate_bounding_boxes(
        diff_map, self.configuration_manager.patch_size, stride='auto',
        margin=(10, 10, 10), max_depth=3)
    if len(bboxes_ordered) == 0:
        center = self.new_interaction_centers[-1] if self.new_interaction_centers \
            else [s // 2 for s in self.interactions[0].shape]
        bboxes_ordered = [[[ci - pi // 2, ci - pi // 2 + pi]
                           for ci, pi in zip(center, self.configuration_manager.patch_size)]]
    del diff_map
    empty_cache(self.device)
    preallocated_input = torch.zeros(
        (8, *self.configuration_manager.patch_size), device=self.device, dtype=torch.float)
    for refinement_bbox in bboxes_ordered:
        crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox, self.preprocessed_image[0])
        crop_and_pad_into_buffer(preallocated_input[1], refinement_bbox, prediction_with_coarse)
        crop_and_pad_into_buffer(preallocated_input[2:], refinement_bbox, self.interactions[1:])
        logits = self.network(preallocated_input[None])[0].detach()
        pred = logits.argmax(0)
        prob_fg = torch.softmax(logits.float(), dim=0)[1]
        del logits
        paste_tensor(self.interactions[0], pred, refinement_bbox)
        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                zip(refinement_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
        paste_tensor(self.target_buffer, pred.to(self.target_buffer.device), bbox)
        paste_tensor(self._prob_buffer, prob_fg.cpu(), bbox)
        del pred, prob_fg
        preallocated_input.zero_()
    del preallocated_input
    empty_cache(self.device)


class _CUDAGraphNet:
    """Wrap network forward with CUDA Graphs for ~6% speedup.

    Caches a compiled graph per input shape. Returns a reference to
    static output (no clone) — caller must consume before next call.
    """

    def __init__(self, net):
        self.net = net
        self.graphs = {}

    def __call__(self, x):
        key = tuple(x.shape)
        if key not in self.graphs:
            self.si = torch.empty_like(x)
            self.si.copy_(x)
            with torch.cuda.amp.autocast(enabled=True):
                self.so = self.net(self.si)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.cuda.amp.autocast(enabled=True):
                self.so = self.net(self.si)
            self.graphs[key] = g
        self.si.copy_(x)
        self.graphs[key].replay()
        return self.so

    def eval(self):
        return self


def fast_init_session(checkpoint_dir, jit_path, device):
    """Initialize session with JIT network, skipping pickle checkpoint.

    Saves ~2.3s/call by:
      - Skipping network architecture build (562ms)
      - Skipping torch.load pickle deserialization (250ms)
      - Skipping load_state_dict (99ms)
      - Loading pre-traced JIT model directly to GPU (~500ms)
    """
    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )

    # Load configs (fast, ~20ms)
    json_content = load_json(
        os.path.join(checkpoint_dir, 'inference_session_class.json'))
    if isinstance(json_content, str):
        session.point_interaction = PointInteraction_stub(4, True)
        session.preferred_scribble_thickness = [2, 2, 2]
        session.pad_mode_data = 'constant'
        session.interaction_decay = 0.9
    else:
        session.point_interaction = PointInteraction_stub(
            json_content['point_radius'], True)
        session.preferred_scribble_thickness = json_content.get(
            'preferred_scribble_thickness', [2, 2, 2])
        if not isinstance(session.preferred_scribble_thickness, (tuple, list)):
            session.preferred_scribble_thickness = \
                [session.preferred_scribble_thickness] * 3
        session.interaction_decay = json_content.get(
            'interaction_decay', 0.98)
        session.pad_mode_data = json_content.get(
            'pad_mode_image', 'constant')

    dataset_json = load_json(
        os.path.join(checkpoint_dir, 'dataset.json'))
    plans = load_json(
        os.path.join(checkpoint_dir, 'plans.json'))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(
        CONFIGURATION_NAME)

    # Load JIT network directly to GPU (skip build + pickle)
    network = torch.jit.load(jit_path, map_location=device).eval()
    network = _CUDAGraphNet(network)

    # Set session attributes
    session.plans_manager = plans_manager
    session.configuration_manager = configuration_manager
    session.network = network
    session.dataset_json = dataset_json
    session.trainer_name = 'nnInteractiveTrainer'
    session.label_manager = plans_manager.get_label_manager(dataset_json)

    return session


def standard_init_session(checkpoint_dir, device):
    """Standard session initialization (fallback if JIT not available)."""
    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=checkpoint_dir,
        use_fold='all',
    )
    return session


def run_inference(image, spacing, bbox, clicks, clicks_order, prev_pred):
    t0 = time.perf_counter()
    device = torch.device('cuda', 0)

    # Use standard init (JIT incompatible with attention wrapper)
    session = standard_init_session(CHECKPOINT_DIR, device)

    # Setup attention module (no-op if checkpoint missing)
    wrapper, use_relative_pos = setup_attention(session, ATTN_CKPT_PATH, device)

    # Detect round from click count: R5 = last round (5 clicks total)
    total_clicks = 0
    if clicks_order is not None:
        for co in clicks_order:
            total_clicks += sum(1 for k in co if k is not None)
    is_r5 = (total_clicks == N_CLICK_ROUNDS)

    # R5-only TTA: wrap network with flip TTA + agreement gating
    tta_wrapper = None
    if is_r5:
        tta_wrapper = NetworkTTAWrapper(
            session.network, TTA_FLIP_AXES, agree_thresh=TTA_AGREE_THRESH)
        session.network = tta_wrapper

    # Monkey-patch _predict for probability capture
    session._predict = types.MethodType(_predict_with_prob, session)
    session._refine_coarse_with_prob = types.MethodType(
        _refine_coarse_with_prob, session)
    t_load = time.perf_counter() - t0

    session.set_image(image[None].astype(np.float32))
    target_buffer = torch.zeros(
        image.shape, dtype=torch.uint8, device='cpu')
    session.set_target_buffer(target_buffer)
    session._prob_buffer = torch.zeros(
        image.shape, dtype=torch.float32, device='cpu')

    if bbox is not None:
        num_objects = len(bbox)
    elif clicks is not None:
        num_objects = len(clicks)
    else:
        del session
        empty_cache(device)
        return np.zeros(image.shape, dtype=np.uint8)

    # Bbox case → full bypass (encoder features OOD for attention module)
    is_bbox_case = (bbox is not None)
    if is_bbox_case and wrapper is not None:
        wrapper._bypass = True
        set_lora_bypass(session.network, True)

    # Spacing for relative pos (if needed)
    sp = list(spacing) if spacing is not None else [1., 1., 1.]
    spacing_dhw = [sp[2], sp[1], sp[0]] if len(sp) == 3 else [1., 1., 1.]

    per_obj_masks = {}
    per_obj_probs = {}

    for oid in range(1, num_objects + 1):
        target_buffer.zero_()
        session._prob_buffer.zero_()

        if prev_pred is not None:
            session.add_initial_seg_interaction(
                (prev_pred == oid).astype(np.uint8),
                run_prediction=False)
        else:
            session.reset_interactions()

        if bbox is not None:
            b = bbox[oid - 1]
            bbox_fmt = [
                [b['z_min'], b['z_max'] + 1],
                [b['z_mid_y_min'], b['z_mid_y_max'] + 1],
                [b['z_mid_x_min'], b['z_mid_x_max'] + 1],
            ]
            session.add_bbox_interaction(
                bbox_fmt, include_interaction=True,
                run_prediction=False)

        if clicks is not None:
            clicks_here = clicks[oid - 1]
            clicks_order_here = clicks_order[oid - 1]
            fg_ptr = bg_ptr = 0
            for kind in clicks_order_here:
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

        # Set attention token info (no-bbox cases only)
        if not is_bbox_case and wrapper is not None:
            build_token_info_for_object(
                session, oid, num_objects, clicks, clicks_order,
                wrapper, use_relative_pos=use_relative_pos,
                spacing_dhw=spacing_dhw)

        # Enable TTA for this object (R5 only)
        if tta_wrapper is not None:
            tta_wrapper.tta_enabled = is_r5

        if not session.new_interaction_centers:
            per_obj_masks[oid] = np.zeros(image.shape, dtype=np.uint8)
            per_obj_probs[oid] = np.zeros(image.shape, dtype=np.float32)
            continue

        session.new_interaction_centers = [
            session.new_interaction_centers[-1]]
        session.new_interaction_zoom_out_factors = [
            session.new_interaction_zoom_out_factors[-1]]
        session._predict()
        per_obj_masks[oid] = (target_buffer.numpy() > 0).astype(np.uint8).copy()
        per_obj_probs[oid] = session._prob_buffer.numpy().copy()

    # Maxprob assembly (all cases)
    result = _maxprob_assemble(per_obj_masks, per_obj_probs, num_objects)

    t_total = time.perf_counter() - t0
    print(f"[predict.py] {num_objects} objects, bbox={is_bbox_case}, "
          f"clicks={total_clicks}, tta={is_r5}, "
          f"load={t_load:.1f}s, total={t_total:.1f}s", flush=True)

    del session
    empty_cache(device)
    return result


def _maxprob_assemble(per_obj_masks, per_obj_probs, num_objects):
    """Maxprob assembly with safety net.

    Overlap voxels assigned to highest-probability object.
    Safety net guarantees every prompted object retains at least MIN_VOXELS
    voxels, preventing empty predictions that crash the evaluator.
    """
    MIN_VOXELS = 5
    oids = list(range(1, num_objects + 1))
    if not oids or oids[0] not in per_obj_masks:
        return np.zeros_like(list(per_obj_masks.values())[0] if per_obj_masks
                             else np.empty(0), dtype=np.uint8)
    shape = per_obj_masks[oids[0]].shape
    result = np.zeros(shape, dtype=np.uint8)
    for oid in oids:
        result[per_obj_masks[oid] > 0] = oid

    stacked = sum(per_obj_masks[o].astype(np.int32) for o in oids)
    overlap_mask = stacked > 1
    if overlap_mask.any():
        idx = np.argwhere(overlap_mask)
        for voxel in idx:
            v = tuple(voxel)
            claiming = [o for o in oids if per_obj_masks[o][v] > 0]
            best_oid = max(claiming, key=lambda o: per_obj_probs[o][v])
            result[v] = best_oid

    for oid in oids:
        if (result == oid).sum() < MIN_VOXELS and per_obj_masks[oid].any():
            raw_probs = per_obj_probs[oid].copy()
            raw_probs[per_obj_masks[oid] == 0] = -1
            flat_idx = np.argsort(raw_probs.ravel())[::-1][:MIN_VOXELS]
            coords = np.unravel_index(flat_idx, shape)
            for i in range(len(coords[0])):
                result[coords[0][i], coords[1][i], coords[2][i]] = oid

    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case_path", required=True)
    p.add_argument("--save_path", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    case_path = Path(args.case_path)
    save_path = Path(args.save_path)

    if not case_path.is_file():
        sys.exit(f"[predict.py] ERROR: {case_path} not found.")

    data = np.load(case_path, allow_pickle=True)
    image = data["imgs"]
    spacing = tuple(data["spacing"])
    bbox = data.get("boxes")
    clicks = data.get("clicks")
    clicks_order = data.get("clicks_order")
    prev_pred = data.get("prev_pred")

    seg = run_inference(image, spacing, bbox, clicks, clicks_order, prev_pred)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, segs=seg.astype(np.uint8))
    print(f"[predict.py] Saved to {save_path}")


if __name__ == "__main__":
    main()
