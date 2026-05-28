"""
RefinementModule on-policy training (v2).

Mirrors the multi-round structure of run_decoder_attn.py but:
  - nnInt runs as a real inference session (no_grad) so pred_nn_k comes from
    the exact pipeline used at eval time → zero train/eval gap.
  - RefinementModule is post-hoc on the assembled per-object sigmoid.
  - pred_prev is the SOFT previous-round refined output for the same oid.
  - Memory tokens are encoded from the module's own refined outputs (detached),
    not from GT.
  - Training data is cross-modal (see data/crossmodal_200.txt).
  - Per-object refinement — no cross-oid memory.

Usage:
    python -u training/run_refinement_v2.py \
        --file_list data/crossmodal_200.txt \
        --feats_dir experiments/refinement_train_feats \
        --save_dir experiments/refinement_v2 \
        --gpu 0 --epochs 10 --lr 1e-3 --num_rounds 4
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import cc3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.utils.crop import paste_tensor, crop_and_pad_into_buffer
from nnInteractive.utils.bboxes import generate_bounding_boxes
from torch.nn.functional import interpolate as F_interpolate

from training.refinement_module import RefinementModule, count_parameters

CHECKPOINT_PATH = Path(
    "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models"
    "/nnInteractive_v1.0_fold_all"
)

N_ROUNDS_DEFAULT = 4


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch _predict to also write softmax[1] into _prob_buffer
# (same as scripts/eval_refinement.py)
# ─────────────────────────────────────────────────────────────────────────────
def _predict_with_prob(self):
    assert self.pad_mode_data == 'constant'
    if len(self.new_interaction_centers) == 0:
        return
    prediction_center = self.new_interaction_centers[-1]
    zoom_out_factor = min(4, self.new_interaction_zoom_out_factors[-1])

    autoctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
    with autoctx:
        inp, sps, sbbox = self._build_network_input(prediction_center, zoom_out_factor)
        logits = self.network(inp[None])[0].detach().float()
        probs = torch.softmax(logits, dim=0)
        pred = logits.argmax(0)
        del inp
        pp = crop_and_pad_nd(self.interactions[0], sbbox)
        if pp.shape != pred.shape:
            pp = F_interpolate(pp[None, None].float(), pred.shape, mode='nearest')[0, 0]
        hc = self._detect_change_at_border(pred, pp)
        del pp
        while hc and self.do_autozoom and zoom_out_factor < 4:
            zoom_out_factor = min(4, zoom_out_factor * 1.5)
            inp, sps, sbbox = self._build_network_input(prediction_center, zoom_out_factor)
            logits = self.network(inp[None])[0].detach().float()
            probs = torch.softmax(logits, dim=0)
            pred = logits.argmax(0)
            del inp
            pp = crop_and_pad_nd(self.interactions[0], sbbox)
            if pp.shape != pred.shape:
                pp = F_interpolate(pp[None, None].float(), pred.shape, mode='nearest')[0, 0]
            hc = self._detect_change_at_border(pred, pp)
        if zoom_out_factor == 1:
            paste_tensor(self.interactions[0], pred.half(), sbbox)
            bbox = [[i[0] + b[0], i[1] + b[0]] for i, b in
                    zip(sbbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self.target_buffer,
                         pred.to(self.target_buffer.device)
                         if isinstance(self.target_buffer, torch.Tensor) else pred.cpu(),
                         bbox)
            paste_tensor(self._prob_buffer, probs[1].cpu(), bbox)
        else:
            pwc = self.interactions[0]
            if pred.shape != tuple(sps):
                pred = (F_interpolate(pred[None, None].float(), sps, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)
                pfg = F_interpolate(probs[1:2][None], sps, mode='trilinear')[0, 0]
            else:
                pfg = probs[1]
            del logits, probs
            dm, _ = self._compute_diff_map(pred, self.interactions[0], sbbox, sps)
            paste_tensor(pwc, pred, sbbox)
            bbox_o = [[i[0] + b[0], i[1] + b[0]] for i, b in
                      zip(sbbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self._prob_buffer, pfg.cpu(), bbox_o)
            self._rcl_with_prob(dm, pwc)
            del pwc
    self.new_interaction_centers = []
    self.new_interaction_zoom_out_factors = []
    empty_cache(self.device)


def _rcl_with_prob(self, diff_map, pwc):
    if self.has_positive_bbox:
        self.interactions[-6][(~(pwc > 0.5))] = 0
        self.has_positive_bbox = False
    bbs = generate_bounding_boxes(
        diff_map, self.configuration_manager.patch_size,
        stride='auto', margin=(10, 10, 10), max_depth=3)
    if not bbs:
        c = self.new_interaction_centers[-1] if self.new_interaction_centers \
            else [s // 2 for s in self.interactions[0].shape]
        bbs = [[[ci - pi // 2, ci - pi // 2 + pi]
                for ci, pi in zip(c, self.configuration_manager.patch_size)]]
    del diff_map
    empty_cache(self.device)
    buf = torch.zeros((8, *self.configuration_manager.patch_size),
                      device=self.device, dtype=torch.float)
    for rb in bbs:
        crop_and_pad_into_buffer(buf[0], rb, self.preprocessed_image[0])
        crop_and_pad_into_buffer(buf[1], rb, pwc)
        crop_and_pad_into_buffer(buf[2:], rb, self.interactions[1:])
        logits = self.network(buf[None])[0].detach().float()
        probs = torch.softmax(logits, dim=0)
        pred = logits.argmax(0)
        paste_tensor(self.interactions[0], pred, rb)
        bbox = [[i[0] + b[0], i[1] + b[0]] for i, b in
                zip(rb, self.preprocessed_props['bbox_used_for_cropping'])]
        paste_tensor(self.target_buffer,
                     pred.to(self.target_buffer.device)
                     if isinstance(self.target_buffer, torch.Tensor) else pred.cpu(),
                     bbox)
        paste_tensor(self._prob_buffer, probs[1].cpu(), bbox)
        del pred, logits, probs
        buf.zero_()
    del buf
    empty_cache(self.device)


def create_session(ckpt_dir, device):
    session = nnInteractiveInferenceSession(
        device=device, use_torch_compile=False, verbose=False,
        torch_n_threads=os.cpu_count(), do_autozoom=True, use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=str(ckpt_dir), use_fold='all',
    )
    session._predict = types.MethodType(_predict_with_prob, session)
    session._rcl_with_prob = types.MethodType(_rcl_with_prob, session)
    session._rcl = session._rcl_with_prob
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Click generation — MUST match eval_bottleneck_attn exactly to avoid
# train/eval divergence. Do not diverge without updating eval too.
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from eval_bottleneck_attn import (
    sample_coord as _sample_coord,
    compute_edt_safe as _compute_edt_safe,
)


def generate_click_eval(seg_cls, gt_cls):
    """Eval-style click generation for ONE class; returns (center, is_fg) or
    (None, None) if no error. Mirrors eval_bottleneck_attn.generate_click's
    per-class branch (deterministic sample_coord + same EDT computation)."""
    err = (seg_cls != gt_cls).astype(np.uint8)
    if err.sum() == 0:
        return None, None
    cc = cc3d.connected_components(err, connectivity=26)
    sizes = np.bincount(cc.flat); sizes[0] = 0
    largest = cc == np.argmax(sizes)
    edt = _compute_edt_safe(largest) * largest
    if edt.sum() == 0:
        edt = largest.astype(np.float64)
    center = _sample_coord(edt)
    is_fg = gt_cls[center] > 0
    return center, bool(is_fg)


# ─────────────────────────────────────────────────────────────────────────────
# Assembly (full-res, maxprob)
# ─────────────────────────────────────────────────────────────────────────────
def maxprob_assemble(per_obj_masks, per_obj_probs, labels_sorted):
    """Keys = oid (1..K). Labels_sorted = list of original GT label values.
    Returns assembled label map (uint8)."""
    shape = next(iter(per_obj_masks.values())).shape
    result = np.zeros(shape, dtype=np.uint8)
    # Last-wins pass
    for oid in sorted(per_obj_masks.keys()):
        label = labels_sorted[oid - 1]
        result[per_obj_masks[oid] > 0] = label
    # Overlap resolution via max fg prob
    stacked = sum(per_obj_masks[o].astype(np.int32) for o in per_obj_masks)
    overlap = stacked > 1
    if overlap.any():
        idx = np.argwhere(overlap)
        oids = sorted(per_obj_masks.keys())
        for v in idx:
            v = tuple(v)
            claiming = [o for o in oids if per_obj_masks[o][v] > 0]
            best = max(claiming, key=lambda o: per_obj_probs[o][v])
            result[v] = labels_sorted[best - 1]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# F_global + click-dist helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_F_global(feats_dir: Path, case_stem: str, device: torch.device, R: int):
    """Return (192, R, R, R) tensor on device, upsampled from F_R."""
    p = feats_dir / f"{case_stem}.pt"
    if not p.exists():
        return None
    blob = torch.load(p, weights_only=False, map_location='cpu')
    feat = blob['F_global_low'].float().to(device)    # (192, F_R, F_R, F_R)
    feat = F_interpolate(feat[None], size=(R, R, R), mode='trilinear',
                         align_corners=False)[0]
    return feat


def compute_click_dist_map(shape_low, click_positions_low):
    if not click_positions_low:
        return np.zeros(shape_low, dtype=np.float32)
    mask = np.ones(shape_low, dtype=bool)
    for (z, y, x) in click_positions_low:
        if 0 <= z < shape_low[0] and 0 <= y < shape_low[1] and 0 <= x < shape_low[2]:
            mask[z, y, x] = False
    edt = distance_transform_edt(mask)
    return (1.0 - np.clip(edt, 0, 20) / 20.0).astype(np.float32)


def compute_cd_channels(click_list_for_oid, orig_shape, R):
    """click_list_for_oid = list of dicts {'pos_image': (z,y,x), 'is_fg': bool}."""
    fg_low, bg_low = [], []
    for c in click_list_for_oid:
        z = int(c['pos_image'][0] / max(1, orig_shape[0] - 1) * (R - 1))
        y = int(c['pos_image'][1] / max(1, orig_shape[1] - 1) * (R - 1))
        x = int(c['pos_image'][2] / max(1, orig_shape[2] - 1) * (R - 1))
        (fg_low if c['is_fg'] else bg_low).append((z, y, x))
    cd_fg = compute_click_dist_map((R, R, R), fg_low)
    cd_bg = compute_click_dist_map((R, R, R), bg_low)
    return cd_fg, cd_bg


def click_dicts_to_token_list(click_list_for_oid, orig_shape):
    """Convert to RefinementModule.encode_memory_token format."""
    out = []
    for c in click_list_for_oid:
        pos = c['pos_image']
        pn = (pos[0] / max(1, orig_shape[0] - 1),
              pos[1] / max(1, orig_shape[1] - 1),
              pos[2] / max(1, orig_shape[2] - 1))
        out.append({'pos_norm': pn, 'fg': bool(c['is_fg']), 'round': c['round']})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
class DiceBCE(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        inter = (pred * target).sum(dim=[2, 3, 4])
        union = pred.sum(dim=[2, 3, 4]) + target.sum(dim=[2, 3, 4])
        dice = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        bce = F.binary_cross_entropy(pred, target, reduction='none').mean(dim=[1, 2, 3, 4])
        return 0.5 * dice.mean() + 0.5 * bce.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class RefinementV2Trainer:

    def __init__(self, gpu: int, lr: float, num_rounds: int, R: int, R_attn: int,
                 feats_dir: Path, frozen: bool = False):
        self.device = torch.device('cuda', gpu)
        self.num_rounds = num_rounds
        self.R = R
        self.R_attn = R_attn
        self.feats_dir = feats_dir
        self.frozen = frozen

        # Session is created per-case in _run_case so each case starts from
        # a clean CUDA workspace (matches eval pipeline). Reusing a single
        # session across cases introduces non-determinism on large CT volumes.
        self.session = None

        print(f"Building RefinementModule (R={R}, R_attn={R_attn})...")
        self.refiner = RefinementModule(R=R, R_attn=R_attn).to(self.device)
        n_params = count_parameters(self.refiner)
        print(f"  params: {n_params:,}")

        self.criterion = DiceBCE().to(self.device)

        if not frozen:
            self.optimizer = torch.optim.AdamW(
                self.refiner.parameters(), lr=lr, weight_decay=1e-5)
        else:
            self.optimizer = None

    # ── Core per-case training step ────────────────────────────────────────
    def _run_case(self, image_np, gt_np, case_stem, train: bool = True,
                  capture=None):
        """One full case: multi-round on-policy loop with refinement.
        Returns mean loss or None on skip.

        capture: optional dict; if provided, populated with:
            'refined_per_round': list (len=num_rounds) of dict oid→full-res refined prob (np.float32)
            'clicks_per_round':  list (len=num_rounds) of dict oid→list of click dicts at that round
            'assembled_final':   final prev_assembled (uint8 label map)
        """
        device = self.device
        shape = image_np.shape

        labels_sorted = sorted([int(l) for l in np.unique(gt_np) if l > 0])
        K = len(labels_sorted)
        if K == 0:
            return None

        F_global_R = load_F_global(self.feats_dir, case_stem, device, self.R)
        if F_global_R is None:
            return None

        # Fresh session per case (matches eval, avoids state leaks / CUDA
        # workspace carry-over → makes train pipeline deterministic)
        if self.session is not None:
            del self.session
            torch.cuda.empty_cache()
        self.session = create_session(CHECKPOINT_PATH, self.device)
        self.session.set_image(image_np[None].astype(np.float32))
        target_buffer = torch.zeros(shape, dtype=torch.uint8, device='cpu')
        self.session.set_target_buffer(target_buffer)
        self.session._prob_buffer = torch.zeros(shape, dtype=torch.float32, device='cpu')

        # Low-res GT per object
        gt_low = F_interpolate(
            torch.from_numpy(gt_np.astype(np.int64))[None, None].float(),
            size=(self.R, self.R, self.R), mode='nearest'
        )[0, 0].long().numpy()

        # State
        click_hist = defaultdict(list)           # oid (1..K) → list of dicts
        prev_assembled = None                     # full-res int label map from last round
        refined_prev_low = {oid: torch.zeros(1, 1, self.R, self.R, self.R, device=device)
                            for oid in range(1, K + 1)}
        memory_bank = {oid: [] for oid in range(1, K + 1)}

        total_loss = 0.0
        n_bwd = 0

        if capture is not None:
            capture['refined_per_round'] = []
            capture['clicks_per_round'] = []

        if train:
            self.refiner.train()
            self.optimizer.zero_grad()
        else:
            self.refiner.eval()

        for round_idx in range(self.num_rounds):
            # ── 1) Generate clicks (EXACT parity with eval R0 + R1+ paths) ──
            for oid_idx, label in enumerate(labels_sorted):
                oid = oid_idx + 1
                gt_cls = (gt_np == label).astype(np.uint8)
                if round_idx == 0:
                    # Mirror eval_decoder_attn_maxprob lines 315-321:
                    if gt_cls.sum() > 0:
                        edt = _compute_edt_safe(gt_cls)
                        c = _sample_coord(edt)
                        click_hist[oid].append(
                            {'pos_image': tuple(c), 'is_fg': True,
                             'round': round_idx})
                else:
                    if prev_assembled is None:
                        continue
                    seg_cls = (prev_assembled == label).astype(np.uint8)
                    c, is_fg = generate_click_eval(seg_cls, gt_cls)
                    if c is not None:
                        click_hist[oid].append(
                            {'pos_image': tuple(c), 'is_fg': is_fg,
                             'round': round_idx})

            # ── 2a) Phase 1: all nnInt forwards (no_grad), collect per-oid probs ──
            # Mirrors eval_refinement.run_refinement structure; avoids
            # interleaving nnInt and refinement forwards, which caused fp16
            # kernel non-determinism on large CT cases.
            per_obj_probs_nn = {}      # full-res nnInt sigmoid, per oid
            for oid_idx, label in enumerate(labels_sorted):
                oid = oid_idx + 1
                if not click_hist[oid]:
                    continue
                target_buffer.zero_()
                self.session._prob_buffer.zero_()
                if prev_assembled is not None:
                    self.session.add_initial_seg_interaction(
                        (prev_assembled == label).astype(np.uint8),
                        run_prediction=False)
                else:
                    self.session.reset_interactions()
                for c in click_hist[oid]:
                    self.session.add_point_interaction(
                        list(c['pos_image']),
                        include_interaction=bool(c['is_fg']),
                        run_prediction=False)
                if not self.session.new_interaction_centers:
                    continue
                self.session.new_interaction_centers = [
                    self.session.new_interaction_centers[-1]]
                self.session.new_interaction_zoom_out_factors = [
                    self.session.new_interaction_zoom_out_factors[-1]]
                with torch.no_grad():
                    self.session._predict()
                per_obj_probs_nn[oid] = self.session._prob_buffer.numpy().astype(
                    np.float32).copy()

            # ── 2b) Phase 2: all refinements (grad), per-oid loss + backward ──
            per_obj_masks = {}
            per_obj_probs_refined = {}
            for oid_idx, label in enumerate(labels_sorted):
                oid = oid_idx + 1
                if oid not in per_obj_probs_nn:
                    continue

                prob_full = per_obj_probs_nn[oid]
                prob_low = F_interpolate(
                    torch.from_numpy(prob_full)[None, None], size=(self.R,) * 3,
                    mode='trilinear', align_corners=False)[0]
                pred_nn_k = prob_low.to(device).unsqueeze(0)
                pred_prev = refined_prev_low[oid]

                cd_fg_np, cd_bg_np = compute_cd_channels(
                    click_hist[oid], shape, self.R)
                cd_fg = torch.from_numpy(cd_fg_np)[None, None].to(device)
                cd_bg = torch.from_numpy(cd_bg_np)[None, None].to(device)

                mem = memory_bank[oid]
                mem_tokens = torch.stack(mem, 0)[None] if mem else None

                refined_low, delta = self.refiner(
                    F_global_R[None], pred_nn_k, pred_prev, cd_fg, cd_bg,
                    mem_tokens)

                gt_oid_low = torch.from_numpy(
                    (gt_low == label).astype(np.float32))[None, None].to(device)
                loss = self.criterion(refined_low, gt_oid_low)

                total_loss += loss.item()
                n_bwd += 1

                if train:
                    (loss / max(1, K * self.num_rounds)).backward()

                refined_prev_low[oid] = refined_low.detach()
                this_round_clicks = [
                    c for c in click_dicts_to_token_list(click_hist[oid], shape)
                    if c['round'] == round_idx
                ]
                tok = self.refiner.encode_memory_token(
                    refined_low.detach(), this_round_clicks, round_idx)
                memory_bank[oid].append(tok[0].detach())

                with torch.no_grad():
                    refined_full = F_interpolate(
                        refined_low, size=shape, mode='trilinear',
                        align_corners=False)[0, 0].cpu().numpy()
                per_obj_probs_refined[oid] = refined_full
                per_obj_masks[oid] = (refined_full > 0.5).astype(np.uint8)

            # ── 3) Assembly at full resolution ──
            if per_obj_masks:
                prev_assembled = maxprob_assemble(
                    per_obj_masks, per_obj_probs_refined, labels_sorted)

            if capture is not None:
                capture['refined_per_round'].append(
                    {oid: arr.copy() for oid, arr in per_obj_probs_refined.items()})
                # Snapshot clicks accumulated up to this round (per oid)
                capture['clicks_per_round'].append(
                    {oid: [dict(c) for c in click_hist[oid]]
                     for oid in range(1, K + 1)})

        if capture is not None:
            capture['assembled_final'] = (
                prev_assembled.copy() if prev_assembled is not None else None)
            capture['labels_sorted'] = list(labels_sorted)

        # Optimizer step (once per case)
        if train and n_bwd > 0:
            nn.utils.clip_grad_norm_(self.refiner.parameters(), max_norm=5.0)
            self.optimizer.step()

        return total_loss / max(1, n_bwd)

    # ── Epoch / training loops ─────────────────────────────────────────────
    def train_epoch(self, files, epoch):
        random.shuffle(files)
        losses = []
        skipped = 0
        t0 = time.time()

        for fi, fp in enumerate(files):
            try:
                data = np.load(fp, allow_pickle=True)
                image = data['imgs'].astype(np.float32)
                gt = data['gts'].astype(np.uint8)
            except Exception as e:
                print(f"  load fail {fp.name}: {e!r}")
                skipped += 1
                continue

            # Skip extremely large volumes that would OOM during nnInt session
            MAX_TRAIN_VOXELS = 50_000_000
            if image.size > MAX_TRAIN_VOXELS:
                print(f"  skip large volume {fp.name}: {image.shape} "
                      f"({image.size/1e6:.1f}M voxels > {MAX_TRAIN_VOXELS/1e6:.0f}M)")
                skipped += 1
                continue

            try:
                loss = self._run_case(image, gt, fp.stem, train=not self.frozen)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM {fp.name}, skipping")
                skipped += 1
                continue
            except Exception as e:
                print(f"  error {fp.name}: {e!r}")
                import traceback; traceback.print_exc()
                skipped += 1
                continue

            if loss is None:
                skipped += 1
                continue
            losses.append(loss)

            if (fi + 1) % 20 == 0:
                dt = time.time() - t0
                mean_l = float(np.mean(losses[-20:])) if losses else 0
                print(f"  [{fi+1}/{len(files)}] loss={mean_l:.4f} "
                      f"skip={skipped} t={dt:.0f}s", flush=True)

        dt = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch}: loss={mean_loss:.4f} n={len(losses)} "
              f"skip={skipped} time={dt:.0f}s", flush=True)
        return mean_loss

    def save_checkpoint(self, path: Path, epoch: int, loss: float, saved_args: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': self.refiner.state_dict(),
            'epoch': epoch, 'loss': loss, 'args': saved_args,
        }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file_list', type=Path, required=True)
    p.add_argument('--feats_dir', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_train_feats')
    p.add_argument('--save_dir', type=Path,
                   default=PROJECT_ROOT / 'experiments/refinement_v2')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_rounds', type=int, default=N_ROUNDS_DEFAULT)
    p.add_argument('--R', type=int, default=96)
    p.add_argument('--R_attn', type=int, default=48)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_files', type=int, default=0)
    p.add_argument('--freeze_sanity', action='store_true',
                   help="Frozen forward only, report baseline loss")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.file_list) as f:
        files = [Path(ln.strip()) for ln in f if ln.strip()]
    if args.max_files:
        files = files[:args.max_files]

    # Filter to those with F_global precomputed
    have_feats = {p.stem for p in args.feats_dir.glob('*.pt')}
    before = len(files)
    files = [f for f in files if f.stem in have_feats]
    print(f"{len(files)}/{before} files have F_global; training on those")
    assert files

    trainer = RefinementV2Trainer(
        gpu=args.gpu, lr=args.lr, num_rounds=args.num_rounds,
        R=args.R, R_attn=args.R_attn, feats_dir=args.feats_dir,
        frozen=args.freeze_sanity,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    saved_args = vars(args).copy()
    saved_args = {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in saved_args.items()}

    if args.freeze_sanity:
        print("\n=== Freeze sanity check (zero-init identity → loss equals baseline) ===")
        trainer.train_epoch(files[:10], epoch=-1)
        return

    for epoch in range(args.epochs):
        mean_loss = trainer.train_epoch(files, epoch)
        ckpt = args.save_dir / f"epoch_{epoch:02d}.pt"
        trainer.save_checkpoint(ckpt, epoch, mean_loss, saved_args)
        print(f"  → saved {ckpt}")

    print("Training done.")


if __name__ == '__main__':
    main()
