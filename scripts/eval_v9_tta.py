#!/usr/bin/env python3
"""
Combined evaluation: v9 attention module + network-level flip TTA.

Runs 4 variants on a test set:
  1. baseline        — no TTA, no v9
  2. v9_only         — v9 attention, no TTA
  3. tta_only        — flip_w TTA with bbox>5000 gate, no v9
  4. v9_tta          — v9 attention + flip_w TTA with bbox>5000 gate

For each variant, runs the official evaluation protocol:
  - Round 0: bbox → predict (skipped if no bbox)
  - Rounds 1-5: generate click from error → predict
"""
from __future__ import annotations
import os, sys, time, json, argparse, types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

from SurfaceDice import compute_dice_coefficient
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnInteractive.utils.crop import paste_tensor, crop_and_pad_into_buffer
from nnInteractive.utils.bboxes import generate_bounding_boxes
import cc3d
from scipy.ndimage import distance_transform_edt

# Import TTA wrapper from eval_network_tta
from scripts.eval_network_tta import (
    NetworkTTAWrapper, _predict_with_tta,
    compute_multi_class_dsc, compute_edt_safe, sample_coord,
    generate_click_official, _get_object_bbox_vol,
)

# Import attention module
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


def make_session_with_v9(checkpoint_dir, device, attn_ckpt_path,
                         flip_axes_list=None, bbox_vol_thresh=0):
    """Create session with v9 attention + optional TTA wrapper.

    Returns (session, attn_wrapper, use_relative_pos).
    attn_wrapper is None if no attention checkpoint.
    """
    session = nnInteractiveInferenceSession(
        device=device, use_torch_compile=False, verbose=False,
        do_autozoom=True, use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=str(checkpoint_dir), use_fold='all')

    # Setup v9 attention (modifies decoder in-place, adds LoRA)
    attn_wrapper = None
    use_relative_pos = False
    if attn_ckpt_path is not None and os.path.exists(str(attn_ckpt_path)):
        attn_wrapper, use_relative_pos = setup_attention(
            session, str(attn_ckpt_path), device)
        print(f"  [v9] Attention loaded, use_relative_pos={use_relative_pos}", flush=True)

    # Wrap with TTA if requested (wraps the attention-modified network)
    if flip_axes_list is not None:
        session.network = NetworkTTAWrapper(
            session.network, flip_axes_list, strategy='always')
        session._predict = types.MethodType(_predict_with_tta, session)

    return session, attn_wrapper, use_relative_pos


def run_one_case_v9_tta(npz_path, gt_path, device, checkpoint_dir,
                        max_obj=8,
                        use_v9=False, attn_ckpt_path=None,
                        flip_axes_list=None, bbox_vol_thresh=0):
    """Run one case with optional v9 attention + optional TTA.

    Handles the attention bypass logic:
    - bbox round (it==0): bypass=True for attention and LoRA
    - click rounds (it>0): bypass=False, build token_info per object
      (but only if the object has multi-object context)
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

    # Create session
    attn_ckpt = str(attn_ckpt_path) if use_v9 and attn_ckpt_path else None
    session, attn_wrapper, use_relative_pos = make_session_with_v9(
        checkpoint_dir, device, attn_ckpt,
        flip_axes_list=flip_axes_list, bbox_vol_thresh=bbox_vol_thresh)

    session.set_image(image[None].astype(np.float32))
    target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
    session.set_target_buffer(target_buffer)

    tta_wrapper = None
    if flip_axes_list is not None:
        tta_wrapper = session.network  # This is the NetworkTTAWrapper

    # Compute spacing for relative pos encoding
    sp = list(spacing) if spacing is not None else [1., 1., 1.]
    spacing_dhw = [sp[2], sp[1], sp[0]] if len(sp) == 3 else [1., 1., 1.]

    # Build cumulative click state (mimics official eval)
    clicks_cls = [{'fg': [], 'bg': []} for _ in gt_classes]
    clicks_order = [[] for _ in gt_classes]
    prev_pred = None
    dscs = []

    has_bbox = boxes is not None
    is_bbox_case = has_bbox

    for it in range(N_CLICKS + 1):
        if it == 0:
            if not has_bbox:
                dscs.append(0)
                continue
        else:
            if prev_pred is None:
                prev_pred = np.zeros_like(gt, dtype=np.uint8)
            generate_click_official(prev_pred, gt, gt_classes,
                                    clicks_cls, clicks_order)

        # --- Round-level TTA control ---
        if tta_wrapper is not None:
            tta_wrapper.tta_enabled = True  # Will be refined per-object below
        else:
            pass  # no TTA

        # --- Round-level attention bypass ---
        # Bbox round: bypass attention entirely (features OOD)
        if attn_wrapper is not None:
            if it == 0 and is_bbox_case:
                attn_wrapper._bypass = True
                set_lora_bypass(session.network, True)
            else:
                # Click rounds: will set per-object below
                pass

        result = np.zeros(image.shape, dtype=np.uint8)
        for oid_idx, cls in enumerate(gt_classes):
            target_buffer.zero_()

            # --- Per-object TTA gating based on bbox volume ---
            if tta_wrapper is not None and bbox_vol_thresh > 0:
                obj_bbox_vol = _get_object_bbox_vol(
                    boxes, oid_idx, prev_pred, cls, it)
                tta_wrapper.tta_enabled = (obj_bbox_vol >= bbox_vol_thresh)
            elif tta_wrapper is not None:
                tta_wrapper.tta_enabled = True

            # --- Per-object attention bypass for click rounds ---
            if attn_wrapper is not None and not (it == 0 and is_bbox_case):
                # Click round: use build_token_info_for_object
                # It handles bypass internally (bypasses if no multi-obj context)
                pass  # Will call build_token_info_for_object after interactions

            # Add prev_pred or reset
            if prev_pred is not None:
                session.add_initial_seg_interaction(
                    (prev_pred == cls).astype(np.uint8), run_prediction=False)
            else:
                session.reset_interactions()

            # Add bbox interaction
            if it == 0 and has_bbox and oid_idx < len(boxes):
                bb = boxes[oid_idx]
                session.add_bbox_interaction(
                    [[int(bb['z_min']), int(bb['z_max']) + 1],
                     [int(bb['z_mid_y_min']), int(bb['z_mid_y_max']) + 1],
                     [int(bb['z_mid_x_min']), int(bb['z_mid_x_max']) + 1]],
                    include_interaction=True, run_prediction=False)

            # Add click interactions
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
                continue

            # Set attention token info for click rounds (no-bbox cases)
            if attn_wrapper is not None and not (it == 0 and is_bbox_case):
                # oid is 1-indexed in build_token_info_for_object
                build_token_info_for_object(
                    session, oid_idx + 1, len(gt_classes),
                    clicks_cls, clicks_order,
                    attn_wrapper, use_relative_pos=use_relative_pos,
                    spacing_dhw=spacing_dhw)

            # Keep only last interaction center
            session.new_interaction_centers = [session.new_interaction_centers[-1]]
            session.new_interaction_zoom_out_factors = [
                session.new_interaction_zoom_out_factors[-1]]
            session._predict()
            result[target_buffer.numpy() > 0] = cls

        dsc = compute_multi_class_dsc(gt, result)
        dscs.append(dsc)
        prev_pred = result.copy()

    del session
    empty_cache(torch.device(device))

    # Compute AUC over click rounds only
    click_dscs = np.array(dscs[1:]) if has_bbox else np.array(dscs[1:])
    auc = integrate.cumulative_trapezoid(
        click_dscs, np.arange(len(click_dscs)))[-1] if len(click_dscs) >= 2 else 0
    return {'DSC_AUC': auc, 'DSC_Final': dscs[-1], 'DSC_bbox': dscs[0],
            'per_round': dscs}


def run_variant(variant_name, val_files, device, max_obj,
                use_v9, attn_ckpt_path,
                flip_axes_list, bbox_vol_thresh):
    """Run one variant over all cases."""
    print(f"\n{'='*70}", flush=True)
    print(f"VARIANT: {variant_name}", flush=True)
    print(f"  v9={use_v9}, TTA={'flip_w' if flip_axes_list else 'none'}, "
          f"bbox_vol_thresh={bbox_vol_thresh}", flush=True)
    print(f"  {len(val_files)} cases", flush=True)
    print(f"{'='*70}", flush=True)

    rows = []
    all_per_round = []
    for i, vf in enumerate(val_files):
        t0 = time.time()
        try:
            r = run_one_case_v9_tta(
                vf['npz'], vf['gt'], device, DEFAULT_CHECKPOINT,
                max_obj=max_obj,
                use_v9=use_v9, attn_ckpt_path=attn_ckpt_path,
                flip_axes_list=flip_axes_list,
                bbox_vol_thresh=bbox_vol_thresh)
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
        has_bbox = vf.get('has_bbox', True)
        bbox_tag = 'bbox' if has_bbox else 'nobbox'
        print(f"  [{i+1}/{len(val_files)}] {vf['fname'][:50]} ({bbox_tag}): "
              f"AUC={r['DSC_AUC']:.4f} Final={r['DSC_Final']:.4f} ({elapsed:.0f}s)",
              flush=True)
        print(f"    rounds=[{rounds_str}]", flush=True)

        rows.append({
            'case': vf['fname'], 'dataset': vf['dataset'],
            'has_bbox': has_bbox,
            'DSC_AUC': r['DSC_AUC'], 'DSC_Final': r['DSC_Final'],
            'DSC_bbox': r['DSC_bbox'], 'time_s': elapsed,
            'variant': variant_name,
        })
        all_per_round.append({
            'case': vf['fname'],
            'rounds': r['per_round'],
        })

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n  --- {variant_name} Summary ({len(df)} cases) ---", flush=True)
        print(f"  mean AUC={df.DSC_AUC.mean():.4f}  "
              f"Final={df.DSC_Final.mean():.4f}  "
              f"time={df.time_s.mean():.1f}s/case", flush=True)

        # Split by bbox/no-bbox
        for has_bb in [True, False]:
            sub = df[df.has_bbox == has_bb]
            if len(sub) > 0:
                tag = 'bbox' if has_bb else 'no-bbox'
                print(f"    {tag:10s} n={len(sub):3d} "
                      f"AUC={sub.DSC_AUC.mean():.4f} "
                      f"Final={sub.DSC_Final.mean():.4f} "
                      f"time={sub.time_s.mean():.1f}s", flush=True)

        # Per dataset
        for ds, sub in df.groupby('dataset'):
            print(f"    {ds:35s} n={len(sub):3d} "
                  f"AUC={sub.DSC_AUC.mean():.4f} "
                  f"Final={sub.DSC_Final.mean():.4f}", flush=True)

    return rows, all_per_round


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_obj', type=int, default=8)
    parser.add_argument('--val_json', required=True)
    parser.add_argument('--n_cases', type=int, default=0)
    parser.add_argument('--variants', default='all',
                        help='Comma-separated: baseline,v9_only,tta_only,v9_tta or "all"')
    parser.add_argument('--v9_ckpt', default=str(V9_CHECKPOINT),
                        help='Path to v9 attention checkpoint')
    parser.add_argument('--bbox_vol_thresh', type=int, default=5000,
                        help='Min bbox volume to enable TTA per object')
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
    print(f"bbox_vol_thresh: {args.bbox_vol_thresh}", flush=True)

    if args.variants == 'all':
        variants = ['baseline', 'v9_only', 'tta_only', 'v9_tta']
    else:
        variants = args.variants.split(',')

    all_rows = []
    all_per_round = {}

    for variant in variants:
        if variant == 'baseline':
            rows, per_round = run_variant(
                'baseline', val_files, device, args.max_obj,
                use_v9=False, attn_ckpt_path=None,
                flip_axes_list=None, bbox_vol_thresh=0)
        elif variant == 'v9_only':
            rows, per_round = run_variant(
                'v9_only', val_files, device, args.max_obj,
                use_v9=True, attn_ckpt_path=args.v9_ckpt,
                flip_axes_list=None, bbox_vol_thresh=0)
        elif variant == 'tta_only':
            rows, per_round = run_variant(
                'tta_only', val_files, device, args.max_obj,
                use_v9=False, attn_ckpt_path=None,
                flip_axes_list=[(2,)],
                bbox_vol_thresh=args.bbox_vol_thresh)
        elif variant == 'v9_tta':
            rows, per_round = run_variant(
                'v9_tta', val_files, device, args.max_obj,
                use_v9=True, attn_ckpt_path=args.v9_ckpt,
                flip_axes_list=[(2,)],
                bbox_vol_thresh=args.bbox_vol_thresh)
        else:
            print(f"Unknown variant: {variant}", flush=True)
            continue

        all_rows.extend(rows)
        all_per_round[variant] = per_round

    # Final comparison table
    if all_rows:
        df = pd.DataFrame(all_rows)
        print(f"\n{'='*70}", flush=True)
        print("FINAL COMPARISON", flush=True)
        print(f"{'='*70}", flush=True)

        for variant in variants:
            sub = df[df.variant == variant]
            if len(sub) == 0:
                continue
            print(f"\n  {variant:15s} (n={len(sub)})", flush=True)
            print(f"    overall   AUC={sub.DSC_AUC.mean():.4f}  "
                  f"Final={sub.DSC_Final.mean():.4f}  "
                  f"time={sub.time_s.mean():.1f}s", flush=True)
            for has_bb in [True, False]:
                sub2 = sub[sub.has_bbox == has_bb]
                if len(sub2) > 0:
                    tag = 'bbox' if has_bb else 'no-bbox'
                    print(f"    {tag:10s} n={len(sub2):3d}  "
                          f"AUC={sub2.DSC_AUC.mean():.4f}  "
                          f"Final={sub2.DSC_Final.mean():.4f}  "
                          f"time={sub2.time_s.mean():.1f}s", flush=True)

        # Per-case delta table (v9_tta vs baseline)
        if 'baseline' in variants and 'v9_tta' in variants:
            bl = df[df.variant == 'baseline'].set_index('case')
            vt = df[df.variant == 'v9_tta'].set_index('case')
            common = bl.index.intersection(vt.index)
            if len(common) > 0:
                print(f"\n  Per-case delta (v9_tta - baseline), {len(common)} cases:",
                      flush=True)
                deltas = []
                for c in common:
                    d_auc = vt.loc[c, 'DSC_AUC'] - bl.loc[c, 'DSC_AUC']
                    d_fin = vt.loc[c, 'DSC_Final'] - bl.loc[c, 'DSC_Final']
                    deltas.append({'case': c, 'd_AUC': d_auc, 'd_Final': d_fin})
                    sign_auc = '+' if d_auc >= 0 else ''
                    sign_fin = '+' if d_fin >= 0 else ''
                    print(f"    {c[:50]:50s} "
                          f"dAUC={sign_auc}{d_auc:.4f} "
                          f"dFinal={sign_fin}{d_fin:.4f}", flush=True)
                dd = pd.DataFrame(deltas)
                n_better = (dd.d_AUC > 0).sum()
                n_worse = (dd.d_AUC < 0).sum()
                print(f"    => mean dAUC={dd.d_AUC.mean():.4f}, "
                      f"mean dFinal={dd.d_Final.mean():.4f}", flush=True)
                print(f"    => {n_better} improved, {n_worse} regressed, "
                      f"{len(dd) - n_better - n_worse} unchanged", flush=True)

        # Save results
        out_csv = PROJECT_ROOT / "experiments" / "v9_tta_combined_eval.csv"
        os.makedirs(out_csv.parent, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}", flush=True)


if __name__ == '__main__':
    main()
