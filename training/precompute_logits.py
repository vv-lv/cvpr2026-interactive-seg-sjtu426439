#!/usr/bin/env python3
"""
预计算所有 multi-object case 的 per-object backbone logits。

对每个 case 的每个 object:
  - 从 GT 生成 bbox 交互 → 8ch input → backbone forward (no_grad) → fg logit
  - 保存到磁盘: {case_name}_obj{oid}.pt = fg_logit (D, H, W) float16

之后 resolver 训练直接读这些文件，不再跑 backbone。

用法:
  python -u training/precompute_logits.py --max_files 500 --device cuda:0
  python -u training/precompute_logits.py --max_files 0 --device cuda:0  # 全部
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.trainer import build_network, autocast_ctx
from training.interaction_sim import InteractionManager
from training.dataset import PATCH_SIZE

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "precomputed_logits"


def precompute_one_case(backbone, image_np, gt_np, object_ids, patch_size, device):
    """对一个 case 的所有 object 预计算 logits。

    为了和训练时一致，提取以目标 object 为中心的 192³ patch。

    Returns:
        results: dict {oid: {'fg_logit': tensor, 'gt_patch': array, 'all_gt_patch': array,
                              'patch_slices': tuple}}
    """
    # Image-level z-score
    nonzero = image_np > 0
    if nonzero.sum() > 0:
        m, s = image_np[nonzero].mean(), image_np[nonzero].std()
        if s > 0:
            image_np = (image_np - m) / s
        else:
            image_np = image_np - m

    results = {}

    # 找一个包含最多 object 的 patch 位置
    best_center, best_n = None, 0
    for _ in range(15):
        oid = random.choice(object_ids)
        fg = np.argwhere(gt_np == oid)
        if len(fg) == 0:
            continue
        center = fg[random.randint(0, len(fg) - 1)]
        slices = tuple(
            slice(max(0, center[d] - patch_size[d] // 2),
                  min(image_np.shape[d], center[d] + patch_size[d] // 2))
            for d in range(3)
        )
        n = len(np.unique(gt_np[slices])) - 1
        if n > best_n:
            best_n = n
            best_center = center
        if n >= 3:
            break

    if best_center is None:
        return {}

    center = best_center

    # 提取 patch
    starts, ends = [], []
    for d in range(3):
        half = patch_size[d] // 2
        s = max(0, min(center[d] - half, image_np.shape[d] - patch_size[d]))
        e = s + patch_size[d]
        if e > image_np.shape[d]:
            e = image_np.shape[d]
            s = max(0, e - patch_size[d])
        starts.append(s)
        ends.append(e)

    slices = tuple(slice(s, e) for s, e in zip(starts, ends))
    img_patch = image_np[slices].copy()
    gt_patch = gt_np[slices].copy()

    # Pad if needed
    pad = [(0, max(0, patch_size[d] - img_patch.shape[d])) for d in range(3)]
    if any(p[1] > 0 for p in pad):
        img_patch = np.pad(img_patch, pad, mode='constant')
        gt_patch = np.pad(gt_patch, pad, mode='constant')

    # Patch 内有哪些 object
    patch_oids = np.unique(gt_patch)
    patch_oids = [int(x) for x in patch_oids if x > 0]
    if len(patch_oids) < 2:
        return {}

    # 对每个 object 跑 backbone
    image_tensor = torch.from_numpy(img_patch[np.newaxis, np.newaxis]).to(device)  # (1,1,D,H,W)

    for oid in patch_oids:
        gt_binary = (gt_patch == oid).astype(np.uint8)
        mgr = InteractionManager(patch_size)
        mgr.set_initial_bbox(gt_binary, jitter=0.05)
        inter = torch.from_numpy(mgr.get_numpy()).unsqueeze(0).to(device)  # (1,7,D,H,W)
        input_8ch = torch.cat([image_tensor, inter], dim=1)

        with torch.no_grad(), autocast_ctx():
            output = backbone(input_8ch)  # (1,2,D,H,W)
        fg_logit = (output[0, 1] - output[0, 0]).cpu().half()  # (D,H,W) float16

        results[oid] = fg_logit
        del input_8ch, inter, output

    return {'logits': results, 'gt_patch': gt_patch, 'patch_oids': patch_oids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 backbone
    backbone, _ = build_network(args.checkpoint, deep_supervision=False)
    backbone = backbone.to(args.device).eval()

    # 加载 multi-object files
    cache_path = Path(args.data_dir).parent / "object_count_cache.json"
    with open(cache_path) as fp:
        cache = json.load(fp)

    all_files = sorted(Path(args.data_dir).rglob("*.npz"))
    multi_files = [f for f in all_files if cache.get(f.name, 0) >= 2]
    if args.max_files > 0:
        random.seed(42)
        random.shuffle(multi_files)
        multi_files = multi_files[:args.max_files]

    print(f"Precomputing logits for {len(multi_files)} multi-object files...")

    t0 = time.time()
    n_saved = 0
    n_skipped = 0

    for i, f in enumerate(multi_files):
        data = np.load(f, allow_pickle=True)
        image = data['imgs'].astype(np.float32)
        gt = data['gts'].astype(np.uint8)
        object_ids = [int(x) for x in np.unique(gt) if x > 0]

        result = precompute_one_case(
            backbone, image, gt, object_ids, PATCH_SIZE, args.device)

        if not result:
            n_skipped += 1
            continue

        # 保存
        save_path = output_dir / f"{f.stem}.pt"
        torch.save({
            'logits': result['logits'],           # {oid: (D,H,W) fp16}
            'gt_patch': result['gt_patch'],       # (D,H,W) uint8
            'patch_oids': result['patch_oids'],   # list of int
        }, save_path)
        n_saved += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(multi_files) - i - 1) / rate
            print(f"  [{i+1}/{len(multi_files)}] saved={n_saved}, skipped={n_skipped}, "
                  f"{rate:.1f} files/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {n_saved} saved, {n_skipped} skipped, {elapsed:.0f}s total")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
