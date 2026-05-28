#!/usr/bin/env python3
"""
将 Layer 2 微调 checkpoint 转换为 nnInteractive 推理 session 兼容格式。

推理 session 需要的目录结构:
  output_dir/
  ├── dataset.json          (从原始 checkpoint 复制)
  ├── plans.json            (从原始 checkpoint 复制)
  ├── inference_session_class.json  (从原始 checkpoint 复制)
  └── fold_all/
      └── checkpoint_final.pth  (包含 network_weights + init_args + trainer_name)

用法:
  python training/export_checkpoint.py \
    --finetuned experiments/layer2_finetune/checkpoint_epoch0004.pth \
    --original /media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all \
    --output experiments/layer2_finetune/inference_ready
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", required=True, help="微调 checkpoint 路径")
    parser.add_argument("--original", required=True, help="原始 nnInteractive 模型目录")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()

    finetuned_path = Path(args.finetuned)
    original_dir = Path(args.original)
    output_dir = Path(args.output)

    # 加载原始 checkpoint（获取 init_args, trainer_name 等元数据）
    original_ckpt_path = original_dir / "fold_all" / "checkpoint_final.pth"
    print(f"Loading original checkpoint: {original_ckpt_path}")
    original_ckpt = torch.load(original_ckpt_path, map_location='cpu')

    # 加载微调 checkpoint（获取 network_weights）
    print(f"Loading finetuned checkpoint: {finetuned_path}")
    finetuned_ckpt = torch.load(finetuned_path, map_location='cpu')

    # 验证权重 key 一致性
    orig_keys = set(original_ckpt['network_weights'].keys())
    fine_keys = set(finetuned_ckpt['network_weights'].keys())
    if orig_keys != fine_keys:
        missing = orig_keys - fine_keys
        extra = fine_keys - orig_keys
        if missing:
            print(f"WARNING: Missing keys in finetuned: {list(missing)[:5]}...")
        if extra:
            print(f"WARNING: Extra keys in finetuned: {list(extra)[:5]}...")
        print("ABORT: Key mismatch, checkpoint may not be compatible")
        sys.exit(1)
    print(f"  Key check: {len(fine_keys)} keys match ✓")

    # 验证权重 shape 一致性
    shape_mismatch = []
    for key in orig_keys:
        if original_ckpt['network_weights'][key].shape != finetuned_ckpt['network_weights'][key].shape:
            shape_mismatch.append(key)
    if shape_mismatch:
        print(f"ABORT: Shape mismatch at: {shape_mismatch}")
        sys.exit(1)
    print(f"  Shape check: all shapes match ✓")

    # 统计权重变化
    n_changed = 0
    max_diff = 0.0
    for key in orig_keys:
        diff = (original_ckpt['network_weights'][key].float() -
                finetuned_ckpt['network_weights'][key].float()).abs().max().item()
        if diff > 0:
            n_changed += 1
            max_diff = max(max_diff, diff)
    print(f"  Changed params: {n_changed}/{len(orig_keys)} "
          f"(max diff={max_diff:.6f})")

    # 构建推理兼容 checkpoint
    inference_ckpt = {
        'network_weights': finetuned_ckpt['network_weights'],
        'trainer_name': original_ckpt['trainer_name'],
        'init_args': original_ckpt['init_args'],
        'current_epoch': finetuned_ckpt.get('epoch', -1),
        'inference_allowed_mirroring_axes': original_ckpt.get(
            'inference_allowed_mirroring_axes', None),
    }

    # 创建输出目录结构
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = output_dir / "fold_all"
    fold_dir.mkdir(exist_ok=True)

    # 复制配置文件
    for cfg_file in ["dataset.json", "plans.json", "inference_session_class.json"]:
        src = original_dir / cfg_file
        dst = output_dir / cfg_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {cfg_file}")

    # 保存推理 checkpoint
    ckpt_path = fold_dir / "checkpoint_final.pth"
    torch.save(inference_ckpt, ckpt_path)
    print(f"\nSaved inference-ready checkpoint: {ckpt_path}")
    print(f"  Size: {ckpt_path.stat().st_size / 1e6:.1f} MB")
    print(f"\nReady for evaluation:")
    print(f"  conda activate nnInteractive")
    print(f"  python scripts/diagnostic_inference.py --checkpoint {output_dir} ...")


if __name__ == "__main__":
    main()
