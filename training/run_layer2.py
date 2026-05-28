#!/usr/bin/env python3
"""
Layer 2 训练入口脚本。

使用方式:
  # 1. 冻结验证（不改变任何权重，验证 pipeline 正确性）
  python training/run_layer2.py --mode frozen --max_files 50 --n_batches 20

  # 2. 小规模微调测试
  python training/run_layer2.py --mode train --max_files 100 --epochs 5 --lr 1e-4

  # 3. 冻结 encoder 微调（只训 decoder）
  python training/run_layer2.py --mode train --freeze_encoder --max_files 100 --epochs 10
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import NNInteractiveTrainDataset
from training.trainer import Layer2Trainer

# ── 默认路径 ──
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "layer2_finetune"


def main():
    parser = argparse.ArgumentParser(description="Layer 2: Minimal viable finetuning")
    parser.add_argument("--mode", choices=["frozen", "train"], default="frozen",
                        help="frozen=验证pipeline, train=微调")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")

    # 数据
    parser.add_argument("--max_files", type=int, default=50,
                        help="加载的最大 NPZ 文件数（0=全部）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size（192³ patch 在 3090 上 bs=1 最安全）")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_followup", action="store_true",
                        help="不生成 follow-up click（仅 bbox initial prompt）")
    parser.add_argument("--no_augment", action="store_true",
                        help="禁用数据增强（仅用于调试）")

    # 冻结验证
    parser.add_argument("--n_batches", type=int, default=20,
                        help="冻结验证的 batch 数")

    # 训练
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率（远小于原始 0.01，避免破坏权重）")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="冻结 encoder，只训 decoder")

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_dir}")
    print(f"Max files: {args.max_files}")
    print(f"Device: {args.device}")

    # ── 数据集 ──
    augment = (args.mode == "train") and (not args.no_augment)
    dataset = NNInteractiveTrainDataset(
        data_dir=args.data_dir,
        max_files=args.max_files,
        include_followup=not args.no_followup,
        augment=augment,
    )
    print(f"Dataset: {len(dataset)} files")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Trainer ──
    trainer = Layer2Trainer(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        lr=args.lr,
        max_epochs=args.epochs,
        freeze_encoder=args.freeze_encoder,
        grad_accumulation_steps=args.grad_accum,
    )

    # ── 执行 ──
    if args.mode == "frozen":
        losses = trainer.frozen_validation(dataloader, n_batches=args.n_batches)
    elif args.mode == "train":
        # 先跑一轮冻结验证确认 pipeline 正常
        print("\n--- Pre-training frozen check (3 batches) ---")
        trainer.frozen_validation(dataloader, n_batches=3)
        print("\n--- Starting training ---")
        trainer.train(dataloader, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
