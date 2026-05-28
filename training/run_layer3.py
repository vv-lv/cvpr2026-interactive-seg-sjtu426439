#!/usr/bin/env python3
"""
Layer 3 训练入口：基于模型预测的多轮交互训练。

核心改进（vs Layer 2）:
- Follow-up 交互基于模型实际预测的错误（不是 GT 腐蚀/膨胀）
- 多轮 sequential backward（每轮独立 backward，梯度累积）
- Follow-up 概率 curriculum（0.3→0.75）
- 论文扩展增强（Transpose, Intensity inversion, Scaling [0.5,2]）
- 连通域分析 + EDT center-biased 采样

用法:
  # 冻结验证
  python training/run_layer3.py --mode frozen --max_files 50 --n_batches 20

  # 训练（冻结 encoder）
  python training/run_layer3.py --mode train --max_files 500 --epochs 20 --lr 1e-4 --freeze_encoder

  # 训练（全部解冻）
  python training/run_layer3.py --mode train --max_files 500 --epochs 20 --lr 1e-5
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# numpy compat
import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.dataset import Layer3Dataset
from training.interaction_sim import InteractionManager
from training.trainer import (build_network, build_loss, downsample_target_for_ds,
                               PolyLRScheduler, autocast_ctx)

# ── 默认路径 ──
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "layer3_finetune"


class Layer3Trainer:
    """多轮交互训练器。

    每个训练步骤：
    Round 0: 从 GT 生成初始 bbox → forward → loss → backward（释放激活）
    Round 1 (概率 p): 取 Round 0 预测 → 对比 GT → 生成 follow-up → forward → loss → backward
    → optimizer step（梯度从两轮累积）
    """

    def __init__(self, checkpoint_path, output_dir, device='cuda:0',
                 lr=1e-4, max_epochs=100, freeze_encoder=False,
                 followup_prob_start=0.3, followup_prob_end=0.75,
                 max_followup_rounds=5, no_bbox_prob=0.2):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.max_epochs = max_epochs
        self.followup_prob_start = followup_prob_start
        self.followup_prob_end = followup_prob_end
        self.max_followup_rounds = max_followup_rounds
        self.no_bbox_prob = no_bbox_prob

        # 网络
        self.network, self.plans = build_network(checkpoint_path, deep_supervision=True)
        self.network = self.network.to(self.device)

        if freeze_encoder:
            for name, param in self.network.named_parameters():
                if name.startswith('encoder'):
                    param.requires_grad = False
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.network.parameters())
            print(f"Encoder frozen: {trainable:,}/{total:,} params trainable")

        # Loss / optimizer / AMP
        self.loss_fn = build_loss(deep_supervision=True)
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.99,
                                          weight_decay=3e-5, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(self.optimizer, lr, max_epochs, exponent=0.9)
        self.grad_scaler = GradScaler()

    def _get_followup_prob(self, epoch):
        """线性 curriculum: start → end over max_epochs."""
        frac = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        return self.followup_prob_start + frac * (self.followup_prob_end - self.followup_prob_start)

    def _compute_loss(self, outputs, target):
        """计算深度监督 loss。"""
        ds_targets = downsample_target_for_ds(target, num_levels=len(outputs))
        return self.loss_fn(outputs, ds_targets)

    def _train_step(self, batch, epoch):
        """多轮训练步骤（sequential backward, 匹配推理协议）。

        bbox case: Round 0 = bbox, Round 1..K = follow-up clicks
        no-bbox case: Round 0 = initial fg click, Round 1..K = follow-up clicks
        """
        image = batch['image'].to(self.device, non_blocking=True)   # (B, 1, D, H, W)
        target = batch['target'].to(self.device, non_blocking=True)  # (B, 1, D, H, W)
        B = image.shape[0]
        spatial_shape = image.shape[2:]

        # Decide: bbox or no-bbox
        use_bbox = random.random() >= self.no_bbox_prob

        # --- 生成初始交互 ---
        interactions_list = []
        for b in range(B):
            gt_np = target[b, 0].cpu().numpy().astype(np.uint8)
            mgr = InteractionManager(spatial_shape)
            if use_bbox:
                mgr.set_initial_bbox(gt_np, jitter=0.05)
            else:
                mgr.add_followup(np.zeros_like(gt_np), gt_np)
            interactions_list.append(mgr)

        # Determine number of follow-up rounds
        followup_prob = self._get_followup_prob(epoch)
        if self.max_followup_rounds > 0 and random.random() < followup_prob:
            n_followups = random.randint(1, self.max_followup_rounds)
        else:
            n_followups = 0
        n_total_rounds = 1 + n_followups

        # === Round 0 ===
        inter_np = np.stack([m.get_numpy() for m in interactions_list])
        interactions = torch.from_numpy(inter_np).to(self.device)
        input_tensor = torch.cat([image, interactions], dim=1)

        with autocast_ctx():
            outputs = self.network(input_tensor)
            loss_r0 = self._compute_loss(outputs, target)

        self.grad_scaler.scale(loss_r0 / n_total_rounds).backward()
        loss_total = loss_r0.item()

        # === Follow-up rounds ===
        for k in range(n_followups):
            with torch.no_grad():
                pred = outputs[0].argmax(1)

            del outputs, input_tensor
            torch.cuda.empty_cache()

            for b in range(B):
                pred_np = pred[b].cpu().numpy().astype(np.uint8)
                gt_np = target[b, 0].cpu().numpy().astype(np.uint8)
                interactions_list[b].add_followup(pred_np, gt_np)

            inter_np = np.stack([m.get_numpy() for m in interactions_list])
            interactions = torch.from_numpy(inter_np).to(self.device)
            input_tensor = torch.cat([image, interactions], dim=1)

            with autocast_ctx():
                outputs = self.network(input_tensor)
                loss_rk = self._compute_loss(outputs, target)

            self.grad_scaler.scale(loss_rk / n_total_rounds).backward()
            loss_total += loss_rk.item()

        del outputs, input_tensor

        # === Optimizer step ===
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.network.parameters() if p.requires_grad], max_norm=12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        return loss_total / n_total_rounds, n_total_rounds

    def frozen_validation(self, dataloader, n_batches=10):
        """冻结验证（验证多轮 pipeline 正确性）。"""
        print("\n" + "=" * 60)
        print("FROZEN VALIDATION (Layer 3, multi-round)")
        print("=" * 60)

        self.network.eval()
        losses_r0, losses_r1 = [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break

                image = batch['image'].to(self.device)
                target = batch['target'].to(self.device)
                B = image.shape[0]
                spatial_shape = image.shape[2:]

                # Round 0
                interactions_list = []
                for b in range(B):
                    gt_np = target[b, 0].cpu().numpy().astype(np.uint8)
                    mgr = InteractionManager(spatial_shape)
                    mgr.set_initial_bbox(gt_np, jitter=0.05)
                    interactions_list.append(mgr)

                inter_np = np.stack([m.get_numpy() for m in interactions_list])
                interactions = torch.from_numpy(inter_np).to(self.device)
                input_r0 = torch.cat([image, interactions], dim=1)

                with autocast_ctx():
                    output_r0 = self.network(input_r0)
                    loss_r0 = self._compute_loss(output_r0, target)
                losses_r0.append(loss_r0.item())

                # Round 1: follow-up from real prediction
                pred_r0 = output_r0[0].argmax(1)
                for b in range(B):
                    pred_np = pred_r0[b].cpu().numpy().astype(np.uint8)
                    gt_np = target[b, 0].cpu().numpy().astype(np.uint8)
                    interactions_list[b].add_followup(pred_np, gt_np)

                inter_np = np.stack([m.get_numpy() for m in interactions_list])
                interactions = torch.from_numpy(inter_np).to(self.device)
                input_r1 = torch.cat([image, interactions], dim=1)

                with autocast_ctx():
                    output_r1 = self.network(input_r1)
                    loss_r1 = self._compute_loss(output_r1, target)
                losses_r1.append(loss_r1.item())

                pred_r1 = output_r1[0].argmax(1)
                gt_fg = target.mean().item()
                print(f"  Batch {i}: R0 loss={loss_r0.item():.4f}, "
                      f"R1 loss={loss_r1.item():.4f}, "
                      f"gt_fg={gt_fg:.3f}, name={batch['name'][0]}")

        mean_r0 = np.mean(losses_r0)
        mean_r1 = np.mean(losses_r1)
        print(f"\nFrozen validation:")
        print(f"  Round 0 loss: {mean_r0:.4f} ± {np.std(losses_r0):.4f}")
        print(f"  Round 1 loss: {mean_r1:.4f} ± {np.std(losses_r1):.4f}")
        print(f"  R1 vs R0: {'improved' if mean_r1 < mean_r0 else 'same/worse'} "
              f"(Δ={mean_r1 - mean_r0:+.4f})")

    def train(self, train_loader, num_epochs=None, save_every=10):
        if num_epochs is None:
            num_epochs = self.max_epochs

        print(f"\n{'=' * 60}")
        print(f"TRAINING Layer 3: {num_epochs} epochs, lr={self.lr}, "
              f"followup {self.followup_prob_start:.1f}→{self.followup_prob_end:.1f}")
        print(f"{'=' * 60}")

        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_rounds = []
            t0 = time.time()
            fp = self._get_followup_prob(epoch)

            for batch_idx, batch in enumerate(train_loader):
                avg_loss, n_rounds = self._train_step(batch, epoch)
                epoch_losses.append(avg_loss)
                epoch_rounds.append(n_rounds)

            self.lr_scheduler.step(epoch)
            mean_loss = np.mean(epoch_losses)
            mean_rounds = np.mean(epoch_rounds)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:03d}: loss={mean_loss:.4f}, "
                  f"rounds={mean_rounds:.2f}, fp={fp:.2f}, "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"time={elapsed:.1f}s")

            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                path = self.output_dir / f"checkpoint_epoch{epoch:04d}.pth"
                torch.save({
                    'epoch': epoch,
                    'network_weights': self.network.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'loss': mean_loss,
                }, path)
                print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Layer 3: Model-prediction-based multi-round training")
    parser.add_argument("--mode", choices=["frozen", "train"], default="frozen")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--fp_start", type=float, default=0.3,
                        help="Follow-up probability start")
    parser.add_argument("--fp_end", type=float, default=0.75,
                        help="Follow-up probability end")
    parser.add_argument("--max_followup_rounds", type=int, default=5,
                        help="Max follow-up click rounds")
    parser.add_argument("--no_bbox_prob", type=float, default=0.2,
                        help="Prob of no-bbox training")
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Max files: {args.max_files}")

    dataset = Layer3Dataset(
        data_dir=args.data_dir,
        max_files=args.max_files,
        augment=(args.mode == "train"),
    )
    print(f"Dataset: {len(dataset)} files")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainer = Layer3Trainer(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        lr=args.lr,
        max_epochs=args.epochs,
        freeze_encoder=args.freeze_encoder,
        followup_prob_start=args.fp_start,
        followup_prob_end=args.fp_end,
        max_followup_rounds=args.max_followup_rounds,
        no_bbox_prob=args.no_bbox_prob,
    )

    if args.mode == "frozen":
        trainer.frozen_validation(dataloader, n_batches=args.n_batches)
    elif args.mode == "train":
        print("\n--- Pre-training frozen check (3 batches) ---")
        trainer.frozen_validation(dataloader, n_batches=3)
        print("\n--- Starting training ---")
        trainer.train(dataloader, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
