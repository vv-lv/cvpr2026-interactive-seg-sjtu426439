#!/usr/bin/env python3
"""
Memory Attention 训练：多 object 顺序处理，memory bank 累积。

训练流程:
  For each patch (含 K 个 objects):
    memory_bank = empty (or from previous round)
    For object k = 1..K:
      1. 生成 interactions (bbox) → 8ch input
      2. encoder (frozen, no_grad) → skips
      3. memory_attention(bottleneck, memory_bank) → enhanced bottleneck
      4. decoder → prediction → loss → backward (sequential)
      5. 更新 memory_bank: add pooled bottleneck feature

用法:
  # Smoke test
  python -u training/run_memory.py --mode frozen --max_files 50 --n_batches 10

  # Training
  python -u training/run_memory.py --mode train --max_files 200 --epochs 20
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.interaction_sim import InteractionManager
from training.memory_module import MemoryAttention, MemoryEnhancedUNet, MemoryBank
from training.run_resolver import MultiObjectDataset

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "memory_attn"

MAX_OBJECTS = 4


class MemoryTrainer:

    def __init__(self, checkpoint_path, output_dir, device='cuda:0',
                 lr=1e-3, decoder_lr=1e-4, max_epochs=20,
                 l2sp_alpha=0.01, followup_start=0.3, followup_end=0.75,
                 no_memory=False, in_only=False):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.l2sp_alpha = l2sp_alpha
        self.followup_start = followup_start
        self.followup_end = followup_end

        # 加载原始网络
        base_network, _ = build_network(checkpoint_path, deep_supervision=True)

        # 保存 decoder 预训练权重用于 L2-SP
        self.pretrained_decoder_weights = {
            name: param.detach().clone().to(device)
            for name, param in base_network.decoder.named_parameters()
        }

        # 冻结 encoder
        for p in base_network.encoder.parameters():
            p.requires_grad = False

        # Memory attention
        memory_attn = MemoryAttention(dim=320, n_heads=8, n_layers=1)

        # 消融模式: 冻结 memory attention → 纯 decoder 微调
        if no_memory or in_only:
            for p in memory_attn.parameters():
                p.requires_grad = False
            if not in_only:
                print(">>> ABLATION: memory attention FROZEN (pure decoder finetuning)")

        # 包装网络
        self.model = MemoryEnhancedUNet(base_network, memory_attn).to(self.device)

        # IN-only 模式: 冻结 decoder 全部参数，只解冻 InstanceNorm affine
        self.in_only = in_only
        if in_only:
            # 先冻结 decoder 全部
            for p in self.model.decoder.parameters():
                p.requires_grad = False
            # 再解冻 InstanceNorm 的 weight 和 bias（encoder + decoder 都解冻）
            in_params = []
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.InstanceNorm3d) and module.affine:
                    for pname, param in module.named_parameters():
                        param.requires_grad = True
                        in_params.append(param)
            n_in = sum(p.numel() for p in in_params)
            print(f">>> IN-ONLY: {n_in:,} InstanceNorm params trainable, all else frozen")

        # 参数统计
        mem_params = sum(p.numel() for p in self.model.memory_attention.parameters() if p.requires_grad)
        dec_params = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
        enc_params = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Memory attention: {mem_params:,} trainable params (lr={lr})")
        print(f"Encoder: {enc_params:,} trainable params")
        print(f"Decoder: {dec_params:,} trainable params (lr={decoder_lr})")
        print(f"Total trainable: {total_trainable:,}")

        # Loss
        self.loss_fn = build_loss(deep_supervision=True)

        # Optimizer
        param_groups = []
        if in_only:
            # IN-only: 所有 IN params 用同一个 lr
            in_params_list = [p for p in self.model.parameters() if p.requires_grad]
            param_groups.append({'params': in_params_list, 'lr': decoder_lr})
        else:
            if not no_memory:
                param_groups.append({'params': list(self.model.memory_attention.parameters()), 'lr': lr})
            param_groups.append({'params': [p for p in self.model.decoder.parameters() if p.requires_grad], 'lr': decoder_lr})
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()

    def _compute_l2sp(self):
        """L2-SP 正则: 惩罚 decoder 偏离预训练权重。"""
        if self.in_only:
            return torch.tensor(0.0, device=self.device)  # IN-only 参数太少，不需要 L2-SP
        reg = 0.0
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad and name in self.pretrained_decoder_weights:
                reg = reg + ((param - self.pretrained_decoder_weights[name]) ** 2).sum()
        return self.l2sp_alpha * reg

    def _get_followup_prob(self, epoch):
        """Follow-up 概率 curriculum: start → end 线性增长。"""
        frac = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        return self.followup_start + frac * (self.followup_end - self.followup_start)

    def _forward_one_round(self, image, gt_binary, mgr, bank):
        """一轮 forward: 构造 input → forward → loss + pred。

        Returns: (loss, pred_numpy, bottleneck_pooled)
        """
        inter = torch.from_numpy(mgr.get_numpy()).unsqueeze(0).to(self.device)
        input_8ch = torch.cat([image, inter], dim=1)
        self.model.set_memory_bank(bank.get_tensor())

        with autocast_ctx():
            outputs = self.model(input_8ch)
            target = torch.from_numpy(
                gt_binary[np.newaxis, np.newaxis].astype(np.float32)).to(self.device)
            ds_targets = downsample_target_for_ds(target, num_levels=len(outputs))
            loss = self.loss_fn(outputs, ds_targets)

        bottleneck_pooled = self.model.get_last_bottleneck_pooled()
        with torch.no_grad():
            pred = outputs[0].argmax(1)[0].cpu().numpy().astype(np.uint8)

        del input_8ch, inter, outputs, target, ds_targets
        return loss, pred, bottleneck_pooled

    def _train_step(self, batch, epoch):
        """一个训练步骤：单 object 多轮交互 + memory bank 跨轮累积 + L2-SP。"""
        image = batch['image'].to(self.device)
        if image.dim() == 4:
            image = image.unsqueeze(0)
        gt_np = batch['gt'].numpy()
        oids = batch['object_ids']
        if isinstance(oids, torch.Tensor):
            oids = oids.tolist()

        if len(oids) < 1 or gt_np.ndim != 3:
            return None
        if gt_np.shape != tuple(image.shape[2:]):
            return None

        # 随机选一个 object 做多轮交互
        oid = random.choice(oids)
        gt_binary = (gt_np == oid).astype(np.uint8)
        if gt_binary.sum() == 0:
            return None

        spatial = tuple(image.shape[2:])
        self.optimizer.zero_grad(set_to_none=True)
        bank = MemoryBank(max_size=24)
        total_loss = 0.0
        total_dice = 0.0
        n_rounds = 0

        # === Round 0: bbox initial prompt ===
        mgr = InteractionManager(spatial)
        mgr.set_initial_bbox(gt_binary, jitter=0.05)

        loss_r0, pred_r0, bp_r0 = self._forward_one_round(image, gt_binary, mgr, bank)
        self.scaler.scale(loss_r0).backward()
        total_loss += loss_r0.item()
        bank.add(bp_r0)
        n_rounds += 1

        gt_fg = gt_binary.sum()
        total_dice += 2 * (gt_binary & pred_r0).sum() / max(gt_fg + pred_r0.sum(), 1)
        torch.cuda.empty_cache()

        # === Round 1+: follow-up based on model prediction (Layer 3 机制) ===
        followup_prob = self._get_followup_prob(epoch)
        if random.random() < followup_prob:
            # 基于模型真实预测更新交互
            mgr.add_followup(pred_r0, gt_binary)

            loss_r1, pred_r1, bp_r1 = self._forward_one_round(image, gt_binary, mgr, bank)
            self.scaler.scale(loss_r1).backward()
            total_loss += loss_r1.item()
            bank.add(bp_r1)
            n_rounds += 1

            total_dice += 2 * (gt_binary & pred_r1).sum() / max(gt_fg + pred_r1.sum(), 1)
            torch.cuda.empty_cache()

        # === L2-SP 正则化 ===
        l2sp = self._compute_l2sp()
        if l2sp.requires_grad:
            self.scaler.scale(l2sp).backward()

        # === Optimizer step ===
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=12)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        avg_loss = total_loss / n_rounds
        avg_dice = total_dice / n_rounds
        return avg_loss, avg_dice, n_rounds

    def frozen_validation(self, dataloader, n_batches=10):
        """冻结验证：确认 pipeline 正确。"""
        print("\n" + "=" * 60)
        print("FROZEN VALIDATION (Memory Attention)")
        print("=" * 60)
        self.model.eval()
        losses, dices = [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                image = batch['image'].to(self.device)
                if image.dim() == 4:
                    image = image.unsqueeze(0)
                gt_np = batch['gt'].numpy()
                oids = batch['object_ids']
                if isinstance(oids, torch.Tensor):
                    oids = oids.tolist()
                if len(oids) < 2 or gt_np.ndim != 3:
                    print(f"  Batch {i}: skipped")
                    continue
                if gt_np.shape != tuple(image.shape[2:]):
                    continue

                K = min(len(oids), MAX_OBJECTS)
                bank = MemoryBank()
                batch_loss, batch_dice = 0, 0

                for oid in oids[:K]:
                    gt_binary = (gt_np == oid).astype(np.uint8)
                    if gt_binary.sum() == 0:
                        continue
                    mgr_v = InteractionManager(tuple(image.shape[2:]))
                    if gt_binary.sum() > 0:
                        mgr_v.set_initial_bbox(gt_binary, jitter=0.05)
                    loss, pred, bp = self._forward_one_round(
                        image, gt_binary, mgr_v, bank)
                    bank.add(bp)
                    batch_loss += loss.item()
                    gt_fg = gt_binary.sum()
                    batch_dice += 2*(gt_binary & pred).sum() / max(gt_fg + pred.sum(), 1)
                    torch.cuda.empty_cache()

                avg_loss = batch_loss / K
                avg_dice = batch_dice / K
                losses.append(avg_loss)
                dices.append(avg_dice)
                print(f"  Batch {i}: loss={avg_loss:.4f}, dice={avg_dice:.3f}, "
                      f"K={K}, bank_size={len(bank)}, name={batch['name']}")

        if losses:
            print(f"\nFrozen: loss={np.mean(losses):.4f}, dice={np.mean(dices):.3f}")

    def train(self, train_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.max_epochs

        print(f"\n{'=' * 60}")
        print(f"TRAINING Memory Attention: {num_epochs} epochs")
        print(f"{'=' * 60}")

        best_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            # 冻结 encoder (确保不被意外解冻)
            self.model.encoder.eval()
            for p in self.model.encoder.parameters():
                p.requires_grad = False

            epoch_losses, epoch_dices, n_steps = [], [], 0
            t0 = time.time()

            epoch_rounds = []
            for batch in train_loader:
                ret = self._train_step(batch, epoch)
                if ret is None:
                    continue
                loss, dice, n_rounds = ret
                epoch_losses.append(loss)
                epoch_dices.append(dice)
                epoch_rounds.append(n_rounds)
                n_steps += 1

            elapsed = time.time() - t0
            ml = np.mean(epoch_losses) if epoch_losses else float('nan')
            md = np.mean(epoch_dices) if epoch_dices else float('nan')
            mr = np.mean(epoch_rounds) if epoch_rounds else 0
            fp = self._get_followup_prob(epoch)
            print(f"Epoch {epoch:02d}: loss={ml:.4f}, dice={md:.3f}, "
                  f"rounds={mr:.2f}, fp={fp:.2f}, "
                  f"steps={n_steps}, time={elapsed:.1f}s")

            if ml < best_loss:
                best_loss = ml
                torch.save({
                    'epoch': epoch,
                    'memory_attn_state': self.model.memory_attention.state_dict(),
                    'decoder_state': self.model.decoder.state_dict(),
                    'loss': ml, 'dice': md,
                }, self.output_dir / "best.pth")

        torch.save({
            'epoch': num_epochs - 1,
            'memory_attn_state': self.model.memory_attention.state_dict(),
            'decoder_state': self.model.decoder.state_dict(),
        }, self.output_dir / "final.pth")
        print(f"Saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "train"], default="frozen")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Memory attention LR")
    parser.add_argument("--decoder_lr", type=float, default=1e-4,
                        help="Decoder LR (10x smaller, with L2-SP)")
    parser.add_argument("--l2sp_alpha", type=float, default=0.01,
                        help="L2-SP 正则化系数")
    parser.add_argument("--fp_start", type=float, default=0.3,
                        help="Follow-up 概率起始值")
    parser.add_argument("--fp_end", type=float, default=0.75,
                        help="Follow-up 概率终止值")
    parser.add_argument("--no_memory", action="store_true",
                        help="消融实验: 冻结 memory attention, 纯 decoder 微调")
    parser.add_argument("--in_only", action="store_true",
                        help="IN-only 适配: 只训 InstanceNorm affine 参数 (CLoPA 风格)")
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    dataset = MultiObjectDataset(
        data_dir=args.data_dir, max_files=args.max_files,
        augment=(args.mode == "train"))
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])

    trainer = MemoryTrainer(
        checkpoint_path=args.checkpoint, output_dir=args.output_dir,
        device=args.device, lr=args.lr, decoder_lr=args.decoder_lr,
        max_epochs=args.epochs, l2sp_alpha=args.l2sp_alpha,
        followup_start=args.fp_start, followup_end=args.fp_end,
        no_memory=args.no_memory, in_only=args.in_only)

    if args.mode == "frozen":
        trainer.frozen_validation(loader, n_batches=args.n_batches)
    elif args.mode == "train":
        print("\n--- Pre-training check ---")
        trainer.frozen_validation(loader, n_batches=3)
        print("\n--- Training ---")
        trainer.train(loader, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
