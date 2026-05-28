#!/usr/bin/env python3
"""
Knowledge Distillation: train a smaller student network guided by the
original nnInteractive teacher.

Inspired by Fast-nnUNet: student has the same architecture with reduced
channels (features_per_stage divided by r).

Loss = α × KD_loss(student_logit, teacher_logit, T) + (1-α) × task_loss(student, GT)

Usage:
  # Student -r 2 (channels halved, 25.6M params, ~88ms forward)
  python training/run_distill.py --reduction 2 --epochs 200 --lr 0.01 --batch_size 2

  # Student -r 4 (channels quartered, 6.4M params, ~55ms forward)
  python training/run_distill.py --reduction 4 --epochs 200 --lr 0.01 --batch_size 3
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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# numpy compat
try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.dataset import Layer3Dataset
from training.interaction_sim import InteractionManager
from training.trainer import (build_network, build_loss, downsample_target_for_ds,
                               PolyLRScheduler, autocast_ctx, resolve_class)

# ── 默认路径 ──
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "distill"


# ═══════════════════════════════════════════════════════════════════════════════
# Student network builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_student_network(checkpoint_path: str, reduction: int = 2,
                          deep_supervision: bool = True):
    """Build a smaller student network with channels divided by reduction factor.

    Uses the same architecture class and hyperparameters as the teacher,
    only features_per_stage is scaled down.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    plans = ckpt['init_args']['plans']
    configuration = ckpt['init_args']['configuration']

    configs = plans['configurations']
    config = {}
    cfg_name = configuration
    while cfg_name:
        cfg = configs[cfg_name]
        for k, v in cfg.items():
            if k not in config and k != 'inherits_from':
                config[k] = v
        cfg_name = cfg.get('inherits_from', None)

    arch_info = config['architecture']
    arch_kwargs = dict(arch_info['arch_kwargs'])
    for key in arch_info.get('_kw_requires_import', []):
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = resolve_class(arch_kwargs[key])

    # Scale down features
    orig_features = arch_kwargs['features_per_stage']
    student_features = [max(8, f // reduction) for f in orig_features]
    arch_kwargs['features_per_stage'] = student_features

    network_class = resolve_class(arch_info['network_class_name'])
    student = network_class(
        input_channels=8,
        num_classes=2,
        deep_supervision=deep_supervision,
        **arch_kwargs
    )

    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student network: features={student_features}, params={n_params:.1f}M")
    print(f"  (Teacher features={orig_features}, reduction={reduction})")

    return student, plans


# ═══════════════════════════════════════════════════════════════════════════════
# KD Loss
# ═══════════════════════════════════════════════════════════════════════════════

def kd_loss_single(student_logits, teacher_logits, temperature=3.0):
    """KL divergence loss on a single pair of logit tensors.

    Args:
        student_logits: (B, C, ...) raw logits from student
        teacher_logits: (B, C, ...) raw logits from teacher (detached)
        temperature: softening temperature

    Returns:
        KL divergence loss (scaled by T², averaged over all elements)
    """
    B, C = student_logits.shape[:2]
    spatial = student_logits.shape[2:]
    student_flat = student_logits.permute(0, *range(2, 2+len(spatial)), 1).reshape(-1, C)
    teacher_flat = teacher_logits.permute(0, *range(2, 2+len(spatial)), 1).reshape(-1, C)

    student_soft = F.log_softmax(student_flat / temperature, dim=1)
    teacher_soft = F.softmax(teacher_flat / temperature, dim=1)
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    return loss


def kd_loss_deep_supervision(student_outputs, teacher_outputs, temperature=3.0,
                              num_ds_levels=5):
    """KD loss with deep supervision: KL divergence at each DS level.

    Weights match task loss DS weights: [1, 1/2, 1/4, 1/8, 1/16] normalized.

    Args:
        student_outputs: list of tensors from student (5 DS levels)
        teacher_outputs: list of tensors from teacher (5 DS levels)
        temperature: softening temperature

    Returns:
        Weighted sum of per-level KD losses
    """
    n_levels = min(len(student_outputs), len(teacher_outputs), num_ds_levels)
    weights = np.array([1 / (2 ** i) for i in range(n_levels)])
    weights = weights / weights.sum()

    total_loss = 0.0
    for i in range(n_levels):
        s_logits = student_outputs[i]
        t_logits = teacher_outputs[i].detach()
        # Resize teacher to match student if shapes differ
        if s_logits.shape != t_logits.shape:
            t_logits = F.interpolate(t_logits, size=s_logits.shape[2:],
                                     mode='trilinear', align_corners=False)
        total_loss = total_loss + weights[i] * kd_loss_single(s_logits, t_logits, temperature)

    return total_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Distillation Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class DistillTrainer:
    """Knowledge distillation trainer.

    Teacher: original nnInteractive (frozen, eval mode)
    Student: channel-reduced version (trainable)
    Loss: α × KD + (1-α) × task_loss
    """

    def __init__(self, checkpoint_path, output_dir, device='cuda:0',
                 reduction=2, lr=0.01, max_epochs=200,
                 kd_alpha=0.3, kd_temperature=3.0,
                 grad_accumulation_steps=1,
                 followup_prob_start=0.3, followup_prob_end=0.75,
                 max_followup_rounds=5, no_bbox_prob=0.2):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.max_epochs = max_epochs
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.grad_accumulation_steps = grad_accumulation_steps
        self.followup_prob_start = followup_prob_start
        self.followup_prob_end = followup_prob_end
        self.max_followup_rounds = max_followup_rounds
        self.no_bbox_prob = no_bbox_prob

        # Teacher (frozen, deep_supervision=True for multi-level KD)
        if self.kd_alpha > 0:
            print("Loading teacher...")
            self.teacher, _ = build_network(checkpoint_path, deep_supervision=True)
            self.teacher = self.teacher.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            t_params = sum(p.numel() for p in self.teacher.parameters()) / 1e6
            print(f"Teacher: {t_params:.1f}M params (frozen, deep_supervision=True)")
        else:
            self.teacher = None
            print("KD alpha=0: skipping teacher (pure GT supervision)")

        # Student
        print("Building student...")
        self.student, self.plans = build_student_network(
            checkpoint_path, reduction=reduction, deep_supervision=True)
        self.student = self.student.to(self.device)

        # Task loss (same as nnInteractive)
        self.task_loss_fn = build_loss(deep_supervision=True)

        # Optimizer (paper: SGD lr=0.01 for training from scratch)
        self.optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=lr, momentum=0.99, weight_decay=3e-5, nesterov=True)
        self.lr_scheduler = PolyLRScheduler(
            self.optimizer, lr, max_epochs, exponent=0.9)
        self.grad_scaler = GradScaler()

    def _get_followup_prob(self, epoch):
        frac = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        return self.followup_prob_start + frac * (self.followup_prob_end - self.followup_prob_start)

    def _get_teacher_click_prob(self, epoch):
        """Curriculum for click source: early=teacher, late=student.

        Starts at 1.0 (always teacher clicks) and linearly decreases to 0.0
        (always student clicks) over training.
        """
        frac = min(epoch / max(self.max_epochs - 1, 1), 1.0)
        return 1.0 - frac  # 1.0 → 0.0

    def _forward_round(self, image, target, interactions_list, epoch):
        """Run one round of teacher+student forward and compute combined loss.

        Returns: (loss, student_outputs, teacher_outputs)
        """
        inter_np = np.stack([m.get_numpy() for m in interactions_list])
        interactions = torch.from_numpy(inter_np).to(self.device)
        input_8ch = torch.cat([image, interactions], dim=1)

        with autocast_ctx():
            # Student forward
            student_outputs = self.student(input_8ch)  # list of 5 DS levels

            # Task loss: student vs GT (deep supervision)
            ds_targets = downsample_target_for_ds(target, num_levels=len(student_outputs))
            loss_task = self.task_loss_fn(student_outputs, ds_targets)

            if self.kd_alpha > 0:
                # Teacher forward (frozen, no grad)
                with torch.no_grad():
                    teacher_outputs = self.teacher(input_8ch)
                # KD loss at all deep supervision levels
                loss_kd = kd_loss_deep_supervision(
                    student_outputs, teacher_outputs,
                    temperature=self.kd_temperature,
                    num_ds_levels=len(student_outputs))
                loss = (1 - self.kd_alpha) * loss_task + self.kd_alpha * loss_kd
            else:
                teacher_outputs = None
                loss = loss_task

        return loss, student_outputs, teacher_outputs

    def _train_step(self, batch, epoch):
        """Single training step with distillation.

        Matches inference protocol:
        - bbox case: Round 0 = bbox prompt, Round 1..K = follow-up clicks
        - no-bbox case: Round 0 = initial fg click from GT, Round 1..K = follow-up clicks
        - Each round: teacher+student forward → combined loss → backward
        - Gradient normalized by n_total_rounds for consistent scale
        """
        image = batch['image'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        B = image.shape[0]
        spatial_shape = image.shape[2:]

        # Decide: bbox or no-bbox (matches ~20% no-bbox cases in competition)
        use_bbox = random.random() >= self.no_bbox_prob

        # Generate initial interactions
        interactions_list = []
        for b in range(B):
            gt_np = target[b, 0].cpu().numpy().astype(np.uint8)
            mgr = InteractionManager(spatial_shape)
            if use_bbox:
                mgr.set_initial_bbox(gt_np, jitter=0.05)
            else:
                # No bbox: generate first fg click from GT
                # Simulates eval round 1 where pred=zeros → entire GT is FN → fg click
                mgr.add_followup(np.zeros_like(gt_np), gt_np)
            interactions_list.append(mgr)

        # Determine number of follow-up rounds (1-5, matching eval protocol)
        followup_prob = self._get_followup_prob(epoch)
        if self.max_followup_rounds > 0 and random.random() < followup_prob:
            n_followups = random.randint(1, self.max_followup_rounds)
        else:
            n_followups = 0
        n_total_rounds = 1 + n_followups

        # Click source curriculum: teacher (early) → student (late)
        use_teacher_clicks = random.random() < self._get_teacher_click_prob(epoch)

        # === Round 0 ===
        loss_r0, student_outputs, teacher_outputs = self._forward_round(
            image, target, interactions_list, epoch)

        self.grad_scaler.scale(
            loss_r0 / (self.grad_accumulation_steps * n_total_rounds)).backward()
        loss_total = loss_r0.item()

        # === Follow-up rounds ===
        for k in range(n_followups):
            with torch.no_grad():
                if use_teacher_clicks and teacher_outputs is not None:
                    pred = teacher_outputs[0].argmax(1)
                else:
                    pred = student_outputs[0].argmax(1)

            del student_outputs, teacher_outputs
            torch.cuda.empty_cache()

            for b in range(B):
                pred_np = pred[b].cpu().numpy().astype(np.uint8)
                gt_np = target[b, 0].cpu().numpy().astype(np.uint8)
                interactions_list[b].add_followup(pred_np, gt_np)

            loss_rk, student_outputs, teacher_outputs = self._forward_round(
                image, target, interactions_list, epoch)

            self.grad_scaler.scale(
                loss_rk / (self.grad_accumulation_steps * n_total_rounds)).backward()
            loss_total += loss_rk.item()

        del student_outputs, teacher_outputs

        return loss_total / n_total_rounds, n_total_rounds

    def _optimizer_step(self):
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def train(self, train_loader, num_epochs=None, steps_per_epoch=250,
              log_every=10, save_every=10):
        """Main training loop.

        Args:
            steps_per_epoch: Fixed number of steps per epoch (nnU-Net standard: 250).
                Each step uses a random batch from the infinite dataloader.
                This decouples epoch length from dataset size.
        """
        if num_epochs is None:
            num_epochs = self.max_epochs

        total_steps = num_epochs * steps_per_epoch
        print(f"\n{'='*60}")
        print(f"DISTILLATION TRAINING")
        print(f"  Epochs: {num_epochs}, steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps} ({total_steps * 3.4 / 3600:.1f}h estimated)")
        print(f"  lr: {self.lr}, KD alpha: {self.kd_alpha}, temperature: {self.kd_temperature}")
        print(f"  Grad accumulation: {self.grad_accumulation_steps}")
        print(f"  Follow-up curriculum: {self.followup_prob_start}→{self.followup_prob_end}, max {self.max_followup_rounds} rounds")
        print(f"  No-bbox prob: {self.no_bbox_prob}")
        print(f"  Save every: {save_every} epochs")
        print(f"{'='*60}")

        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        global_step = 0

        # Create infinite iterator from dataloader
        data_iter = iter(train_loader)

        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_rounds = []
            t0 = time.time()

            for step in range(steps_per_epoch):
                # Get next batch (restart iterator if exhausted)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                loss_val, n_rounds = self._train_step(batch, epoch)
                epoch_losses.append(loss_val)
                epoch_rounds.append(n_rounds)
                global_step += 1

                if (step + 1) % self.grad_accumulation_steps == 0:
                    self._optimizer_step()

                if global_step % log_every == 0:
                    print(f"  [E{epoch:03d} S{global_step:05d}] "
                          f"loss={loss_val:.4f} rounds={n_rounds} "
                          f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            # Handle remaining gradient accumulation
            if steps_per_epoch % self.grad_accumulation_steps != 0:
                self._optimizer_step()

            self.lr_scheduler.step(epoch)
            mean_loss = np.mean(epoch_losses)
            mean_rounds = np.mean(epoch_rounds)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:03d}: loss={mean_loss:.4f}, "
                  f"rounds={mean_rounds:.1f}, "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"time={elapsed:.1f}s")

            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(epoch, mean_loss)

    def _save_checkpoint(self, epoch, loss):
        path = self.output_dir / f"student_epoch{epoch:04d}.pth"
        torch.save({
            'epoch': epoch,
            'network_weights': self.student.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'features_per_stage': [m for m in self.student.encoder.stages[0].blocks[0].conv1.conv.weight.shape],
        }, path)
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation training")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--reduction", type=int, default=2, help="Channel reduction factor (2 or 4)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=250, help="Steps per epoch (nnU-Net standard: 250)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (paper: 0.01)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--kd_alpha", type=float, default=0.3, help="KD loss weight")
    parser.add_argument("--kd_temp", type=float, default=3.0, help="KD temperature")
    parser.add_argument("--max_followup_rounds", type=int, default=5, help="Max follow-up click rounds (eval protocol: 5)")
    parser.add_argument("--no_bbox_prob", type=float, default=0.2, help="Prob of no-bbox training (matches ~20% no-bbox cases)")
    parser.add_argument("--max_files", type=int, default=0, help="Max training files (0=all)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"r{args.reduction}_lr{args.lr}_a{args.kd_alpha}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print(f"Loading data from {args.train_dir}...")
    dataset = Layer3Dataset(
        args.train_dir,
        max_files=args.max_files,
        augment=True
    )
    print(f"Dataset: {len(dataset)} files")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Trainer
    trainer = DistillTrainer(
        checkpoint_path=args.checkpoint,
        output_dir=str(output_dir),
        device=args.device,
        reduction=args.reduction,
        lr=args.lr,
        max_epochs=args.epochs,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temp,
        grad_accumulation_steps=args.grad_accum,
        max_followup_rounds=args.max_followup_rounds,
        no_bbox_prob=args.no_bbox_prob,
    )

    # Train
    trainer.train(loader, num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)


if __name__ == '__main__':
    main()
