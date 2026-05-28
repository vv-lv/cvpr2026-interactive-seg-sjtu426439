"""
Layer 2 训练器：最小可行微调。

功能：
- 从 plans.json / checkpoint 构建网络 + 加载权重
- DC_and_CE_loss + DeepSupervisionWrapper
- SGD + PolyLR + AMP
- 支持 frozen 模式（验证 pipeline）和 finetune 模式
- 定期保存 checkpoint，永不覆盖原始权重
"""
import sys
import time
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

def autocast_ctx():
    """兼容新旧版 PyTorch 的 autocast"""
    return torch.cuda.amp.autocast()
from torch.utils.data import DataLoader

# ── numpy 兼容性 workaround ──
try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

# ── nnU-Net loss imports ──
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


# ═══════════════════════════════════════════════════════════════════════════════
# 网络构建（复用 verify_network_build.py 的逻辑）
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_class(class_name: str):
    """从字符串解析 Python 类"""
    parts = class_name.rsplit('.', 1)
    module = __import__(parts[0], fromlist=[parts[1]])
    return getattr(module, parts[1])


def build_network(checkpoint_path: str, deep_supervision: bool = True):
    """从 checkpoint 构建网络并加载权重。

    Returns:
        network: nn.Module
        plans: dict (从 checkpoint 提取)
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    plans = ckpt['init_args']['plans']
    configuration = ckpt['init_args']['configuration']

    # 解析配置（处理继承链）
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

    network_class = resolve_class(arch_info['network_class_name'])
    network = network_class(
        input_channels=8,  # 1 image + 7 interaction
        num_classes=2,      # bg/fg binary
        deep_supervision=deep_supervision,
        **arch_kwargs
    )

    # 加载权重
    network.load_state_dict(ckpt['network_weights'])
    print(f"Loaded weights from {checkpoint_path} (epoch={ckpt.get('current_epoch', '?')})")

    return network, plans


# ═══════════════════════════════════════════════════════════════════════════════
# Loss 构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_loss(deep_supervision: bool = True, num_ds_levels: int = 5):
    """构建 DC_and_CE_loss + DeepSupervisionWrapper。

    参数与 nnUNetTrainer._build_loss() 一致：
    - MemoryEfficientSoftDiceLoss, do_bg=False, smooth=1e-5
    - batch_dice=False (from plans.json)
    - CE 和 Dice 权重均=1
    """
    base_loss = DC_and_CE_loss(
        soft_dice_kwargs={
            'batch_dice': False,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': False,
        },
        ce_kwargs={},
        weight_ce=1.0,
        weight_dice=1.0,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
    )

    if deep_supervision:
        # 权重：[1, 1/2, 1/4, 1/8, 1/16]，归一化 sum=1
        weights = np.array([1 / (2 ** i) for i in range(num_ds_levels)])
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(base_loss, weights)
    else:
        loss = base_loss

    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 深度监督 target 下采样
# ═══════════════════════════════════════════════════════════════════════════════

def downsample_target_for_ds(target: torch.Tensor, num_levels: int = 5) -> list:
    """将 GT 下采样到各深度监督级别。

    用 nearest interpolation（不用 max_pool，因为 max_pool 会膨胀前景标签）。
    这与 nnU-Net 的 DownsampleSegForDSTransform2 行为一致。

    Args:
        target: (B, 1, D, H, W) float32 binary GT（值为 0.0 或 1.0）
        num_levels: 深度监督级别数

    Returns:
        targets: list of (B, 1, D_i, H_i, W_i) tensors
    """
    targets = [target]  # level 0 = full resolution
    for i in range(1, num_levels):
        # 每级缩小 2 倍，nearest 保持标签值
        scale = 0.5 ** i
        size = [max(1, int(s * scale)) for s in target.shape[2:]]
        ds = nn.functional.interpolate(target, size=size, mode='nearest')
        targets.append(ds)
    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# PolyLR Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class PolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """论文使用的多项式 LR 调度：lr = initial_lr × (1 - epoch/max_epoch)^exponent"""

    def __init__(self, optimizer, initial_lr: float, max_steps: int,
                 exponent: float = 0.9, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.initial_lr * (1 - self.last_epoch / self.max_steps) ** self.exponent
                for _ in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class Layer2Trainer:
    """最小可行微调 Trainer。

    用法:
        trainer = Layer2Trainer(checkpoint_path, output_dir, ...)
        trainer.frozen_validation(dataloader, n_batches=10)  # 先验证 pipeline
        trainer.train(train_loader, val_loader, num_epochs=...)
    """

    def __init__(self,
                 checkpoint_path: str,
                 output_dir: str,
                 device: str = 'cuda:0',
                 lr: float = 1e-4,
                 max_epochs: int = 100,
                 freeze_encoder: bool = False,
                 grad_accumulation_steps: int = 1):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.max_epochs = max_epochs
        self.freeze_encoder = freeze_encoder
        self.grad_accumulation_steps = grad_accumulation_steps

        # 构建网络
        self.network, self.plans = build_network(checkpoint_path, deep_supervision=True)
        self.network = self.network.to(self.device)

        # 可选冻结 encoder
        if self.freeze_encoder:
            for name, param in self.network.named_parameters():
                if name.startswith('encoder'):
                    param.requires_grad = False
            trainable = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.network.parameters())
            print(f"Encoder frozen: {trainable:,}/{total:,} params trainable")

        # Loss
        self.loss_fn = build_loss(deep_supervision=True)

        # Optimizer
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            trainable_params,
            lr=self.lr,
            momentum=0.99,
            weight_decay=3e-5,
            nesterov=True,
        )
        self.lr_scheduler = PolyLRScheduler(
            self.optimizer, self.lr, self.max_epochs, exponent=0.9
        )

        # AMP
        self.grad_scaler = GradScaler()

    def _train_step(self, batch: dict) -> float:
        """单步训练，返回 loss 值。"""
        inputs = batch['input'].to(self.device, non_blocking=True)   # (B, 8, D, H, W)
        targets = batch['target'].to(self.device, non_blocking=True)  # (B, 1, D, H, W)

        with autocast_ctx():
            outputs = self.network(inputs)  # list of 5 tensors

            # 下采样 target 匹配各深度监督级别
            ds_targets = downsample_target_for_ds(targets, num_levels=len(outputs))
            loss = self.loss_fn(outputs, ds_targets)

        # 梯度累积
        loss_scaled = loss / self.grad_accumulation_steps
        self.grad_scaler.scale(loss_scaled).backward()

        return loss.item()

    def _optimizer_step(self):
        """执行优化器步骤（梯度累积完成后调用）。"""
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.network.parameters() if p.requires_grad],
            max_norm=12
        )
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def frozen_validation(self, dataloader: DataLoader, n_batches: int = 10):
        """冻结验证：所有参数不动，只看 loss 是否合理且稳定。

        这一步完全不改变权重，用于验证 pipeline 正确性。
        """
        print("\n" + "=" * 60)
        print("FROZEN VALIDATION (no weight updates)")
        print("=" * 60)

        self.network.eval()
        losses = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break

                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                with autocast_ctx():
                    outputs = self.network(inputs)
                    ds_targets = downsample_target_for_ds(targets, num_levels=len(outputs))
                    loss = self.loss_fn(outputs, ds_targets)

                losses.append(loss.item())
                # 打印 argmax 统计
                pred = outputs[0].argmax(1)  # (B, D, H, W)
                fg_frac = pred.float().mean().item()
                gt_fg_frac = targets.mean().item()
                print(f"  Batch {i}: loss={loss.item():.4f}, "
                      f"pred_fg={fg_frac:.3f}, gt_fg={gt_fg_frac:.3f}, "
                      f"name={batch['name'][0]}")

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        print(f"\nFrozen validation: loss={mean_loss:.4f} ± {std_loss:.4f}")
        print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")

        if mean_loss > 5.0:
            print("WARNING: Loss very high, pipeline may have issues")
        elif std_loss / mean_loss > 0.5:
            print("WARNING: Loss very unstable across batches")
        else:
            print("OK: Loss looks reasonable and stable")

        return losses

    def train(self, train_loader: DataLoader, num_epochs: int = None,
              val_loader: Optional[DataLoader] = None,
              log_every: int = 10, save_every: int = 50):
        """主训练循环。"""
        if num_epochs is None:
            num_epochs = self.max_epochs

        print(f"\n{'=' * 60}")
        print(f"TRAINING: {num_epochs} epochs, lr={self.lr}, "
              f"grad_accum={self.grad_accumulation_steps}")
        print(f"{'=' * 60}")

        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        global_step = 0

        for epoch in range(num_epochs):
            epoch_losses = []
            t0 = time.time()

            for batch_idx, batch in enumerate(train_loader):
                loss_val = self._train_step(batch)
                epoch_losses.append(loss_val)
                global_step += 1

                # 梯度累积步骤
                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    self._optimizer_step()

                if global_step % log_every == 0:
                    print(f"  [E{epoch:03d} S{global_step:05d}] "
                          f"loss={loss_val:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}")

            # Epoch 结束
            self.lr_scheduler.step(epoch)
            mean_loss = np.mean(epoch_losses)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:03d}: loss={mean_loss:.4f}, "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"time={elapsed:.1f}s")

            # 保存 checkpoint
            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                self._save_checkpoint(epoch, mean_loss)

    def _save_checkpoint(self, epoch: int, loss: float):
        """保存 checkpoint（永不覆盖原始权重）。"""
        path = self.output_dir / f"checkpoint_epoch{epoch:04d}.pth"
        torch.save({
            'epoch': epoch,
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }, path)
        print(f"  Saved checkpoint: {path}")
