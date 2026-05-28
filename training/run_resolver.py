#!/usr/bin/env python3
"""
Phase 1 Resolver 训练：3ch 输入，~700 参数，backbone 冻结。

用法:
  # Smoke test
  python training/run_resolver.py --mode frozen --max_files 100 --n_batches 20

  # 训练
  python training/run_resolver.py --mode train --max_files 0 --epochs 15 --lr 1e-3

  # 评估
  python training/run_resolver.py --mode eval --max_files 100 --checkpoint experiments/resolver_p1/best.pth
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
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy.core as _nc
sys.modules['numpy._core'] = _nc
sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.trainer import build_network, autocast_ctx
from training.dataset import extract_patch, augment_patch, PATCH_SIZE
from training.interaction_sim import InteractionManager
from training.resolver import PerObjectResolver, compute_competition_channels

DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = "/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "resolver_p1"

MAX_OBJECTS_PER_STEP = 4  # 限制每步处理的 object 数


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Object Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObjectDataset(Dataset):
    """返回包含多个 object 的 patch + 完整多类 GT。

    只保留 ≥2 objects 的 case。
    """

    def __init__(self, data_dir: str, patch_size=PATCH_SIZE,
                 max_files: int = 0, augment: bool = True):
        self.patch_size = patch_size
        self.augment = augment

        data_dir = Path(data_dir)
        all_files = sorted(data_dir.rglob("*.npz"))

        # 用缓存快速过滤 ≥2 objects 的 case
        import json
        cache_path = data_dir.parent / "object_count_cache.json"
        if cache_path.exists():
            with open(cache_path) as fp:
                cache = json.load(fp)
            self.files = [f for f in all_files if cache.get(f.name, 0) >= 2]
        else:
            # 无缓存: 逐个扫描 (慢)
            print("WARNING: No object_count_cache.json, scanning files (slow)...")
            self.files = []
            for f in all_files:
                data = np.load(f, allow_pickle=True)
                if len(np.unique(data['gts'])) - 1 >= 2:
                    self.files.append(f)

        if max_files > 0:
            random.shuffle(self.files)
            self.files = self.files[:max_files]

        print(f"MultiObjectDataset: {len(self.files)} files with ≥2 objects (from {len(all_files)} total)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        image = data['imgs'].astype(np.float32)
        gt = data['gts'].astype(np.uint8)

        # Image-level z-score
        nonzero = image > 0
        if nonzero.sum() > 0:
            m, s = image[nonzero].mean(), image[nonzero].std()
            if s > 0:
                image = (image - m) / s
            else:
                image = image - m

        labels = np.unique(gt)
        labels = labels[labels > 0]

        # 策略：找一个 patch 中心使得 patch 内包含 ≥2 objects
        # 尝试多次，优先选靠近多个 object 的区域
        best_center = np.array([d // 2 for d in image.shape])
        best_n_obj = 0

        for _attempt in range(10):
            # 随机选一个 object，以其前景体素为中心候选
            target = int(np.random.choice(labels))
            fg_coords = np.argwhere(gt == target)
            if len(fg_coords) == 0:
                continue
            center = fg_coords[random.randint(0, len(fg_coords) - 1)]

            # 快速检查这个中心附近有多少 objects
            slices_check = tuple(
                slice(max(0, center[d] - self.patch_size[d] // 2),
                      min(image.shape[d], center[d] + self.patch_size[d] // 2))
                for d in range(3)
            )
            patch_gt_check = gt[slices_check]
            n_obj = len(np.unique(patch_gt_check)) - 1  # exclude bg
            if n_obj > best_n_obj:
                best_n_obj = n_obj
                best_center = center
            if n_obj >= 2:
                break  # 找到了 ≥2 objects 的位置

        center = best_center

        # 计算 patch 范围
        starts, ends = [], []
        for d in range(3):
            half = self.patch_size[d] // 2
            s = max(0, min(center[d] - half, image.shape[d] - self.patch_size[d]))
            e = s + self.patch_size[d]
            if e > image.shape[d]:
                e = image.shape[d]
                s = max(0, e - self.patch_size[d])
            starts.append(s)
            ends.append(e)

        slices = tuple(slice(s, e) for s, e in zip(starts, ends))
        img_patch = image[slices]
        gt_patch = gt[slices]

        # Pad if needed
        pad = [(0, max(0, self.patch_size[d] - img_patch.shape[d])) for d in range(3)]
        if any(p[1] > 0 for p in pad):
            img_patch = np.pad(img_patch, pad, mode='constant', constant_values=0)
            gt_patch = np.pad(gt_patch, pad, mode='constant', constant_values=0)

        # 增强（augment_patch 已兼容多类 GT：rotation 用 order=0 nearest + round）
        if self.augment:
            img_patch, gt_patch = augment_patch(img_patch, gt_patch, layer3=True)

        # Patch 内有哪些 objects
        patch_labels = np.unique(gt_patch)
        patch_labels = patch_labels[patch_labels > 0].tolist()

        return {
            'image': torch.from_numpy(img_patch[np.newaxis].copy()),
            'gt': torch.from_numpy(gt_patch.copy().astype(np.int64)),
            'object_ids': patch_labels,
            'name': self.files[idx].stem,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Resolver Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class ResolverTrainer:

    def __init__(self, backbone_ckpt, output_dir, device='cuda:0',
                 lr=1e-3, max_epochs=15):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs

        # Backbone (冻结)
        self.backbone, _ = build_network(backbone_ckpt, deep_supervision=False)
        self.backbone = self.backbone.to(self.device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Resolver
        self.resolver = PerObjectResolver(hidden=16).to(self.device)
        n_params = sum(p.numel() for p in self.resolver.parameters())
        print(f"Resolver params: {n_params}")

        self.optimizer = torch.optim.AdamW(self.resolver.parameters(), lr=lr, weight_decay=1e-4)

    def _get_fg_logits(self, image, gt_multiclass, object_ids):
        """为每个 object 跑 backbone，返回 fg logits 列表。

        Args:
            image: (1, 1, D, H, W) on GPU
            gt_multiclass: (D, H, W) numpy int
            object_ids: list of int

        Returns:
            fg_logits: list of (D, H, W) tensors on GPU (detached)
        """
        fg_logits = []
        spatial = image.shape[2:]

        for oid in object_ids:
            gt_binary = (gt_multiclass == oid).astype(np.uint8)

            # 生成 bbox 交互
            mgr = InteractionManager(spatial)
            mgr.set_initial_bbox(gt_binary, jitter=0.05)
            inter = torch.from_numpy(mgr.get_numpy()).unsqueeze(0).to(self.device)

            input_8ch = torch.cat([image, inter], dim=1)  # (1, 8, D, H, W)

            with torch.no_grad(), autocast_ctx():
                output = self.backbone(input_8ch)  # (1, 2, D, H, W)

            # fg logit = output[:, 1] - output[:, 0]  (unnormalized)
            fg = (output[0, 1] - output[0, 0]).float()
            fg_logits.append(fg)

            del input_8ch, output

        return fg_logits

    def _train_step(self, batch):
        """一个训练步骤。"""
        image = batch['image'].to(self.device)           # (1, 1, D, H, W)
        gt_np = batch['gt'][0].numpy()                   # (D, H, W) int64
        # DataLoader 对 variable-length list 的处理不一致，安全提取
        raw_ids = batch['object_ids']
        if isinstance(raw_ids, torch.Tensor):
            object_ids = raw_ids[0].tolist() if raw_ids.dim() > 1 else raw_ids.tolist()
        elif isinstance(raw_ids, (list, tuple)):
            object_ids = [x.item() if isinstance(x, torch.Tensor) else x for x in raw_ids]
        else:
            object_ids = list(raw_ids)

        # 限制 object 数量
        if len(object_ids) > MAX_OBJECTS_PER_STEP:
            object_ids = random.sample(object_ids, MAX_OBJECTS_PER_STEP)
        if len(object_ids) < 2:
            return None  # 跳过单 object patch

        K = len(object_ids)

        # 1. Backbone forward (no_grad) for each object
        fg_logits = self._get_fg_logits(image, gt_np, object_ids)

        # 2. Resolver forward for each object
        refined_list = []
        for k in range(K):
            comp_ch = compute_competition_channels(fg_logits, k)  # (3, D, H, W)
            resolver_in = comp_ch.unsqueeze(0).to(self.device)    # (1, 3, D, H, W)
            refined = self.resolver(resolver_in)                   # (1, 1, D, H, W)
            refined_list.append(refined[0, 0])                     # (D, H, W)

        # 3. Assembly: [bg_scalar, refined_1, ..., refined_K]
        bg = self.resolver.bg_logit.expand_as(refined_list[0])
        stacked = torch.stack([bg] + refined_list)  # (K+1, D, H, W)

        # 4. Remap GT to 0..K
        gt_remapped = torch.zeros_like(batch['gt'][0], device=self.device)
        for new_idx, oid in enumerate(object_ids):
            gt_remapped[gt_np == oid] = new_idx + 1
        # 0=background, 1..K=objects

        # 5. CE loss
        loss = F.cross_entropy(
            stacked.unsqueeze(0),          # (1, K+1, D, H, W)
            gt_remapped.unsqueeze(0).long() # (1, D, H, W)
        )

        # 6. Contested voxel 统计 (for logging)
        with torch.no_grad():
            pred = stacked.argmax(0)
            # Contested: ≥2 objects 的 fg logit > 0
            fg_pos = torch.stack(fg_logits) > 0  # (K, D, H, W)
            contested = fg_pos.sum(0) >= 2         # (D, H, W)
            n_contested = contested.sum().item()

            if n_contested > 0:
                contested_correct = (pred[contested] == gt_remapped[contested]).float().mean().item()
            else:
                contested_correct = float('nan')

            overall_correct = (pred == gt_remapped).float().mean().item()

        return loss, overall_correct, contested_correct, n_contested

    def frozen_validation(self, dataloader, n_batches=20):
        """冻结 resolver 验证 pipeline。"""
        print("\n" + "=" * 60)
        print("FROZEN VALIDATION (Resolver Phase 1)")
        print("=" * 60)

        self.resolver.eval()
        results = []

        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            with torch.no_grad():
                ret = self._train_step(batch)
            if ret is None:
                print(f"  Batch {i}: skipped (single object)")
                continue
            loss, acc, contested_acc, n_contested = ret
            results.append((loss.item(), acc, contested_acc, n_contested))
            print(f"  Batch {i}: loss={loss.item():.4f}, acc={acc:.3f}, "
                  f"contested_acc={contested_acc:.3f}, n_contested={n_contested}, "
                  f"name={batch['name'][0]}")

        if results:
            losses = [r[0] for r in results]
            accs = [r[1] for r in results]
            c_accs = [r[2] for r in results if not np.isnan(r[2])]
            print(f"\nFrozen: loss={np.mean(losses):.4f}, "
                  f"overall_acc={np.mean(accs):.3f}, "
                  f"contested_acc={np.mean(c_accs):.3f} ({len(c_accs)} batches with contested voxels)")

    def train(self, train_loader, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.max_epochs

        print(f"\n{'=' * 60}")
        print(f"TRAINING Resolver Phase 1: {num_epochs} epochs, "
              f"{sum(p.numel() for p in self.resolver.parameters())} params")
        print(f"{'=' * 60}")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.resolver.train()
            epoch_losses = []
            epoch_accs = []
            epoch_caccs = []
            t0 = time.time()

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                ret = self._train_step(batch)
                if ret is None:
                    continue
                loss, acc, contested_acc, n_contested = ret
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_accs.append(acc)
                if not np.isnan(contested_acc):
                    epoch_caccs.append(contested_acc)

            elapsed = time.time() - t0
            mean_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            mean_acc = np.mean(epoch_accs) if epoch_accs else float('nan')
            mean_cacc = np.mean(epoch_caccs) if epoch_caccs else float('nan')

            print(f"Epoch {epoch:03d}: loss={mean_loss:.4f}, "
                  f"acc={mean_acc:.3f}, contested_acc={mean_cacc:.3f}, "
                  f"bg_logit={self.resolver.bg_logit.item():.3f}, "
                  f"time={elapsed:.1f}s")

            # Save best
            if mean_loss < best_loss:
                best_loss = mean_loss
                path = self.output_dir / "best.pth"
                torch.save({
                    'epoch': epoch,
                    'resolver_state': self.resolver.state_dict(),
                    'loss': mean_loss,
                    'contested_acc': mean_cacc,
                }, path)

        # Save final
        path = self.output_dir / "final.pth"
        torch.save({
            'epoch': num_epochs - 1,
            'resolver_state': self.resolver.state_dict(),
            'loss': mean_loss,
        }, path)
        print(f"Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "train"], default="frozen")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 = main process (safer for variable-length object_ids)")
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Mode: {args.mode}")

    dataset = MultiObjectDataset(
        data_dir=args.data_dir,
        max_files=args.max_files,
        augment=(args.mode == "train"),
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)

    trainer = ResolverTrainer(
        backbone_ckpt=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        lr=args.lr,
        max_epochs=args.epochs,
    )

    if args.mode == "frozen":
        trainer.frozen_validation(dataloader, n_batches=args.n_batches)
    elif args.mode == "train":
        print("\n--- Pre-training check (5 batches) ---")
        trainer.frozen_validation(dataloader, n_batches=5)
        print("\n--- Training ---")
        trainer.train(dataloader, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
