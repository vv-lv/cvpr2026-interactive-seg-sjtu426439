"""
Phase 2: 冻结 Attention + 训练 LoRA Decoder

前置条件: Phase 1 (sparse-only attention) 已训练好 checkpoint
策略: 冻结 attention module，只训练 LoRA adapters 让 decoder 适应修改后的 bottleneck

Usage:
    python -m training.run_lora_phase2 \
        --attn_ckpt experiments/bottleneck_attn_sparse/best.pth \
        --num_files 300 --epochs 10 --gpu 0
"""
import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

try:
    import numpy._core
except ImportError:
    import numpy.core as _nc
    sys.modules['numpy._core'] = _nc
    sys.modules['numpy._core.multiarray'] = _nc.multiarray

from training.bottleneck_attention import (
    BottleneckInteractionAttention, build_token_info, normalize_pos,
    ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG,
)
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.lora import apply_lora_to_decoder, get_lora_params, count_lora_params
from training.interaction_sim import (
    InteractionManager, generate_point_blob, sample_point_from_error_region,
    POINT_RADIUS,
)
from training.dataset import _normalize_like_inference, augment_patch
from training.run_bottleneck_attn import (
    find_brats_files, load_and_prepare, generate_initial_click,
    generate_followup_click, PATCH_SIZE, CHECKPOINT_PATH,
)


class LoRAPhase2Trainer:

    def __init__(self, attn_ckpt: str, gpu: int = 0, lr: float = 1e-4,
                 num_rounds: int = 4, lora_rank: int = 4,
                 lora_stages: str = '3,4'):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds

        # 1. 网络
        print("Building network...")
        self.network, _ = build_network(CHECKPOINT_PATH, deep_supervision=True)
        self.network.to(self.device)
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        # 2. LoRA on decoder
        target_stages = [int(s) for s in lora_stages.split(',')]
        n_lora = apply_lora_to_decoder(
            self.network.decoder, target_stages=target_stages,
            rank=lora_rank, alpha=1.0)
        # LoRA 模块创建在 CPU，需要移到 GPU
        self.network.decoder.to(self.device)
        self.lora_params = get_lora_params(self.network)
        for p in self.lora_params:
            p.requires_grad_(True)
        print(f"LoRA params: {n_lora:,} (rank={lora_rank}, stages={target_stages})")
        print(f"  LoRA on device: {self.lora_params[0].device}")

        # 3. Attention module（加载并冻结）
        self.attention = BottleneckInteractionAttention(
            feat_dim=320, num_layers=2, num_heads=8, num_bg_tokens=4,
        ).to(self.device)

        ckpt = torch.load(attn_ckpt, map_location=self.device, weights_only=False)
        missing, unexpected = self.attention.load_state_dict(
            ckpt['attention_state_dict'], strict=False)
        if missing:
            print(f"  [warn] Missing keys: {missing}")
        if unexpected:
            print(f"  [warn] Unexpected keys: {unexpected}")
        print(f"Loaded attention from {attn_ckpt} (epoch={ckpt.get('epoch', '?')})")

        # 冻结 attention
        self.attention.eval()
        for p in self.attention.parameters():
            p.requires_grad_(False)
        print("Attention module FROZEN")

        # 4. Loss
        self.criterion = build_loss(deep_supervision=True).to(self.device)

        # 5. Optimizer（只有 LoRA params）
        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=lr, weight_decay=1e-4)
        self.scaler = GradScaler()

    def train_epoch(self, files: list, epoch: int):
        random.shuffle(files)
        losses = []
        t0 = time.time()
        skipped = 0

        for fi, fpath in enumerate(files):
            image_patch, gt_patch, labels = load_and_prepare(fpath, augment=True)
            if image_patch is None or len(labels) < 2:
                skipped += 1
                continue

            step_loss = self._train_step(image_patch, gt_patch, labels)
            if step_loss is not None:
                losses.append(step_loss)

            if (fi + 1) % 50 == 0:
                elapsed = time.time() - t0
                mean_l = np.mean(losses[-50:]) if losses else 0
                print(f"  [{fi+1}/{len(files)}] loss={mean_l:.4f} "
                      f"skip={skipped} t={elapsed:.0f}s")

        elapsed = time.time() - t0
        mean_loss = np.mean(losses) if losses else 0
        print(f"Epoch {epoch}: loss={mean_loss:.4f} "
              f"n={len(losses)} skip={skipped} time={elapsed:.0f}s")
        return mean_loss

    def _train_step(self, image_patch, gt_patch, labels):
        device = self.device
        K = len(labels)
        shape = (PATCH_SIZE,) * 3

        image_t = torch.from_numpy(image_patch[None, None]).to(device)

        mgrs = {}
        click_hist = defaultdict(list)
        pred_hist = defaultdict(list)
        gt_binaries = {}

        for k in labels:
            mgrs[k] = InteractionManager(shape)
            gt_binaries[k] = (gt_patch == k).astype(np.float32)

        self.optimizer.zero_grad()
        total_loss_val = 0.0
        n_fwd = 0

        for round_idx in range(self.num_rounds):
            for k in labels:
                gt_k = gt_binaries[k]
                if round_idx == 0:
                    center = generate_initial_click(gt_k)
                    if center is None:
                        continue
                    blob = generate_point_blob(shape, center, POINT_RADIUS)
                    mgrs[k].interactions[3] = np.maximum(
                        mgrs[k].interactions[3], blob)
                    click_hist[k].append({
                        'pos': normalize_pos(center),
                        'role': ROLE_SELF_FG,
                        'round': round_idx,
                    })
                else:
                    if not pred_hist[k]:
                        continue
                    pred_np = pred_hist[k][-1]
                    center, is_fg = generate_followup_click(pred_np, gt_k)
                    if center is not None:
                        mgrs[k].set_prev_pred(pred_np)
                        mgrs[k].interactions[3:5] *= mgrs[k].decay
                        blob = generate_point_blob(shape, center, POINT_RADIUS)
                        ch = 3 if is_fg else 4
                        mgrs[k].interactions[ch] = np.maximum(
                            mgrs[k].interactions[ch], blob)
                        click_hist[k].append({
                            'pos': normalize_pos(center),
                            'role': ROLE_SELF_FG if is_fg else ROLE_SELF_BG,
                            'round': round_idx,
                        })

            for k in labels:
                gt_k = gt_binaries[k]
                if gt_k.sum() == 0:
                    continue

                interactions = mgrs[k].get_numpy()
                inter_t = torch.from_numpy(interactions[None]).to(device)
                input_8ch = torch.cat([image_t, inter_t], dim=1)

                other_clicks = []
                for j in labels:
                    if j == k:
                        continue
                    for c in click_hist[j]:
                        other_clicks.append({
                            'pos': c['pos'],
                            'role': ROLE_OTHER_FG if c['role'] == ROLE_SELF_FG else ROLE_OTHER_BG,
                            'round': c['round'],
                        })
                token_info = build_token_info(click_hist[k], other_clicks)

                with autocast_ctx():
                    with torch.no_grad():
                        skips = self.network.encoder(input_8ch)
                        skips = list(skips)
                        skips[-1] = self.attention(skips[-1], token_info)

                    # Detach skips 并重新启用梯度追踪，让 LoRA 能收到梯度
                    skips = [s.detach() for s in skips]

                    # Decoder with LoRA（LoRA params 需要梯度）
                    outputs = self.network.decoder(skips)

                    gt_t = torch.from_numpy(gt_k[None, None]).float().to(device)
                    targets = downsample_target_for_ds(gt_t)
                    loss = self.criterion(outputs, targets)

                n_fwd += 1
                self.scaler.scale(loss / (K * self.num_rounds)).backward()
                total_loss_val += loss.item()

                with torch.no_grad():
                    pred = (outputs[0].argmax(1).squeeze(0) == 1
                            ).cpu().numpy().astype(np.float32)
                    pred_hist[k].append(pred)

        if n_fwd > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.lora_params, max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float,
                        attn_ckpt: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        lora_state = {}
        from training.lora import LoRAConv3d
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()

        torch.save({
            'attention_state_dict': self.attention.state_dict(),
            'lora_state_dict': lora_state,
            'attn_ckpt_source': attn_ckpt,
            'epoch': epoch,
            'loss': loss,
        }, path)
        print(f"Saved: {path} ({len(lora_state)} lora tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_ckpt', required=True,
                        help='Phase 1 attention checkpoint')
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_stages', default='3,4')
    parser.add_argument('--save_dir',
                        default='experiments/bottleneck_attn_lora')
    args = parser.parse_args()

    files = find_brats_files(args.data_root, max_files=args.num_files)
    if not files:
        print("No BraTS files found!")
        return

    trainer = LoRAPhase2Trainer(
        attn_ckpt=args.attn_ckpt, gpu=args.gpu, lr=args.lr,
        num_rounds=args.num_rounds, lora_rank=args.lora_rank,
        lora_stages=args.lora_stages,
    )

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch)

        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(
                os.path.join(args.save_dir, 'best.pth'),
                epoch, loss, args.attn_ckpt)

        if (epoch + 1) % 3 == 0:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, f'epoch_{epoch}.pth'),
                epoch, loss, args.attn_ckpt)

    trainer.save_checkpoint(
        os.path.join(args.save_dir, 'final.pth'),
        args.epochs - 1, loss, args.attn_ckpt)
    print("Done!")


if __name__ == '__main__':
    main()
