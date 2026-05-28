"""
Joint Training: Attention + LoRA Decoder

从 sparse-only best checkpoint 出发，联合训练：
- Attention module: lr=1e-4（比 Phase 1 的 3e-4 低，防止偏离太远）
- LoRA decoder: lr=3e-5（让 decoder 慢慢适应）
两者共同优化，attention 可以从"保守 delta"过渡到"更有信息量的 delta"，
LoRA 同时学习如何利用这些 delta。

Usage:
    python -m training.run_joint \
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
    count_parameters, ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG,
)
from training.trainer import build_network, build_loss, downsample_target_for_ds, autocast_ctx
from training.lora import apply_lora_to_decoder, get_lora_params
from training.interaction_sim import (
    InteractionManager, generate_point_blob, sample_point_from_error_region,
    POINT_RADIUS,
)
from training.run_bottleneck_attn import (
    find_brats_files, load_and_prepare, generate_initial_click,
    generate_followup_click, PATCH_SIZE, CHECKPOINT_PATH,
)


class JointTrainer:

    def __init__(self, attn_ckpt: str, gpu: int = 0,
                 attn_lr: float = 1e-4, lora_lr: float = 3e-5,
                 num_rounds: int = 4, lora_rank: int = 4,
                 lora_stages: str = '3,4'):
        self.device = torch.device(f'cuda:{gpu}')
        self.num_rounds = num_rounds

        # 1. 网络（encoder 全冻结）
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
        self.network.decoder.to(self.device)
        self.lora_params = get_lora_params(self.network)
        for p in self.lora_params:
            p.requires_grad_(True)
        print(f"LoRA: {n_lora:,} params, device={self.lora_params[0].device}")

        # 3. Attention module（从 checkpoint 加载，可训练）
        self.attention = BottleneckInteractionAttention(
            feat_dim=320, num_layers=2, num_heads=8, num_bg_tokens=4,
        ).to(self.device)

        ckpt = torch.load(attn_ckpt, map_location=self.device, weights_only=False)
        self.attention.load_state_dict(ckpt['attention_state_dict'], strict=False)
        print(f"Loaded attention from {attn_ckpt} (epoch={ckpt.get('epoch', '?')})")

        n_attn = count_parameters(self.attention)
        print(f"Attention: {n_attn:,} params (TRAINABLE, lr={attn_lr})")
        print(f"LoRA: {n_lora:,} params (TRAINABLE, lr={lora_lr})")

        # 4. Loss
        self.criterion = build_loss(deep_supervision=True).to(self.device)

        # 5. Optimizer — 分组 lr
        self.optimizer = torch.optim.AdamW([
            {'params': list(self.attention.parameters()),
             'lr': attn_lr, 'weight_decay': 1e-4},
            {'params': self.lora_params,
             'lr': lora_lr, 'weight_decay': 1e-4},
        ])
        self.scaler = GradScaler()

    def train_epoch(self, files: list, epoch: int):
        self.attention.train()
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
                        'role': ROLE_SELF_FG, 'round': round_idx,
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
                    # Encoder: 冻结，no_grad
                    with torch.no_grad():
                        skips = self.network.encoder(input_8ch)
                    skips = [s.detach() for s in skips]

                    # Attention: 可训练
                    skips[-1] = self.attention(skips[-1], token_info)

                    # Decoder with LoRA: LoRA 可训练
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
            all_params = list(self.attention.parameters()) + self.lora_params
            nn.utils.clip_grad_norm_(
                [p for p in all_params if p.requires_grad], max_norm=12.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss_val / max(n_fwd, 1)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        from training.lora import LoRAConv3d
        lora_state = {}
        for name, module in self.network.named_modules():
            if isinstance(module, LoRAConv3d):
                lora_state[f'{name}.lora_A.weight'] = module.lora_A.weight.data.cpu()
                lora_state[f'{name}.lora_B.weight'] = module.lora_B.weight.data.cpu()

        torch.save({
            'attention_state_dict': self.attention.state_dict(),
            'lora_state_dict': lora_state,
            'epoch': epoch,
            'loss': loss,
        }, path)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_ckpt', required=True)
    parser.add_argument('--data_root',
                        default='/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all')
    parser.add_argument('--num_files', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--attn_lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=3e-5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_stages', default='3,4')
    parser.add_argument('--save_dir',
                        default='experiments/bottleneck_attn_joint')
    args = parser.parse_args()

    files = find_brats_files(args.data_root, max_files=args.num_files)
    if not files:
        print("No BraTS files found!")
        return

    trainer = JointTrainer(
        attn_ckpt=args.attn_ckpt, gpu=args.gpu,
        attn_lr=args.attn_lr, lora_lr=args.lora_lr,
        num_rounds=args.num_rounds, lora_rank=args.lora_rank,
        lora_stages=args.lora_stages,
    )

    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(files, epoch)

        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(
                os.path.join(args.save_dir, 'best.pth'), epoch, loss)

        if (epoch + 1) % 3 == 0:
            trainer.save_checkpoint(
                os.path.join(args.save_dir, f'epoch_{epoch}.pth'),
                epoch, loss)

    trainer.save_checkpoint(
        os.path.join(args.save_dir, 'final.pth'),
        args.epochs - 1, loss)
    print("Done!")


if __name__ == '__main__':
    main()
