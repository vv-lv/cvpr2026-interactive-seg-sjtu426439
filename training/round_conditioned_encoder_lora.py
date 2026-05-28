"""
Round-Conditioned Encoder LoRA.

Two groups of LoRA on encoder stages 3-4, hard-switched by round:
  Group 0 (R0-R1): aggressive mode for initial segmentation + first correction
  Group 1 (R2+): conservative mode for fine-grained refinement

LoRA uses 1×1×1 pointwise decomposition (not matching original kernel size)
for parameter efficiency. Injected on conv2 of each residual block (residual branch).
"""
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn


class RCLoRAConv3d(nn.Module):
    """Round-conditioned LoRA wrapper for a frozen Conv3d.

    Uses 1×1×1 pointwise decomposition:
        delta = conv1x1_B(conv1x1_A(x)) * (alpha / rank)
    """

    def __init__(self, original_conv: nn.Conv3d, rank: int = 4,
                 alpha: float = 1.0, num_groups: int = 2):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.num_groups = num_groups
        self._active_group = 0
        self._bypass = False

        for p in self.original_conv.parameters():
            p.requires_grad_(False)

        in_ch = original_conv.in_channels
        out_ch = original_conv.out_channels

        self.lora_A = nn.ModuleList([
            nn.Conv3d(in_ch, rank, kernel_size=1, bias=False)
            for _ in range(num_groups)
        ])
        self.lora_B = nn.ModuleList([
            nn.Conv3d(rank, out_ch, kernel_size=1, bias=False)
            for _ in range(num_groups)
        ])

        for a in self.lora_A:
            nn.init.kaiming_normal_(a.weight, mode='fan_out')
        for b in self.lora_B:
            nn.init.zeros_(b.weight)

    def forward(self, x):
        original_out = self.original_conv(x)
        if self._bypass:
            return original_out
        g = self._active_group
        lora_out = self.lora_B[g](self.lora_A[g](x)) * (self.alpha / self.rank)
        return original_out + lora_out


def apply_rc_lora_to_encoder(encoder: nn.Module,
                              target_stages: List[int] = (3, 4),
                              rank: int = 4, alpha: float = 1.0,
                              num_groups: int = 2) -> int:
    """Apply round-conditioned LoRA to conv2 of each residual block in target stages."""
    total_params = 0

    for stage_idx in target_stages:
        stage = encoder.stages[stage_idx]
        for block_name, block in stage.named_children():
            if not block_name.startswith('blocks'):
                continue
            for sub_name, sub_block in block.named_children():
                conv2_mod = None
                conv2_name = None
                for n, m in sub_block.named_modules():
                    if n == 'conv2.conv' and isinstance(m, nn.Conv3d):
                        conv2_mod = m
                        conv2_name = n
                        break
                if conv2_mod is None:
                    continue

                lora = RCLoRAConv3d(conv2_mod, rank=rank, alpha=alpha,
                                    num_groups=num_groups)
                # Replace conv2.conv AND all_modules[0] with LoRA wrapper
                conv2_container = sub_block.conv2
                conv2_container.conv = lora
                if hasattr(conv2_container, 'all_modules'):
                    for idx, mod in enumerate(conv2_container.all_modules):
                        if mod is conv2_mod:
                            conv2_container.all_modules[idx] = lora
                            break

                n_params = sum(
                    p.numel() for g in range(num_groups)
                    for p in [lora.lora_A[g].weight, lora.lora_B[g].weight])
                total_params += n_params
                print(f"  RCLoRA: encoder.stages.{stage_idx}.{block_name}.{sub_name}.conv2.conv "
                      f"({conv2_mod.in_channels}→{conv2_mod.out_channels}) "
                      f"rank={rank} groups={num_groups} +{n_params}")

    return total_params


def set_rc_lora_group(model: nn.Module, group: int):
    for m in model.modules():
        if isinstance(m, RCLoRAConv3d):
            m._active_group = group


def set_rc_lora_bypass(model: nn.Module, bypass: bool):
    for m in model.modules():
        if isinstance(m, RCLoRAConv3d):
            m._bypass = bypass


def get_rc_lora_params(model: nn.Module) -> list:
    params = []
    for m in model.modules():
        if isinstance(m, RCLoRAConv3d):
            for a in m.lora_A:
                params.append(a.weight)
            for b in m.lora_B:
                params.append(b.weight)
    return params


def save_rc_lora_state(model: nn.Module) -> dict:
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, RCLoRAConv3d):
            for g in range(m.num_groups):
                state[f'{name}.lora_A.{g}.weight'] = m.lora_A[g].weight.data.cpu()
                state[f'{name}.lora_B.{g}.weight'] = m.lora_B[g].weight.data.cpu()
    return state


def load_rc_lora_state(model: nn.Module, state: dict):
    for name, m in model.named_modules():
        if isinstance(m, RCLoRAConv3d):
            for g in range(m.num_groups):
                a_key = f'{name}.lora_A.{g}.weight'
                b_key = f'{name}.lora_B.{g}.weight'
                if a_key in state:
                    m.lora_A[g].weight.data.copy_(state[a_key])
                if b_key in state:
                    m.lora_B[g].weight.data.copy_(state[b_key])
