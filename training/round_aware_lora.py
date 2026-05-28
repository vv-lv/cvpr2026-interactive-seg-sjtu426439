"""
Round-Aware LoRA: different LoRA weights for different interaction rounds.

Key insight from data:
  R1: improvement/regression = 15x → decoder should be AGGRESSIVE
  R4: improvement/regression = 0.87x → decoder should be CONSERVATIVE

A single set of weights cannot optimize for both. Round-aware LoRA lets the
decoder learn round-specific behavior.
"""
import torch
import torch.nn as nn

from training.lora import LoRAConv3d, _find_parent


class RoundAwareLoRAConv3d(nn.Module):
    """Conv3d with round-group-specific LoRA weights."""

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
        k = original_conv.kernel_size
        s = original_conv.stride
        p = original_conv.padding

        self.lora_A = nn.ModuleList([
            nn.Conv3d(in_ch, rank, kernel_size=k, stride=s,
                      padding=p, bias=False)
            for _ in range(num_groups)
        ])
        self.lora_B = nn.ModuleList([
            nn.Conv3d(rank, out_ch, kernel_size=1, bias=False)
            for _ in range(num_groups)
        ])

        for a in self.lora_A:
            nn.init.kaiming_uniform_(a.weight)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)

    def forward(self, x):
        original_out = self.original_conv(x)
        if self._bypass:
            return original_out
        g = self._active_group
        lora_out = self.lora_B[g](self.lora_A[g](x)) * (self.alpha / self.rank)
        return original_out + lora_out


def apply_round_aware_lora(decoder: nn.Module, target_stages: list = None,
                           rank: int = 4, alpha: float = 1.0,
                           num_groups: int = 2) -> int:
    if target_stages is None:
        target_stages = [2, 3]

    total_params = 0
    for stage_idx in target_stages:
        if stage_idx >= len(decoder.stages):
            continue
        stage = decoder.stages[stage_idx]
        targets = []
        for block_name, block in stage.named_modules():
            if isinstance(block, nn.Conv3d) and block.kernel_size[0] > 1:
                parent, attr = _find_parent(stage, block_name, block)
                if parent is not None:
                    targets.append((block_name, block, parent, attr))

        for block_name, block, parent, attr in targets:
            lora_conv = RoundAwareLoRAConv3d(
                block, rank=rank, alpha=alpha, num_groups=num_groups)
            setattr(parent, attr, lora_conv)
            if hasattr(parent, 'all_modules'):
                for idx, mod in enumerate(parent.all_modules):
                    if mod is block:
                        parent.all_modules[idx] = lora_conv
                        break
            n = sum(p.numel() for g in range(num_groups)
                    for p in [lora_conv.lora_A[g].weight, lora_conv.lora_B[g].weight])
            total_params += n
            print(f"  RoundLoRA added: decoder.stages.{stage_idx}.{block_name} "
                  f"({list(block.weight.shape)}) → rank={rank}, groups={num_groups}, +{n} params")

    return total_params


def set_round_group(model: nn.Module, group: int):
    for m in model.modules():
        if isinstance(m, RoundAwareLoRAConv3d):
            m._active_group = group


def set_round_lora_bypass(model: nn.Module, bypass: bool):
    for m in model.modules():
        if isinstance(m, RoundAwareLoRAConv3d):
            m._bypass = bypass


def get_round_lora_params(model: nn.Module):
    params = []
    for m in model.modules():
        if isinstance(m, RoundAwareLoRAConv3d):
            for a in m.lora_A:
                params.append(a.weight)
            for b in m.lora_B:
                params.append(b.weight)
    return params
