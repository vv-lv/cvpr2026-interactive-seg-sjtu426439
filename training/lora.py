"""
LoRA (Low-Rank Adaptation) for Conv3d layers.

对 nnInteractive decoder 的 conv 层添加低秩适配器。
初始状态: lora_B 零初始化 → delta=0 → 不改变原始行为。
"""
import math

import torch
import torch.nn as nn


class LoRAConv3d(nn.Module):
    """Conv3d 的 LoRA 适配器。

    原始 conv 冻结，额外学习低秩更新：
        output = original_conv(x) + (alpha/rank) * lora_B(lora_A(x))

    lora_A: Conv3d(in_ch, rank, kernel_size=original, stride=original, padding=original)
    lora_B: Conv3d(rank, out_ch, kernel_size=1)
    lora_B 零初始化 → 初始 delta = 0
    """

    def __init__(self, original_conv: nn.Conv3d, rank: int = 4, alpha: float = 1.0,
                 dropout: float = 0.0, use_rslora: bool = False):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self._bypass = False
        self.use_rslora = use_rslora

        # 冻结原始 conv
        for p in self.original_conv.parameters():
            p.requires_grad_(False)

        in_ch = original_conv.in_channels
        out_ch = original_conv.out_channels
        k = original_conv.kernel_size
        s = original_conv.stride
        p = original_conv.padding

        # LoRA 分解
        self.lora_A = nn.Conv3d(in_ch, rank, kernel_size=k, stride=s,
                                padding=p, bias=False)
        self.lora_B = nn.Conv3d(rank, out_ch, kernel_size=1, bias=False)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 初始化：A ~ kaiming, B = 0 → 初始 delta = 0
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_out = self.original_conv(x)
        if self._bypass:
            return original_out
        scale = getattr(self, '_external_scale', None)
        if scale is not None:
            divisor = math.sqrt(self.rank) if self.use_rslora else self.rank
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * (scale / divisor)
        else:
            divisor = math.sqrt(self.rank) if self.use_rslora else self.rank
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * (self.alpha / divisor)
        return original_out + lora_out


def apply_lora_to_decoder(decoder: nn.Module, target_stages: list = None,
                          rank: int = 4, alpha: float = 1.0,
                          dropout: float = 0.0, use_rslora: bool = False) -> int:
    """对 decoder 指定 stages 的 conv 层添加 LoRA。

    Args:
        decoder: UNetDecoder module
        target_stages: 要添加 LoRA 的 stage 索引列表，如 [3, 4]（最浅 2 层）
        rank: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        新增的 LoRA 参数量
    """
    if target_stages is None:
        target_stages = [3, 4]  # 最浅 2 层

    total_params = 0

    for stage_idx in target_stages:
        if stage_idx >= len(decoder.stages):
            continue

        stage = decoder.stages[stage_idx]

        # 先收集所有目标 conv（避免迭代中修改导致重复匹配）
        targets = []
        for block_name, block in stage.named_modules():
            if isinstance(block, nn.Conv3d) and block.kernel_size[0] > 1:
                parent, attr = _find_parent(stage, block_name, block)
                if parent is not None:
                    targets.append((block_name, block, parent, attr))

        # 再逐一替换
        for block_name, block, parent, attr in targets:
            lora_conv = LoRAConv3d(block, rank=rank, alpha=alpha,
                                   dropout=dropout, use_rslora=use_rslora)
            setattr(parent, attr, lora_conv)

            # 修复 all_modules Sequential 引用
            if hasattr(parent, 'all_modules'):
                for idx, mod in enumerate(parent.all_modules):
                    if mod is block:
                        parent.all_modules[idx] = lora_conv
                        break

            n = sum(p.numel() for p in [lora_conv.lora_A.weight,
                                         lora_conv.lora_B.weight])
            total_params += n
            print(f"  LoRA added: decoder.stages.{stage_idx}.{block_name} "
                  f"({list(block.weight.shape)}) → rank={rank}, +{n} params")

    return total_params


def _find_parent(root: nn.Module, dotted_name: str, target: nn.Module):
    """找到 target module 的 parent 和 attribute name。"""
    parts = dotted_name.split('.')
    if len(parts) == 1:
        return root, parts[0]

    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def set_lora_bypass(model: nn.Module, bypass: bool):
    """设置所有 LoRA 模块的 bypass 状态。"""
    for module in model.modules():
        if isinstance(module, LoRAConv3d):
            module._bypass = bypass


def get_lora_params(model: nn.Module):
    """收集所有 LoRA 参数。"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRAConv3d):
            params.extend([module.lora_A.weight, module.lora_B.weight])
    return params


def count_lora_params(model: nn.Module) -> int:
    """统计 LoRA 参数量。"""
    return sum(p.numel() for p in get_lora_params(model))
