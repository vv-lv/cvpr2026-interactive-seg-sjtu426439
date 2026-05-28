"""
Phase 1 Per-Object Resolver: 从竞争上下文学习 overlap 仲裁。

设计：
- 对每个 object k，输入 3ch: (own_logit, max_competitor, sum_pressure)
- 输出 1ch: refined fg logit
- 背景: 可学习 scalar (init=0)
- Assembly: argmax([bg_scalar, refined_1, ..., refined_K])
- ~700 参数，backbone 完全冻结

Phase 2（如果有正信号）:
- 加 decoder features → 35ch 输入
- 加 overlap penalty loss
"""
import torch
import torch.nn as nn


class PerObjectResolver(nn.Module):
    """轻量 per-object resolver。

    输入 3ch:
      ch0: own_fg_logit       — 当前 object 的 fg logit
      ch1: max_competitor      — 最强竞争者的 fg logit
      ch2: sum_pressure        — sum(sigmoid(其他 objects 的 fg logits))
    输出 1ch: refined_fg_logit

    所有 object 共享同一套权重。
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, hidden, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, 1, 1),
        )
        self.bg_logit = nn.Parameter(torch.zeros(1))  # 可学习背景 logit

        # 初始化: 让 resolver 一开始近似于 identity（直接输出 own_logit）
        self._init_near_identity()

    def _init_near_identity(self):
        """初始化使输出 ≈ own_logit，确保零回归。"""
        # 第一层 conv: 只让 ch0 (own_logit) 通过
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        # 最后一层 conv: identity-like
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        """
        Args:
            x: (B, 3, D, H, W)
        Returns:
            refined: (B, 1, D, H, W) — refined fg logit
        """
        return self.net(x)

    def assemble(self, refined_logits: list) -> torch.Tensor:
        """将多个 object 的 refined logits 组装成多类预测。

        Args:
            refined_logits: list of (D, H, W) tensors, 每个是一个 object 的 refined fg logit

        Returns:
            pred: (D, H, W) int64, 0=bg, 1..K=objects
        """
        bg = self.bg_logit.expand_as(refined_logits[0])
        stacked = torch.stack([bg] + refined_logits)  # (K+1, D, H, W)
        return stacked.argmax(0)  # (D, H, W)


def compute_competition_channels(fg_logits: list, k: int) -> torch.Tensor:
    """为 object k 计算 3ch 竞争输入。

    Args:
        fg_logits: list of K tensors, each (D, H, W) — all objects' fg logits
        k: 当前 object 的 index

    Returns:
        channels: (3, D, H, W) — [own_logit, max_competitor, sum_pressure]
    """
    own = fg_logits[k]

    others = [fg_logits[j] for j in range(len(fg_logits)) if j != k]

    if len(others) == 0:
        max_comp = torch.zeros_like(own)
        sum_press = torch.zeros_like(own)
    else:
        stacked = torch.stack(others)          # (K-1, D, H, W)
        max_comp = stacked.max(0)[0]           # (D, H, W)
        sum_press = torch.sigmoid(stacked).sum(0)  # (D, H, W)

    return torch.stack([own, max_comp, sum_press])  # (3, D, H, W)
