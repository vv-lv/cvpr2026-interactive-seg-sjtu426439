"""
Memory Attention Module: 插在 encoder 和 decoder 之间的轻量跨 object/跨轮次记忆机制。

架构:
  encoder(8ch) → skips[0..5]
  skips[-1] (320, 6, 6, 6) → MemoryAttention(skips[-1], memory_bank) → enhanced
  decoder(enhanced + skips[0..4]) → prediction

Memory Bank:
  存储 global_avg_pool(encoder bottleneck) = (320,) per entry
  FIFO, max_size=24 (6 rounds × 4 objects)
  序列化到文件用于跨 Docker 轮次

零回归保证:
  output_proj 初始化为零 → residual add → decoder 初始看到原始 encoder output
"""
import os
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """单层 cross-attention: query attend to memory bank。"""

    def __init__(self, dim: int = 320, n_heads: int = 8, ff_mult: int = 2):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (B, Nq, C) — 当前 bottleneck features (flattened spatial)
            kv: (B, Nk, C) — memory bank entries
        Returns:
            (B, Nq, C) — attended features
        """
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.attn(q_normed, kv_normed, kv_normed)
        q = q + attn_out
        q = q + self.ffn(self.norm_ff(q))
        return q


class MemoryAttention(nn.Module):
    """Encoder-Decoder 之间的 Memory Attention 模块。

    对 encoder bottleneck features 做 cross-attention with memory bank,
    然后 residual add 回原始 features。

    零初始化 output_proj 确保初始行为 = pass-through。
    """

    def __init__(self, dim: int = 320, n_heads: int = 8, n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, n_heads) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(dim, dim)

        # 零初始化: 初始输出 = 0, residual add 后 = 原始 features
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, features: torch.Tensor,
                memory_bank: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (B, C, D, H, W) — encoder bottleneck output
            memory_bank: (N, C) — stacked memory entries, or None
        Returns:
            (B, C, D, H, W) — enhanced features (residual)
        """
        if memory_bank is None or memory_bank.shape[0] == 0:
            return features  # no memory → pass-through

        B, C, D, H, W = features.shape
        # Flatten spatial: (B, C, D*H*W) → (B, D*H*W, C)
        q = features.reshape(B, C, -1).permute(0, 2, 1)

        # Expand memory bank for batch: (N, C) → (B, N, C)
        kv = memory_bank.unsqueeze(0).expand(B, -1, -1).to(q.device)

        # Cross-attention layers
        for layer in self.layers:
            q = layer(q, kv)

        # Project and reshape
        out = self.output_proj(q)  # (B, D*H*W, C)
        out = out.permute(0, 2, 1).reshape(B, C, D, H, W)

        return features + out  # residual add


class MemoryBank:
    """FIFO memory bank: 存储 encoder bottleneck 的 pooled features。"""

    def __init__(self, max_size: int = 24):
        self.max_size = max_size
        self.entries: List[torch.Tensor] = []

    def add(self, feature: torch.Tensor):
        """添加一个 (C,) feature vector。"""
        self.entries.append(feature.detach().cpu())
        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def get_tensor(self) -> Optional[torch.Tensor]:
        """返回 (N, C) stacked tensor, 或 None。"""
        if not self.entries:
            return None
        return torch.stack(self.entries)

    def clear(self):
        self.entries.clear()

    def save(self, path: str):
        torch.save(self.entries, path)

    def load(self, path: str):
        if os.path.exists(path):
            self.entries = torch.load(path, map_location='cpu')

    def __len__(self):
        return len(self.entries)


class MemoryEnhancedUNet(nn.Module):
    """包装原始 ResidualEncoderUNet，在 encoder-decoder 之间插入 memory attention。

    用法:
        base_net = build_network(...)  # 原始 nnInteractive 网络
        memory_attn = MemoryAttention(dim=320)
        model = MemoryEnhancedUNet(base_net, memory_attn)
        model.set_memory_bank(bank.get_tensor())
        output = model(input_8ch)
    """

    def __init__(self, base_network: nn.Module, memory_attention: MemoryAttention):
        super().__init__()
        self.encoder = base_network.encoder
        self.decoder = base_network.decoder
        self.memory_attention = memory_attention
        self._memory_bank: Optional[torch.Tensor] = None

        # 用于从外部捕获 bottleneck features（训练时需要）
        self._last_bottleneck: Optional[torch.Tensor] = None

    def set_memory_bank(self, bank: Optional[torch.Tensor]):
        """设置当前 memory bank（在每个 object 的 forward 之前调用）。"""
        self._memory_bank = bank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 8, D, H, W) — 标准 8ch 输入
        Returns:
            output: same as original network (2ch logits or list for deep supervision)
        """
        skips = self.encoder(x)

        # 保存 bottleneck 用于后续 memory bank 更新
        self._last_bottleneck = skips[-1].detach()

        # Memory attention at bottleneck
        skips[-1] = self.memory_attention(skips[-1], self._memory_bank)

        return self.decoder(skips)

    def get_last_bottleneck_pooled(self) -> torch.Tensor:
        """返回最近一次 forward 的 bottleneck global avg pooled feature (C,)。"""
        if self._last_bottleneck is None:
            raise RuntimeError("No forward pass has been done yet")
        return self._last_bottleneck.mean(dim=[0, 2, 3, 4])  # (C,)
