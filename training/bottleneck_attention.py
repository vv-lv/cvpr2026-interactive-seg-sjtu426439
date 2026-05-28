"""
Bottleneck Interaction Attention Module (Sparse-Only)

在 nnInteractive 的 encoder 和 decoder 之间插入 attention 模块，
将 click 交互信息以 sparse token 形式重新注入 bottleneck features。

核心假设：click 信息经深层 encoder 后空间精度丢失，在 bottleneck 重新注入可改善预测。
同时注入其他 object 的 click 信息，打破 per-object 信息孤岛。

设计原则（控制变量）：
- Sparse-only：只用 click tokens + BG embeddings，不用 dense mask tokens
- 无 gate：output_proj 零初始化已保证 identity 起步
- 无 LoRA：先验证 attention 本身有效，再考虑 decoder 适配
- 绕开 mask 分辨率不匹配、prev_mask 质量等工程问题
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Role 常量 ──
ROLE_SELF_FG = 0
ROLE_SELF_BG = 1
ROLE_OTHER_FG = 2
ROLE_OTHER_BG = 3
ROLE_BACKGROUND = 4
NUM_ROLES = 5

# ── 模块常量 ──
BOTTLENECK_SPATIAL = 6  # 192 / 32 = 6
PATCH_SIZE = 192


class FourierPositionalEncoding(nn.Module):
    """Fourier 位置编码。

    输入 (..., input_dim) 坐标，输出 (..., output_dim) 向量。
    频率带 = [1, 2, 4, ..., 2^(num_freqs-1)] × π
    """

    def __init__(self, output_dim: int, num_freqs: int = 8, input_dim: int = 3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        raw_dim = input_dim * 2 * num_freqs
        self.proj = nn.Linear(raw_dim, output_dim)
        freqs = (2.0 ** torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 3) in [0, 1] → (..., output_dim)"""
        scaled = coords.unsqueeze(-1) * self.freqs
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # (..., 3, 2*nf)
        return self.proj(encoded.flatten(-2))


class NormalizedMultiheadAttention(nn.Module):
    """L2-normalized attention with learnable temperature."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.log_tau = nn.Parameter(torch.tensor(math.log(self.head_dim ** 0.5)))

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, D = q.shape
        Nkv = kv.shape[1]
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(q).reshape(B, Nq, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(kv).reshape(B, Nkv, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(kv).reshape(B, Nkv, H, d).permute(0, 2, 1, 3)

        Q = F.normalize(Q.float(), dim=-1).to(V.dtype)
        K = F.normalize(K.float(), dim=-1).to(V.dtype)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.log_tau.exp()
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, Nq, D)
        return self.out_proj(out)


class AttentionLayer(nn.Module):
    """Token self-attn → Cross(token→spatial) → Cross(spatial→token)"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.token_self_attn = NormalizedMultiheadAttention(dim, num_heads)
        self.norm_tsa = nn.LayerNorm(dim)

        self.cross_t2s = NormalizedMultiheadAttention(dim, num_heads)
        self.norm_t2s_q = nn.LayerNorm(dim)
        self.norm_t2s_kv = nn.LayerNorm(dim)

        self.cross_s2t = NormalizedMultiheadAttention(dim, num_heads)
        self.norm_s2t_q = nn.LayerNorm(dim)
        self.norm_s2t_kv = nn.LayerNorm(dim)

        self.ffn_tok = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_ffn_tok = nn.LayerNorm(dim)

        self.ffn_sp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_ffn_sp = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, spatial: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Token self-attention
        t_norm = self.norm_tsa(tokens)
        tokens = tokens + self.token_self_attn(t_norm, t_norm)

        # Tokens attend to spatial
        tokens = tokens + self.cross_t2s(
            self.norm_t2s_q(tokens), self.norm_t2s_kv(spatial))
        tokens = tokens + self.ffn_tok(self.norm_ffn_tok(tokens))

        # Spatial attend to tokens
        spatial = spatial + self.cross_s2t(
            self.norm_s2t_q(spatial), self.norm_s2t_kv(tokens))
        spatial = spatial + self.ffn_sp(self.norm_ffn_sp(spatial))

        return tokens, spatial


class BottleneckInteractionAttention(nn.Module):
    """Sparse-Only Bottleneck Attention.

    Token Pool:
      - Self FG/BG clicks: Fourier PE + role_emb + temporal_emb → MLP
      - Other FG/BG clicks: 同上
      - BG embeddings: 4 个 learnable tokens
      总计约 ~25 tokens

    无 MaskEncoder, 无 output embedding, 无 gate.
    output_proj 零初始化 → identity 起步.
    """

    def __init__(self, feat_dim: int = 320, num_layers: int = 2,
                 num_heads: int = 8, num_bg_tokens: int = 4,
                 num_fourier_freqs: int = 8, max_rounds: int = 6):
        super().__init__()
        self.feat_dim = feat_dim
        S = BOTTLENECK_SPATIAL

        # ── Sparse token 编码 ──
        self.fourier_pe = FourierPositionalEncoding(feat_dim, num_fourier_freqs)
        self.role_emb = nn.Embedding(NUM_ROLES, feat_dim)
        self.temporal_emb = nn.Embedding(max_rounds, feat_dim)

        self.click_proj = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

        # ── Spatial PE ──
        self.spatial_pe = nn.Parameter(torch.randn(S ** 3, feat_dim) * 0.02)

        # ── BG tokens ──
        self.bg_tokens = nn.Parameter(torch.randn(num_bg_tokens, feat_dim) * 0.02)

        # ── Attention layers ──
        self.layers = nn.ModuleList([
            AttentionLayer(feat_dim, num_heads) for _ in range(num_layers)
        ])

        # ── Output projection (零初始化 → identity) ──
        self.output_proj = nn.Conv3d(feat_dim, feat_dim, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _encode_clicks(self, clicks: List[Dict], device: torch.device
                       ) -> Optional[torch.Tensor]:
        """编码 sparse click tokens.

        clicks: list of {'pos': (3,) tensor [0,1], 'role': int, 'round': int}
        returns: (1, N, D) or None
        """
        if not clicks:
            return None

        pos = torch.stack([c['pos'] for c in clicks]).to(device)
        roles = torch.tensor([c['role'] for c in clicks],
                             dtype=torch.long, device=device)
        rounds = torch.tensor([c['round'] for c in clicks],
                              dtype=torch.long, device=device)

        pe = self.fourier_pe(pos)
        re = self.role_emb(roles)
        te = self.temporal_emb(rounds.clamp(min=0, max=5))

        combined = torch.cat([pe, re, te], dim=-1)  # (N, 3D)
        tokens = self.click_proj(combined)           # (N, D)
        return tokens.unsqueeze(0)

    def forward(self, bottleneck_feat: torch.Tensor,
                token_info: Dict) -> torch.Tensor:
        """
        bottleneck_feat: (B, 320, 6, 6, 6)
        token_info: {'clicks': list of click dicts}
        """
        B, C, D, H, W = bottleneck_feat.shape
        device = bottleneck_feat.device

        # 1. Flatten spatial + add spatial PE
        spatial = bottleneck_feat.flatten(2).permute(0, 2, 1)  # (B, 216, C)
        spatial = spatial + self.spatial_pe.unsqueeze(0)

        # 2. Build token pool
        parts = []

        # Click tokens
        click_tok = self._encode_clicks(token_info.get('clicks', []), device)
        if click_tok is not None:
            parts.append(click_tok.expand(B, -1, -1))

        # BG tokens
        parts.append(self.bg_tokens.unsqueeze(0).expand(B, -1, -1))

        tokens = torch.cat(parts, dim=1)  # (B, N_total, C)

        # 3. Attention
        for layer in self.layers:
            tokens, spatial = layer(tokens, spatial)

        # 4. Output (zero-init residual, no gate)
        spatial = spatial.permute(0, 2, 1).reshape(B, C, D, H, W)
        delta = self.output_proj(spatial)
        return bottleneck_feat + delta


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_pos(pos: tuple, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """将 patch 坐标归一化到 [0, 1]."""
    return torch.tensor([p / patch_size for p in pos], dtype=torch.float32)


def build_token_info(self_clicks: List[Dict],
                     other_clicks: List[Dict]) -> Dict:
    """构建 sparse-only token_info."""
    return {'clicks': self_clicks + other_clicks}


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
