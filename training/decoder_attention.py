"""
Decoder Attention Module — 在更高分辨率 feature map 上注入 click 信息。

关键：比 bottleneck (6³) 更细的分辨率才能让不同 object 的 click 可区分。
默认插入到 decoder stage 1 输出（24³, 256ch）。

Memory 优化：
- Project down 256 → 128（内部 dim）
- 1 层 attention（vs bottleneck 的 2 层）
- 4 heads（vs bottleneck 的 8 heads）
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bottleneck_attention import (
    FourierPositionalEncoding, NormalizedMultiheadAttention, AttentionLayer,
    ROLE_SELF_FG, ROLE_SELF_BG, ROLE_OTHER_FG, ROLE_OTHER_BG, NUM_ROLES,
)


PATCH_SIZE_DEFAULT = 192


def compute_token_pos(click_pos, patch_center, patch_size=PATCH_SIZE_DEFAULT,
                      use_relative=False, spacing_dhw=None):
    """Compute position tensor for a click token.

    Shared between training and eval to guarantee consistency.

    Args:
        click_pos: [d, h, w] — same coordinate system as patch_center
        patch_center: [d, h, w] — center of 192³ patch (latest self click)
        patch_size: int (192)
        use_relative: False → 3D absolute [0,1]; True → 4D [dx,dy,dz,dist]
        spacing_dhw: [sp_d, sp_h, sp_w] voxel spacing (None → isotropic)

    Returns:
        torch.Tensor of shape (3,) or (4,)
    """
    if not use_relative:
        patch_start = [patch_center[d] - patch_size // 2 for d in range(3)]
        pos = [(click_pos[d] - patch_start[d]) / patch_size for d in range(3)]
        return torch.tensor(pos, dtype=torch.float32)

    disp = [(click_pos[d] - patch_center[d]) / patch_size for d in range(3)]
    if spacing_dhw is not None:
        phys = [(click_pos[d] - patch_center[d]) * spacing_dhw[d] for d in range(3)]
        phys_dist = math.sqrt(sum(x * x for x in phys))
        diag = math.sqrt(sum((patch_size * s) ** 2 for s in spacing_dhw))
        dist = phys_dist / max(diag, 1e-6)
    else:
        vox = [click_pos[d] - patch_center[d] for d in range(3)]
        dist = math.sqrt(sum(x * x for x in vox)) / (patch_size * math.sqrt(3))
    return torch.tensor(disp + [dist], dtype=torch.float32)


class DecoderAttentionModule(nn.Module):
    """Click token 注入到 decoder 中层 feature（比 bottleneck 更细）。

    输入: (B, C_in, S, S, S) — 例如 (B, 256, 24, 24, 24)
    输出: (B, C_in, S, S, S) — 零初始化 residual，识别起步

    Token pool 与 sparse-only bottleneck 模块一致：
      - Self/Other FG/BG clicks
      - BG learnable tokens
    """

    def __init__(self, input_dim: int = 256, spatial_size: int = 24,
                 internal_dim: int = 128, num_layers: int = 1,
                 num_heads: int = 4, num_bg_tokens: int = 1,
                 num_fourier_freqs: int = 8, max_rounds: int = 6,
                 use_tanh_gate: bool = False,
                 use_softmax_competition: bool = False,
                 use_relative_pos: bool = False,
                 use_token_gate: bool = False,
                 use_voxel_gate: bool = False,
                 use_learnable_scale: bool = False,
                 use_lora_scale: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.spatial_size = spatial_size
        self.internal_dim = internal_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_tanh_gate = use_tanh_gate
        self.use_softmax_competition = use_softmax_competition
        self.use_relative_pos = use_relative_pos
        self.use_token_gate = use_token_gate
        self.use_voxel_gate = use_voxel_gate
        self.use_learnable_scale = use_learnable_scale
        self.use_lora_scale = use_lora_scale

        # Project down: input_dim → internal_dim
        self.proj_in = nn.Conv3d(input_dim, internal_dim, 1)

        # Spatial PE at given resolution
        self.spatial_pe = nn.Parameter(
            torch.randn(spatial_size ** 3, internal_dim) * 0.02)

        # Token encoders
        pos_input_dim = 4 if use_relative_pos else 3
        self.fourier_pe = FourierPositionalEncoding(
            internal_dim, num_fourier_freqs, input_dim=pos_input_dim)
        self.role_emb = nn.Embedding(NUM_ROLES, internal_dim)
        self.temporal_emb = nn.Embedding(max_rounds, internal_dim)
        self.click_proj = nn.Sequential(
            nn.Linear(internal_dim * 3, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
        )

        # Token relation gate: per-token scalar gate on K/V
        if use_token_gate:
            self.token_gate = nn.Sequential(
                nn.Linear(internal_dim * 3, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.token_gate[-1].weight)
            self.token_gate[-1].bias.data.fill_(3.0)

        # BG tokens
        self.bg_tokens = nn.Parameter(
            torch.zeros(num_bg_tokens, internal_dim))

        # Attention layers
        self.layers = nn.ModuleList([
            AttentionLayer(internal_dim, num_heads) for _ in range(num_layers)
        ])

        # Softmax competition (v5): 每个 token 独立产出 value，空间位置通过 softmax 选择
        if use_softmax_competition:
            self.value_proj = nn.Linear(internal_dim, internal_dim)
            self.softmax_temperature = nn.Parameter(torch.tensor(1.0))

        # Project back: internal_dim → input_dim（零初始化，保证 identity 起步）
        self.proj_out = nn.Conv3d(internal_dim, input_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        if use_tanh_gate:
            self.gate_alpha = nn.Parameter(torch.tensor(0.01))

        if use_voxel_gate:
            self.voxel_gate = nn.Sequential(
                nn.Conv3d(input_dim, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv3d(16, 1, 1),
            )
            nn.init.zeros_(self.voxel_gate[-1].weight)
            self.voxel_gate[-1].bias.data.fill_(3.0)

        if use_learnable_scale:
            self.attn_scale_param = nn.Parameter(torch.tensor(0.0))
        if use_lora_scale:
            self.lora_scale_param = nn.Parameter(torch.tensor(0.663))

    def _encode_clicks(self, clicks: List[Dict], device: torch.device
                       ) -> Optional[torch.Tensor]:
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
        combined = torch.cat([pe, re, te], dim=-1)
        tokens = self.click_proj(combined)
        if self.use_token_gate:
            gate = torch.sigmoid(self.token_gate(combined))
            tokens = tokens * gate
        return tokens.unsqueeze(0)

    def forward(self, feat: torch.Tensor, token_info: Dict) -> torch.Tensor:
        """
        feat: (B, input_dim, S, S, S)
        token_info: {'clicks': list of click dicts}
        returns: (B, input_dim, S, S, S)

        三种模式：
        - v3 (default): BG tokens 参与, 直接 spatial→proj_out→delta
        - v4 (no BG, tanh gate): 无 click 时 identity, click-only tokens
        - v5 (softmax competition): BG + click tokens, softmax 竞争决定每个位置的 delta
        """
        B, C, D, H, W = feat.shape
        device = feat.device

        click_tok = self._encode_clicks(token_info.get('clicks', []), device)

        # v4 模式（无 softmax_competition 且 use_tanh_gate）：无 click 时 identity
        if not self.use_softmax_competition and click_tok is None:
            return feat

        # 1. Project down
        x = self.proj_in(feat)  # (B, internal_dim, S, S, S)

        # 2. Flatten spatial + add PE
        spatial = x.flatten(2).permute(0, 2, 1)  # (B, S³, internal_dim)
        spatial = spatial + self.spatial_pe.unsqueeze(0)

        # 3. Build tokens (click tokens + BG tokens)
        parts = []
        if click_tok is not None:
            parts.append(click_tok.expand(B, -1, -1))
        parts.append(self.bg_tokens.unsqueeze(0).expand(B, -1, -1))
        tokens = torch.cat(parts, dim=1)

        # 4. Attention layers（token self-attn, token→spatial, spatial→token）
        for layer in self.layers:
            tokens, spatial = layer(tokens, spatial)

        # 5. 产出 delta
        if self.use_softmax_competition:
            # v5: 每个 token 独立产出 value，空间位置通过 softmax 选择
            token_values = self.value_proj(tokens)  # (B, N_tokens, internal_dim)
            # 每个空间位置对每个 token 算 ownership score
            scores = torch.bmm(spatial, tokens.transpose(1, 2))  # (B, S³, N_tokens)
            weights = F.softmax(scores / self.softmax_temperature, dim=-1)  # (B, S³, N_tokens)
            # 每个位置的 delta = weighted sum of token values
            delta_spatial = torch.bmm(weights, token_values)  # (B, S³, internal_dim)
            delta_spatial = delta_spatial.permute(0, 2, 1).reshape(
                B, self.internal_dim, D, H, W)
            delta = self.proj_out(delta_spatial)
        else:
            # v3/v4: 直接用修改后的 spatial
            spatial = spatial.permute(0, 2, 1).reshape(
                B, self.internal_dim, D, H, W)
            delta = self.proj_out(spatial)

        # 6. Voxel gate + learnable scale + residual
        if self.use_voxel_gate:
            vg = torch.sigmoid(self.voxel_gate(feat))
            delta = vg * delta
        if self.use_learnable_scale:
            alpha_attn = torch.sigmoid(self.attn_scale_param) * 2
            delta = alpha_attn * delta
        if self.use_tanh_gate:
            return feat + torch.tanh(self.gate_alpha) * delta
        else:
            return feat + delta


class StageWithAttentionWrapper(nn.Module):
    """包装 decoder stage，在其 forward 输出后调用 attention module。

    bypass 模式：跳过 attention，只跑原始 stage（相当于原始 nnInteractive）。
    用于训练时当前 patch 内没有 other click 的情况。
    """

    def __init__(self, original_stage: nn.Module, attention: DecoderAttentionModule):
        super().__init__()
        self.original_stage = original_stage
        self.attention = attention
        self._token_info = {'clicks': []}
        self._bypass = False

    def set_token_info(self, token_info: Dict):
        self._token_info = token_info

    def forward(self, x):
        out = self.original_stage(x)
        if self._bypass:
            return out
        out = self.attention(out, self._token_info)
        return out


def wrap_decoder_stage(decoder: nn.Module, stage_idx: int,
                        attention: DecoderAttentionModule) -> StageWithAttentionWrapper:
    """替换 decoder.stages[stage_idx] 为带 attention 的 wrapper。

    Returns the wrapper so external code can call set_token_info.
    """
    original = decoder.stages[stage_idx]
    wrapper = StageWithAttentionWrapper(original, attention)
    decoder.stages[stage_idx] = wrapper
    return wrapper


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
