"""
RefinementModule — post-hoc, full-image refinement for nnInteractive.

Strictly per-object: each (case, round, oid) is an independent refinement
call. Cross-object interaction only goes through the (detached) prev_pred
that comes from the assembly of all refined objects last round.

Works at R³ (default 96³) resolution, independent of the nnInt patch
pipeline:

  inputs:
    F_global         (B, 192, R, R, R)  VISTA3D stage-2 feat (projected)
    pred_nn_k_low    (B, 1,   R, R, R)  current round's per-object sigmoid
    pred_prev_low    (B, 1,   R, R, R)  last round's refined soft prob for
                                         THIS oid (not a binary mask). At R0
                                         pass zeros.
    cd_fg, cd_bg     (B, 1,   R, R, R)  EDT-derived distance maps from all
                                         accumulated fg/bg clicks for this oid
    memory_tokens    (B, T,   token_dim) or None — per-oid history. Tokens
                                         should be encoded from THIS oid's
                                         refined output (detached), not GT.
  output: refined (1, R³) in [0,1], and delta (logit residual).

Design:
  - delta_head zero-initialized → starts as identity (delta=0 → refined==pred_nn_k)
  - Cross-attention Q is downsampled to R_attn (48³ by default, 8× fewer tokens)
  - Memory tokens = mask_summary + click_summary + round_emb → token_dim

Training (v2, on-policy): embedded in a multi-round loop that runs real
nnInt session forwards (no_grad) + refinement (grad) + maxprob assembly,
matching the evaluation pipeline 1:1.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding3D(nn.Module):
    """Same as bottleneck_attention.FourierPositionalEncoding, kept standalone."""

    def __init__(self, output_dim: int, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        raw_dim = 3 * 2 * num_freqs
        self.proj = nn.Linear(raw_dim, output_dim)
        freqs = (2.0 ** torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 3) in [0,1] → (..., output_dim)"""
        scaled = coords.unsqueeze(-1) * self.freqs
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return self.proj(encoded.flatten(-2))


class RefinementModule(nn.Module):
    """
    Args:
        vista_channels: input channels from VISTA3D stage-2 features (default 192)
        vista_compressed: channels after 1×1 projection (default 16)
        c_s: spatial feature channels (default 32)
        token_dim: memory/attention token dim (default 64)
        num_heads: cross-attention heads (default 4)
        R: full working resolution (default 96)
        R_attn: resolution at which cross-attention runs (default 48)
        max_rounds: max interaction rounds for temporal embedding
        max_clicks_per_token: clicks per memory token (flat encoded)

    Param budget target: ~450K.
    """

    def __init__(
        self,
        vista_channels: int = 192,
        vista_compressed: int = 16,
        c_s: int = 32,
        token_dim: int = 64,
        num_heads: int = 4,
        R: int = 96,
        R_attn: int = 48,
        max_rounds: int = 6,
    ):
        super().__init__()
        assert R % R_attn == 0, "R must be divisible by R_attn"
        self.R = R
        self.R_attn = R_attn
        self.token_dim = token_dim
        self.num_heads = num_heads

        # ── Spatial branch ────────────────────────────────────────────────
        self.vista_proj = nn.Conv3d(vista_channels, vista_compressed, 1)
        # Spatial input channels:
        #   vista_compressed + pred_nn_k (1) + pred_prev (1) + diff (1) +
        #   click_dist_fg (1) + click_dist_bg (1) = vista_compressed + 5
        in_ch = vista_compressed + 5
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_ch, c_s, 3, padding=1),
            nn.InstanceNorm3d(c_s, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(c_s, c_s, 3, padding=1),
            nn.InstanceNorm3d(c_s, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # ── Memory encoder ────────────────────────────────────────────────
        # Per-round memory token = mask_summary (16) + click_summary (16) +
        #                          round_emb (16) → proj → token_dim
        # mask_summary: global avg pool + a small conv
        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=4, padding=1),          # R → R/4
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, 3, stride=4, padding=1),         # R/4 → R/16
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        # click_summary: pad/truncate to fixed max_clicks_per_round clicks
        self.max_clicks_per_round = 8
        self.click_mlp = nn.Sequential(
            nn.Linear(self.max_clicks_per_round * 4, 64),     # 4 = pos(3) + fg_flag(1)
            nn.GELU(),
            nn.Linear(64, 16),
        )
        self.round_emb = nn.Embedding(max_rounds, 16)
        self.memory_proj = nn.Linear(16 + 16 + 16, token_dim)

        # ── Cross-attention (Q=spatial @ R_attn, KV=memory) ───────────────
        self.q_proj = nn.Conv3d(c_s, token_dim, 1)
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            token_dim, num_heads, batch_first=True)

        # ── Fusion + head ─────────────────────────────────────────────────
        self.fuse = nn.Sequential(
            nn.Conv3d(c_s + token_dim, c_s, 3, padding=1),
            nn.InstanceNorm3d(c_s, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(c_s, c_s // 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.delta_head = nn.Conv3d(c_s // 2, 1, 1)
        # Zero-init → identity at init time
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def logit_from_sigmoid(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))

    def encode_memory_token(
        self,
        pred_low: torch.Tensor,              # (B, 1, R, R, R) sigmoid, any resolution
        clicks: List[Dict],                  # click dicts for THIS round, for THIS object
        round_idx: int,
    ) -> torch.Tensor:
        """Encode one memory token for one (round, object)."""
        B = pred_low.shape[0]
        device = pred_low.device

        mask_feat = self.mask_encoder(pred_low)              # (B, 16)

        # Click summary: flatten up to max_clicks_per_round
        max_c = self.max_clicks_per_round
        click_vec = torch.zeros(B, max_c * 4, device=device)
        for ci, c in enumerate(clicks[:max_c]):
            base = ci * 4
            click_vec[:, base + 0] = c['pos_norm'][0]
            click_vec[:, base + 1] = c['pos_norm'][1]
            click_vec[:, base + 2] = c['pos_norm'][2]
            click_vec[:, base + 3] = 1.0 if c.get('fg', True) else -1.0
        click_feat = self.click_mlp(click_vec)               # (B, 16)

        r_idx = torch.tensor(
            [min(round_idx, self.round_emb.num_embeddings - 1)],
            dtype=torch.long, device=device).expand(B)
        r_feat = self.round_emb(r_idx)                       # (B, 16)

        token = self.memory_proj(
            torch.cat([mask_feat, click_feat, r_feat], -1))  # (B, token_dim)
        return token

    # ────────────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────────────
    def forward(
        self,
        F_global: torch.Tensor,              # (B, vista_channels, R, R, R)
        pred_nn_k_low: torch.Tensor,         # (B, 1, R, R, R) sigmoid
        pred_prev_low: torch.Tensor,         # (B, 1, R, R, R) sigmoid (zeros at R0)
        click_dist_fg_low: torch.Tensor,     # (B, 1, R, R, R)
        click_dist_bg_low: torch.Tensor,     # (B, 1, R, R, R)
        memory_tokens: Optional[torch.Tensor],  # (B, T, token_dim) or None
    ):
        B, _, D, H, W = pred_nn_k_low.shape
        assert D == H == W == self.R, f"Expected R={self.R}, got ({D},{H},{W})"

        vista = self.vista_proj(F_global)
        diff = pred_nn_k_low - pred_prev_low
        x = torch.cat(
            [vista, pred_nn_k_low, pred_prev_low, diff,
             click_dist_fg_low, click_dist_bg_low],
            dim=1,
        )
        F_sp = self.spatial_conv(x)   # (B, c_s, R, R, R)

        # Cross-attention at R_attn
        R_a = self.R_attn
        if memory_tokens is not None and memory_tokens.shape[1] > 0:
            Q3 = self.q_proj(F_sp)              # (B, token_dim, R, R, R)
            # Downsample Q to R_attn for attention
            Q3_low = F.interpolate(
                Q3, size=(R_a, R_a, R_a), mode='trilinear', align_corners=False)
            Q_seq = Q3_low.flatten(2).transpose(1, 2)   # (B, R_a³, token_dim)
            attn_out, _ = self.attn(
                self.norm_q(Q_seq), self.norm_kv(memory_tokens), memory_tokens)
            attn_out = attn_out.transpose(1, 2).reshape(
                B, self.token_dim, R_a, R_a, R_a)
            # Upsample back to R
            attn_feat = F.interpolate(
                attn_out, size=(self.R, self.R, self.R),
                mode='trilinear', align_corners=False)
        else:
            attn_feat = torch.zeros(
                B, self.token_dim, self.R, self.R, self.R,
                device=F_sp.device, dtype=F_sp.dtype)

        fused = self.fuse(torch.cat([F_sp, attn_feat], dim=1))
        delta = self.delta_head(fused)    # (B, 1, R, R, R)

        logit_nn = self.logit_from_sigmoid(pred_nn_k_low)
        refined = torch.sigmoid(logit_nn + delta)
        return refined, delta


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
