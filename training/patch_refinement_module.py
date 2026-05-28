"""
PatchRefinementModule (v5) — patch-native full-resolution refinement.

Difference from v3 (RefinementModule):
  - Operates on 96^3 full-resolution patches (not globally-downsampled).
  - F_global is NOT spatially concatenated. Instead, k learnable queries
    attend over the 48^3 VISTA feature to produce k global tokens.
  - A patch-position Fourier token tells the module where the patch sits
    inside the volume.
  - Spatial input is 5 channels (no vista on spatial path).
  - Intended to be trained from scratch on v4 full-res fp16 rollouts.

Forward inputs:
    pred_nn_k_patch      (B, 1, R, R, R)  full-res per-obj sigmoid crop
    pred_prev_patch      (B, 1, R, R, R)  last round refined (zeros at R0)
    click_dist_fg_patch  (B, 1, R, R, R)  EDT at full-res, cropped, clipped
    click_dist_bg_patch  (B, 1, R, R, R)
    global_tokens        (B, K, token_dim)  pre-extracted per-case
    patch_center_norm    (B, 3)  in [0,1]^3
    memory_tokens        (B, T, token_dim) or None

Forward outputs:
    refined (B, 1, R, R, R), delta (B, 1, R, R, R)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding3D(nn.Module):
    def __init__(self, output_dim: int, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        raw_dim = 3 * 2 * num_freqs
        self.proj = nn.Linear(raw_dim, output_dim)
        freqs = (2.0 ** torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scaled = coords.unsqueeze(-1) * self.freqs
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return self.proj(encoded.flatten(-2))


class PatchRefinementModule(nn.Module):
    def __init__(
        self,
        vista_channels: int = 192,
        c_s: int = 32,
        token_dim: int = 64,
        num_heads: int = 4,
        R: int = 96,
        R_attn: int = 48,
        max_rounds: int = 6,
        num_global_queries: int = 8,
    ):
        super().__init__()
        assert R % R_attn == 0, "R must be divisible by R_attn"
        self.R = R
        self.R_attn = R_attn
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_global_queries = num_global_queries

        in_ch = 5
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_ch, c_s, 3, padding=1),
            nn.InstanceNorm3d(c_s, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(c_s, c_s, 3, padding=1),
            nn.InstanceNorm3d(c_s, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # Learnable-query extraction from F_global
        self.vista_kv = nn.Conv3d(vista_channels, token_dim, 1)
        self.global_queries = nn.Parameter(
            torch.randn(num_global_queries, token_dim) * 0.02)
        self.global_attn = nn.MultiheadAttention(
            token_dim, num_heads, batch_first=True)
        self.norm_gq = nn.LayerNorm(token_dim)
        self.norm_gk = nn.LayerNorm(token_dim)

        # Patch position encoder
        self.pos_encoder = FourierPositionalEncoding3D(token_dim)

        # Per-round memory token (mask+click+round → token_dim)
        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=4, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, 3, stride=4, padding=1),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.max_clicks_per_round = 8
        self.click_mlp = nn.Sequential(
            nn.Linear(self.max_clicks_per_round * 4, 64),
            nn.GELU(),
            nn.Linear(64, 16),
        )
        self.round_emb = nn.Embedding(max_rounds, 16)
        self.memory_proj = nn.Linear(16 + 16 + 16, token_dim)

        # Main cross-attention (Q=patch spatial, KV=[global+pos+memory])
        self.q_proj = nn.Conv3d(c_s, token_dim, 1)
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            token_dim, num_heads, batch_first=True)

        # Fusion + delta head
        self.fuse = nn.Sequential(
            nn.Conv3d(c_s + token_dim, c_s, 3, padding=1),
            nn.InstanceNorm3d(c_s, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(c_s, c_s // 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.delta_head = nn.Conv3d(c_s // 2, 1, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def logit_from_sigmoid(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))

    def extract_global_tokens(self, F_global: torch.Tensor) -> torch.Tensor:
        """F_global: (B, vista_channels, Dg, Hg, Wg) → (B, K, token_dim)."""
        B = F_global.shape[0]
        kv = self.vista_kv(F_global)                       # (B, d, Dg, Hg, Wg)
        kv = kv.flatten(2).transpose(1, 2)                 # (B, L, d)
        Q = self.global_queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.global_attn(
            self.norm_gq(Q), self.norm_gk(kv), self.norm_gk(kv))
        return out                                         # (B, K, d)

    def encode_memory_token(
        self,
        pred_patch: torch.Tensor,       # (B, 1, R, R, R) sigmoid — use refined patch
        clicks: List[Dict],             # clicks THIS round for THIS object (any res norm pos)
        round_idx: int,
    ) -> torch.Tensor:
        B = pred_patch.shape[0]
        device = pred_patch.device

        mask_feat = self.mask_encoder(pred_patch)          # (B, 16)

        max_c = self.max_clicks_per_round
        click_vec = torch.zeros(B, max_c * 4, device=device)
        for ci, c in enumerate(clicks[:max_c]):
            base = ci * 4
            click_vec[:, base + 0] = c['pos_norm'][0]
            click_vec[:, base + 1] = c['pos_norm'][1]
            click_vec[:, base + 2] = c['pos_norm'][2]
            click_vec[:, base + 3] = 1.0 if c.get('fg', True) else -1.0
        click_feat = self.click_mlp(click_vec)             # (B, 16)

        r_idx = torch.tensor(
            [min(round_idx, self.round_emb.num_embeddings - 1)],
            dtype=torch.long, device=device).expand(B)
        r_feat = self.round_emb(r_idx)                     # (B, 16)

        token = self.memory_proj(
            torch.cat([mask_feat, click_feat, r_feat], -1))
        return token                                       # (B, token_dim)

    def forward(
        self,
        pred_nn_k_patch: torch.Tensor,
        pred_prev_patch: torch.Tensor,
        click_dist_fg_patch: torch.Tensor,
        click_dist_bg_patch: torch.Tensor,
        global_tokens: torch.Tensor,          # (B, K, token_dim)
        patch_center_norm: torch.Tensor,      # (B, 3) in [0,1]^3
        memory_tokens: Optional[torch.Tensor],
    ):
        B, _, D, H, W = pred_nn_k_patch.shape
        assert D == H == W == self.R, f"Expected R={self.R}, got ({D},{H},{W})"

        diff = pred_nn_k_patch - pred_prev_patch
        x = torch.cat(
            [pred_nn_k_patch, pred_prev_patch, diff,
             click_dist_fg_patch, click_dist_bg_patch],
            dim=1,
        )
        F_sp = self.spatial_conv(x)                        # (B, c_s, R,R,R)

        pos_token = self.pos_encoder(patch_center_norm).unsqueeze(1)  # (B, 1, d)
        ctx = [global_tokens, pos_token]
        if memory_tokens is not None and memory_tokens.shape[1] > 0:
            ctx.append(memory_tokens)
        context = torch.cat(ctx, dim=1)                    # (B, K+1+T, d)

        R_a = self.R_attn
        Q3 = self.q_proj(F_sp)
        Q3_low = F.interpolate(
            Q3, size=(R_a, R_a, R_a), mode='trilinear', align_corners=False)
        Q_seq = Q3_low.flatten(2).transpose(1, 2)          # (B, R_a^3, d)
        attn_out, _ = self.attn(
            self.norm_q(Q_seq), self.norm_kv(context), context)
        attn_out = attn_out.transpose(1, 2).reshape(
            B, self.token_dim, R_a, R_a, R_a)
        attn_feat = F.interpolate(
            attn_out, size=(self.R, self.R, self.R),
            mode='trilinear', align_corners=False)

        fused = self.fuse(torch.cat([F_sp, attn_feat], dim=1))
        delta = self.delta_head(fused)

        logit_nn = self.logit_from_sigmoid(pred_nn_k_patch)
        refined = torch.sigmoid(logit_nn + delta)
        return refined, delta


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
