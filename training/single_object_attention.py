"""
Single-Object Attention Module with Mask Memory.

Stage 1 validation: inject temporal mask memory into decoder attention,
independently of v7's multi-object (other_tokens) logic.

Reuses v7's click encoding and AttentionLayer; adds MaskTokenEncoder +
TemporalCrossAttention for cross-round memory.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.bottleneck_attention import (
    FourierPositionalEncoding, AttentionLayer,
    ROLE_SELF_FG, ROLE_SELF_BG, NUM_ROLES,
)
from training.mask_token_encoder import MaskTokenEncoder, TemporalCrossAttention


class SingleObjectAttentionModule(nn.Module):
    """Decoder attention with mask memory for single-object scenarios.

    Token pool: [click_tokens, bg_tokens, memory_tokens(optional)]
    No other_tokens — single object only.

    Bypass: when no mask history is available (Round 0), returns feat unchanged.
    """

    MASK_ENCODE_SIZE = 128

    def __init__(self, input_dim: int = 256, spatial_size: int = 24,
                 internal_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, num_bg_tokens: int = 4,
                 num_fourier_freqs: int = 8, max_rounds: int = 6,
                 memory_queries: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.internal_dim = internal_dim
        self.spatial_size = spatial_size

        self.proj_in = nn.Conv3d(input_dim, internal_dim, 1)

        self.spatial_pe = nn.Parameter(
            torch.randn(spatial_size ** 3, internal_dim) * 0.02)

        self.fourier_pe = FourierPositionalEncoding(internal_dim, num_fourier_freqs)
        self.role_emb = nn.Embedding(NUM_ROLES, internal_dim)
        self.temporal_emb = nn.Embedding(max_rounds, internal_dim)
        self.click_proj = nn.Sequential(
            nn.Linear(internal_dim * 3, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
        )

        self.bg_tokens = nn.Parameter(
            torch.randn(num_bg_tokens, internal_dim) * 0.02)

        self.mask_encoder = MaskTokenEncoder(
            num_queries=memory_queries, dim=internal_dim)
        self.temporal_model = TemporalCrossAttention(
            num_summary=memory_queries, dim=internal_dim)

        self.layers = nn.ModuleList([
            AttentionLayer(internal_dim, num_heads) for _ in range(num_layers)
        ])

        self.proj_out = nn.Conv3d(internal_dim, input_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

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
        return self.click_proj(combined).unsqueeze(0)

    def _encode_mask_history(self, mask_snapshots: List[np.ndarray],
                             device: torch.device
                             ) -> Optional[torch.Tensor]:
        if not mask_snapshots:
            return None
        S = self.MASK_ENCODE_SIZE
        tokens_per_round = []
        for snap in mask_snapshots:
            mask_t = torch.from_numpy(
                snap.astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)
            mask_t = F.interpolate(
                mask_t, size=(S, S, S), mode='trilinear',
                align_corners=False)                 # (1, 1, S, S, S)
            tokens = self.mask_encoder(mask_t)       # (1, Q, dim)
            tokens_per_round.append(tokens)
        return self.temporal_model(tokens_per_round)  # (1, S, dim)

    def forward(self, feat: torch.Tensor, token_info: Dict,
                mask_snapshots: Optional[List[np.ndarray]] = None,
                bbox_normalized: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            feat: (B, input_dim, S, S, S) decoder stage output
            token_info: {'clicks': list of click dicts}
            mask_snapshots: list of prev rounds' full-mask prob (np float16)
            bbox_normalized: reserved for FiLM (Phase 2)
        """
        B, C, D, H, W = feat.shape
        device = feat.device

        memory_tok = self._encode_mask_history(mask_snapshots or [], device)

        if memory_tok is None:
            return feat

        click_tok = self._encode_clicks(token_info.get('clicks', []), device)

        x = self.proj_in(feat)
        spatial = x.flatten(2).permute(0, 2, 1)
        spatial = spatial + self.spatial_pe.unsqueeze(0)

        parts = []
        if click_tok is not None:
            parts.append(click_tok.expand(B, -1, -1))
        parts.append(self.bg_tokens.unsqueeze(0).expand(B, -1, -1))
        parts.append(memory_tok.expand(B, -1, -1))
        tokens = torch.cat(parts, dim=1)

        for layer in self.layers:
            tokens, spatial = layer(tokens, spatial)

        spatial = spatial.permute(0, 2, 1).reshape(B, self.internal_dim, D, H, W)
        delta = self.proj_out(spatial)
        return feat + delta


class StageWithSingleObjAttentionWrapper(nn.Module):
    """Wraps a decoder stage, applying SingleObjectAttentionModule after it."""

    def __init__(self, original_stage: nn.Module,
                 attention: SingleObjectAttentionModule):
        super().__init__()
        self.original_stage = original_stage
        self.attention = attention
        self._token_info: Dict = {'clicks': []}
        self._mask_snapshots: Optional[List[np.ndarray]] = None
        self._bypass = False

    def set_state(self, token_info: Dict,
                  mask_snapshots: Optional[List[np.ndarray]] = None):
        self._token_info = token_info
        self._mask_snapshots = mask_snapshots

    def forward(self, x):
        out = self.original_stage(x)
        if self._bypass:
            return out
        return self.attention(out, self._token_info,
                              mask_snapshots=self._mask_snapshots)


def wrap_decoder_stage_single_obj(
        decoder: nn.Module, stage_idx: int,
        attention: SingleObjectAttentionModule
) -> StageWithSingleObjAttentionWrapper:
    original = decoder.stages[stage_idx]
    wrapper = StageWithSingleObjAttentionWrapper(original, attention)
    decoder.stages[stage_idx] = wrapper
    return wrapper


if __name__ == '__main__':
    import sys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    passed = 0
    total = 4

    # --- T1: instantiation ---
    mod = SingleObjectAttentionModule().to(device)
    n_params = sum(p.numel() for p in mod.parameters())
    print(f"T1 PASS: instantiated, {n_params:,} params")
    passed += 1

    # --- T2: Round 0 bypass (no mask history) ---
    feat = torch.randn(1, 256, 24, 24, 24, device=device)
    token_info = {'clicks': [{
        'pos': torch.tensor([0.5, 0.5, 0.5]),
        'role': ROLE_SELF_FG, 'round': 0,
    }]}
    out = mod(feat, token_info, mask_snapshots=None)
    assert torch.equal(out, feat), "T2 FAIL: bypass didn't return identical feat"
    print("T2 PASS: Round 0 bypass (output == input)")
    passed += 1

    # --- T3: Round 1+ with mock mask_snapshots ---
    mock_snaps = [np.random.rand(48, 48, 48).astype(np.float16) for _ in range(2)]
    token_info_r2 = {'clicks': [
        {'pos': torch.tensor([0.3, 0.4, 0.5]), 'role': ROLE_SELF_FG, 'round': 0},
        {'pos': torch.tensor([0.6, 0.7, 0.2]), 'role': ROLE_SELF_BG, 'round': 1},
    ]}
    out2 = mod(feat, token_info_r2, mask_snapshots=mock_snaps)
    assert out2.shape == feat.shape, f"T3 FAIL: shape {out2.shape}"
    assert not torch.isnan(out2).any(), "T3 FAIL: NaN"
    print(f"T3 PASS: Round 2 with 2 snapshots, shape {tuple(out2.shape)}")
    passed += 1

    # --- T4: proj_out zero-init → delta is zero ---
    diff = (out2 - feat).abs().max().item()
    print(f"T4 INFO: max |output - input| = {diff:.6e} (should be ~0 due to proj_out zero-init)")
    assert diff < 1e-5, f"T4 FAIL: diff too large: {diff}"
    print("T4 PASS: initial output ≈ input (proj_out zero-init)")
    passed += 1

    print(f"\nAll {passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)
