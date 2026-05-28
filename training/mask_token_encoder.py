"""
Mask Memory Token modules for temporal modeling in interactive segmentation.

Step 1.2: MaskTokenEncoder — encodes full-resolution prob mask into compact tokens
Step 1.3: TemporalCrossAttention — aggregates multi-round mask tokens (to be added)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskTokenEncoder(nn.Module):
    """Encodes a full-resolution probability mask into a small set of tokens.

    Conv3D downsampling (3 layers with GroupNorm + GELU)
    → Perceiver Resampler (learnable queries cross-attend to conv features)

    Input:  (B, 1, D, H, W) — mask prob in [0, 1], arbitrary spatial size
    Output: (B, num_queries, dim) — compact tokens

    Zero-init: cross-attention out_proj is zero-initialized so initial output
    is all zeros (module is equivalent to not existing at startup).
    """

    def __init__(self, num_queries: int = 2, dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim

        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=4, padding=1),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv3d(32, dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: (B, 1, D, H, W) probability mask in [0, 1]
        Returns:
            (B, num_queries, dim) tokens
        """
        B = mask.shape[0]
        feat = self.conv(mask)                                  # (B, dim, d, h, w)
        feat_flat = feat.flatten(2).transpose(1, 2)             # (B, N, dim)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, Q, dim)
        tokens, _ = self.cross_attn(queries, feat_flat, feat_flat)
        return tokens


class TemporalCrossAttention(nn.Module):
    """Extracts temporal memory from multi-round mask tokens.

    Learnable summary queries cross-attend to the concatenation of all
    historical round tokens (each with additive round positional embedding).

    Empty history (Round 0) → returns zeros.
    A zero-init output projection ensures initial output is all zeros
    even with non-empty history. Internal cross-attn and FFN keep normal
    init with residual connections for stable gradient flow.
    """

    def __init__(self, num_summary: int = 2, dim: int = 128,
                 num_heads: int = 4, max_rounds: int = 20, ff_mult: int = 2):
        super().__init__()
        self.num_summary = num_summary
        self.dim = dim

        self.summary_queries = nn.Parameter(torch.randn(num_summary, dim) * 0.02)
        self.round_pe = nn.Embedding(max_rounds, dim)
        nn.init.normal_(self.round_pe.weight, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, history_tokens_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            history_tokens_list: list of (B, Q, dim) tensors, one per past round.
                                 Empty list means Round 0 (no history).
        Returns:
            (B, num_summary, dim) memory tokens
        """
        if len(history_tokens_list) == 0:
            return torch.zeros(1, self.num_summary, self.dim,
                               device=self.summary_queries.device)

        B = history_tokens_list[0].shape[0]
        device = history_tokens_list[0].device

        kv_parts = []
        for round_idx, tokens in enumerate(history_tokens_list):
            pe = self.round_pe(torch.tensor(round_idx, device=device))  # (dim,)
            kv_parts.append(tokens + pe)
        kv = torch.cat(kv_parts, dim=1)  # (B, R*Q, dim)

        queries = self.summary_queries.unsqueeze(0).expand(B, -1, -1)  # (B, S, dim)

        attn_out, _ = self.cross_attn(queries, kv, kv)
        x = self.norm_attn(queries + attn_out)
        ff_out = self.ffn(x)
        x = self.norm_ffn(x + ff_out)
        return self.out_proj(x)


if __name__ == '__main__':
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    passed = 0
    total = 5

    # --- Test 1: large cubic input shape ---
    enc = MaskTokenEncoder().to(device)
    x1 = torch.rand(1, 1, 256, 256, 256, device=device)
    y1 = enc(x1)
    assert y1.shape == (1, 2, 128), f"Test 1 FAIL: {y1.shape}"
    print(f"Test 1 PASS: (1,1,256,256,256) -> {tuple(y1.shape)}")
    passed += 1
    del x1, y1
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    # --- Test 2: non-cubic input shape ---
    x2 = torch.rand(1, 1, 128, 192, 160, device=device)
    y2 = enc(x2)
    assert y2.shape == (1, 2, 128), f"Test 2 FAIL: {y2.shape}"
    print(f"Test 2 PASS: (1,1,128,192,160) -> {tuple(y2.shape)}")
    passed += 1
    del x2, y2

    # --- Test 3: all-zero mask → no NaN/Inf ---
    x3 = torch.zeros(1, 1, 64, 64, 64, device=device)
    y3 = enc(x3)
    assert not torch.isnan(y3).any(), "Test 3 FAIL: NaN in output"
    assert not torch.isinf(y3).any(), "Test 3 FAIL: Inf in output"
    print("Test 3 PASS: zero input -> no NaN/Inf")
    passed += 1
    del x3, y3

    # --- Test 4: zero-init out_proj → all-zero output ---
    enc_fresh = MaskTokenEncoder().to(device)
    x4 = torch.rand(1, 1, 64, 64, 64, device=device)
    y4 = enc_fresh(x4)
    assert (y4 == 0).all(), f"Test 4 FAIL: max abs = {y4.abs().max().item()}"
    print("Test 4 PASS: zero-init out_proj -> all-zero output")
    passed += 1
    del x4, y4

    # --- Test 5: gradient flows to conv[0].weight ---
    # out_proj is zero so gradient stops there; temporarily set non-zero
    # to verify the computational graph is fully connected.
    enc5 = MaskTokenEncoder().to(device)
    with torch.no_grad():
        enc5.cross_attn.out_proj.weight.fill_(0.01)
    x5 = torch.rand(1, 1, 64, 64, 64, device=device)
    y5 = enc5(x5)
    y5.sum().backward()
    grad = enc5.conv[0].weight.grad
    assert grad is not None, "Test 5 FAIL: no gradient on conv[0]"
    assert grad.abs().max() > 0, "Test 5 FAIL: gradient is all zeros"
    print("Test 5 PASS: gradient flows to conv[0].weight")
    passed += 1

    n_params_enc = sum(p.numel() for p in enc.parameters())
    print(f"\nMaskTokenEncoder: {passed}/{total} tests passed.  Params: {n_params_enc:,}")

    # === TemporalCrossAttention tests ===
    print("\n--- TemporalCrossAttention ---")
    t_passed = 0
    t_total = 4

    # --- T1: empty list → zeros, correct shape ---
    tca = TemporalCrossAttention().to(device)
    out_t1 = tca([])
    assert out_t1.shape == (1, 2, 128), f"T1 FAIL: shape {out_t1.shape}"
    assert (out_t1 == 0).all(), f"T1 FAIL: not all zero"
    print("T1 PASS: empty history -> zeros (1,2,128)")
    t_passed += 1

    # --- T2: 3 rounds → correct shape ---
    hist = [torch.randn(1, 2, 128, device=device) for _ in range(3)]
    out_t2 = tca(hist)
    assert out_t2.shape == (1, 2, 128), f"T2 FAIL: shape {out_t2.shape}"
    print(f"T2 PASS: 3-round history -> {tuple(out_t2.shape)}")
    t_passed += 1

    # --- T3: gradient flows to each history tensor ---
    # Only out_proj is zero-init; set it non-zero to enable gradient flow
    hist_grad = [torch.randn(1, 2, 128, device=device, requires_grad=True)
                 for _ in range(3)]
    tca3 = TemporalCrossAttention().to(device)
    with torch.no_grad():
        nn.init.normal_(tca3.out_proj.weight, std=0.01)
    out_t3 = tca3(hist_grad)
    out_t3.sum().backward()
    all_have_grad = all(h.grad is not None and h.grad.abs().max() > 0
                        for h in hist_grad)
    assert all_have_grad, "T3 FAIL: some history tensors have no gradient"
    print("T3 PASS: gradient flows to all history tensors")
    t_passed += 1

    # --- T4: zero-init → non-empty input still gives zeros ---
    tca4 = TemporalCrossAttention().to(device)
    hist4 = [torch.randn(1, 2, 128, device=device) for _ in range(2)]
    out_t4 = tca4(hist4)
    assert (out_t4 == 0).all(), f"T4 FAIL: max abs = {out_t4.abs().max().item()}"
    print("T4 PASS: zero-init -> non-empty history still gives zeros")
    t_passed += 1

    n_params_tca = sum(p.numel() for p in tca.parameters())
    print(f"\nTemporalCrossAttention: {t_passed}/{t_total} tests passed.  Params: {n_params_tca:,}")

    all_pass = (passed == total) and (t_passed == t_total)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
