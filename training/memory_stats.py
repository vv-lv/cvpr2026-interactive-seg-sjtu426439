"""
Memory Stats Module: hand-crafted temporal statistics injected as spatial residual.

Instead of learning to encode masks into tokens, compute simple statistics
(mean, var, first_pred) from historical mask snapshots and inject them
directly into decoder features at 24³ resolution.

Motivated by diagnostic finding: var in regression regions is ~70,000x higher
than in stable regions — the signal is already there, just needs to be visible
to the decoder.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.run_bottleneck_attn import _extract_patch_single, PATCH_SIZE


def compute_memory_stats(mask_snapshots: List[np.ndarray],
                         patch_center: tuple) -> Optional[torch.Tensor]:
    """Compute hand-crafted stats from historical snapshots, cropped to current patch.

    Args:
        mask_snapshots: list of full-resolution probability masks (np float16)
        patch_center: (z, y, x) center of current 192³ patch
    Returns:
        (1, 3, 192, 192, 192) float32 tensor, or None if no history
    """
    if not mask_snapshots:
        return None

    crops = []
    for snap in mask_snapshots:
        crop = _extract_patch_single(snap.astype(np.float32), patch_center)
        crops.append((crop > 0.5).astype(np.float32))  # binarize: signal [0,1]

    stacked = np.stack(crops, axis=0)  # (T, 192, 192, 192), values {0, 1}

    mean_pred = stacked.mean(axis=0)        # consensus, [0, 1]
    var_pred = stacked.var(axis=0) * 4.0    # instability, scaled to [0, 1]
    first_pred = stacked[0]                 # initial judgment, {0, 1}

    stats = np.stack([mean_pred, var_pred, first_pred], axis=0)  # (3, 192, 192, 192)
    return torch.from_numpy(stats).unsqueeze(0)  # (1, 3, 192, 192, 192)


class MemoryStatsEncoder(nn.Module):
    """Encodes 3-channel memory stats (192³) to decoder resolution (24³)."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),   # 96
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),            # 48
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, feat_dim, 3, stride=2, padding=1),      # 24
            nn.GroupNorm(8, feat_dim), nn.GELU(),
        )
        self.gate = nn.Conv3d(feat_dim, feat_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, decoder_feat: torch.Tensor,
                memory_stats: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            decoder_feat: (B, 256, 24, 24, 24)
            memory_stats: (B, 3, 192, 192, 192) or None
        """
        if memory_stats is None:
            return decoder_feat
        encoded = self.conv(memory_stats)   # (B, 256, 24, 24, 24)
        return decoder_feat + self.gate(encoded)


class StageWithMemoryStatsWrapper(nn.Module):
    """Wraps a decoder stage, applying MemoryStatsEncoder after it."""

    def __init__(self, original_stage: nn.Module,
                 memory_encoder: MemoryStatsEncoder):
        super().__init__()
        self.original_stage = original_stage
        self.memory_encoder = memory_encoder
        self._memory_stats: Optional[torch.Tensor] = None
        self._bypass = False

    def set_memory_stats(self, stats: Optional[torch.Tensor]):
        self._memory_stats = stats

    def forward(self, x):
        out = self.original_stage(x)
        if self._bypass or self._memory_stats is None:
            return out
        return self.memory_encoder(out, self._memory_stats.to(out.device))


def wrap_decoder_stage_memory_stats(
        decoder: nn.Module, stage_idx: int,
        memory_encoder: MemoryStatsEncoder
) -> StageWithMemoryStatsWrapper:
    original = decoder.stages[stage_idx]
    wrapper = StageWithMemoryStatsWrapper(original, memory_encoder)
    decoder.stages[stage_idx] = wrapper
    return wrapper


if __name__ == '__main__':
    import sys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = MemoryStatsEncoder().to(device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"MemoryStatsEncoder params: {n_params:,}")

    feat = torch.randn(1, 256, 24, 24, 24, device=device)
    stats = torch.randn(1, 3, 192, 192, 192, device=device)

    out_none = enc(feat, None)
    assert torch.equal(out_none, feat), "None input should return feat unchanged"
    print("PASS: None → identity")

    out = enc(feat, stats)
    assert out.shape == feat.shape
    diff = (out - feat).abs().max().item()
    print(f"PASS: shape correct, initial delta max={diff:.6f} (zero-init gate)")

    # Gradient check
    enc.train()
    out2 = enc(feat, stats)
    out2.sum().backward()
    conv_grad = enc.conv[0].weight.grad is not None and enc.conv[0].weight.grad.abs().max() > 0
    gate_grad = enc.gate.weight.grad is not None
    print(f"PASS: conv grad={conv_grad}, gate grad={gate_grad}")

    print(f"\nAll tests passed. Params: {n_params:,}")
    sys.exit(0)
