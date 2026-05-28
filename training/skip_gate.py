"""
Skip Connection Gate: multiplicative gating on encoder→decoder skip connections.

Driven by prev_pred confidence (softmax prob), gates control how much new
encoder information flows to the decoder at each resolution level.

High-confidence correct regions → gate low → suppress encoder changes → protect
Low-confidence / error regions → gate high → let encoder changes through → allow fix

Key design: multiplicative (feat × gate) not additive (feat + delta).
Going from gate=1.0 to gate=0.3 suppresses 70% of skip features,
much more powerful than an additive delta of 0.1.
"""
from __future__ import annotations
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.run_bottleneck_attn import _extract_patch_single, PATCH_SIZE


class SkipGate(nn.Module):
    """Per-voxel multiplicative gate on a skip connection."""

    def __init__(self, skip_channels: int, gate_input_channels: int = 1):
        super().__init__()
        self.gate_conv = nn.Conv3d(gate_input_channels, skip_channels, 1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, 2.0)  # sigmoid(2) ≈ 0.88, gradient ≈ 0.10

    def forward(self, skip_feat: torch.Tensor,
                gate_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skip_feat: (B, C, D, H, W) encoder skip features
            gate_input: (B, gate_input_channels, D, H, W) e.g. prev_pred prob
        Returns:
            gated skip features, same shape
        """
        gate = torch.sigmoid(self.gate_conv(gate_input))
        return skip_feat * gate


class GatedDecoder(nn.Module):
    """Wraps the original decoder, inserting SkipGates on selected skip connections.

    Replaces decoder.forward() to intercept skip features before concat.
    """

    def __init__(self, original_decoder: nn.Module,
                 gate_stages: List[int] = (2, 3, 4),
                 gate_input_channels: int = 1):
        super().__init__()
        self.decoder = original_decoder
        self.gate_stages = set(gate_stages)
        self.gate_input_channels = gate_input_channels

        skip_channel_map = {0: 320, 1: 256, 2: 128, 3: 64, 4: 32}
        self.gates = nn.ModuleDict()
        for s in gate_stages:
            self.gates[str(s)] = SkipGate(skip_channel_map[s], gate_input_channels)

        self._prev_pred_prob: Optional[torch.Tensor] = None
        self._bypass = False

    def set_prev_pred_prob(self, prob: Optional[np.ndarray],
                           patch_center: Optional[tuple] = None):
        """Set the prev_pred probability map for gating.

        Args:
            prob: full-resolution softmax prob (float32 or float16), or None
            patch_center: if provided, extract 192³ patch; otherwise use as-is
        """
        if prob is None:
            self._prev_pred_prob = None
            return
        if patch_center is not None:
            prob_patch = _extract_patch_single(
                prob.astype(np.float32), patch_center)
        else:
            prob_patch = prob.astype(np.float32)
        self._prev_pred_prob = torch.from_numpy(prob_patch).unsqueeze(0).unsqueeze(0)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        d = self.decoder

        for s in range(len(d.stages)):
            x = d.transpconvs[s](lres_input)
            skip = skips[-(s + 2)]

            if (not self._bypass and str(s) in self.gates
                    and self._prev_pred_prob is not None):
                target_size = tuple(skip.shape[2:])
                gate_input = F.interpolate(
                    self._prev_pred_prob.to(skip.device),
                    size=target_size, mode='trilinear',
                    align_corners=False)
                if gate_input.shape[0] != skip.shape[0]:
                    gate_input = gate_input.expand(skip.shape[0], -1, -1, -1, -1)
                skip = self.gates[str(s)](skip, gate_input)

            x = torch.cat((x, skip), 1)
            x = d.stages[s](x)
            if d.deep_supervision:
                seg_outputs.append(d.seg_layers[s](x))
            elif s == (len(d.stages) - 1):
                seg_outputs.append(d.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if d.deep_supervision else seg_outputs[0]


if __name__ == '__main__':
    import sys
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test SkipGate
    gate = SkipGate(128, gate_input_channels=1).to(device)
    skip = torch.randn(1, 128, 48, 48, 48, device=device)
    gi = torch.rand(1, 1, 48, 48, 48, device=device)

    out = gate(skip, gi)
    ratio = out.abs().mean() / skip.abs().mean()
    print(f"SkipGate: init gate ≈ sigmoid(5) = {torch.sigmoid(torch.tensor(5.)):.4f}")
    print(f"  output/input ratio: {ratio:.4f} (should be ~0.993)")
    print(f"  params: {sum(p.numel() for p in gate.parameters())}")

    # Test GatedDecoder
    sys.path.insert(0, '.')
    from training.trainer import build_network
    net, _ = build_network(
        '/media/sjtu426/lby_t/seg_models/nninteractive_cache/nninteractive_models'
        '/nnInteractive_v1.0_fold_all/fold_all/checkpoint_final.pth',
        deep_supervision=True)
    net.to(device).eval()

    gd = GatedDecoder(net.decoder, gate_stages=[2, 3, 4]).to(device)
    n_params = sum(p.numel() for p in gd.gates.parameters())
    print(f"\nGatedDecoder params: {n_params}")

    # Bypass mode = original decoder
    gd._bypass = True
    dummy = torch.randn(1, 8, 192, 192, 192, device=device)
    with torch.no_grad():
        skips = net.encoder(dummy)
        out_orig = net.decoder(skips)
        out_bypass = gd(skips)
    diff = (out_orig[0] - out_bypass[0]).abs().max().item()
    print(f"Bypass mode diff: {diff:.8f} (should be 0)")

    # Active mode with prev_pred
    gd._bypass = False
    prob = np.random.rand(192, 192, 192).astype(np.float32)
    gd.set_prev_pred_prob(prob)
    with torch.no_grad():
        out_gated = gd(skips)
    diff_active = (out_orig[0] - out_gated[0]).abs().max().item()
    print(f"Active mode diff: {diff_active:.4f} (should be small, gate ≈ 0.993)")

    # Gradient check
    gd.train()
    for p in net.parameters():
        p.requires_grad_(False)
    out = gd(skips)
    out[0].sum().backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().max() > 0
        for p in gd.gates.parameters()
    )
    print(f"Gate gradient: {has_grad}")
    print("\nAll tests passed." if has_grad else "\nGradient FAILED!")
