"""
Standalone attention module for Docker inference.

Copied from training/bottleneck_attention.py + training/decoder_attention.py
to avoid any dependency on the training/ directory. All class definitions
EXACTLY match the training code to ensure checkpoint keys load correctly.

Usage:
    from attention_inference import setup_attention, build_token_info_for_object
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──
ROLE_SELF_FG = 0
ROLE_SELF_BG = 1
ROLE_OTHER_FG = 2
ROLE_OTHER_BG = 3
ROLE_BACKGROUND = 4
NUM_ROLES = 5
PATCH_SIZE = 192


# ── Positional Encoding (from bottleneck_attention.py) ──

class FourierPositionalEncoding(nn.Module):
    def __init__(self, output_dim: int, num_freqs: int = 8, input_dim: int = 3):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        raw_dim = input_dim * 2 * num_freqs
        self.proj = nn.Linear(raw_dim, output_dim)
        freqs = (2.0 ** torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freqs', freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scaled = coords.unsqueeze(-1) * self.freqs
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        return self.proj(encoded.flatten(-2))


# ── Attention primitives (from bottleneck_attention.py) ──

class NormalizedMultiheadAttention(nn.Module):
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
        t_norm = self.norm_tsa(tokens)
        tokens = tokens + self.token_self_attn(t_norm, t_norm)
        tokens = tokens + self.cross_t2s(
            self.norm_t2s_q(tokens), self.norm_t2s_kv(spatial))
        tokens = tokens + self.ffn_tok(self.norm_ffn_tok(tokens))
        spatial = spatial + self.cross_s2t(
            self.norm_s2t_q(spatial), self.norm_s2t_kv(tokens))
        spatial = spatial + self.ffn_sp(self.norm_ffn_sp(spatial))
        return tokens, spatial


# ── Decoder Attention Module (from decoder_attention.py) ──

class DecoderAttentionModule(nn.Module):
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

        self.proj_in = nn.Conv3d(input_dim, internal_dim, 1)
        self.spatial_pe = nn.Parameter(
            torch.randn(spatial_size ** 3, internal_dim) * 0.02)

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

        if use_token_gate:
            self.token_gate = nn.Sequential(
                nn.Linear(internal_dim * 3, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        self.bg_tokens = nn.Parameter(
            torch.zeros(num_bg_tokens, internal_dim))

        self.layers = nn.ModuleList([
            AttentionLayer(internal_dim, num_heads) for _ in range(num_layers)
        ])

        if use_softmax_competition:
            self.value_proj = nn.Linear(internal_dim, internal_dim)
            self.softmax_temperature = nn.Parameter(torch.tensor(1.0))

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
        B, C, D, H, W = feat.shape
        device = feat.device
        click_tok = self._encode_clicks(token_info.get('clicks', []), device)

        if not self.use_softmax_competition and click_tok is None:
            return feat

        x = self.proj_in(feat)
        spatial = x.flatten(2).permute(0, 2, 1)
        spatial = spatial + self.spatial_pe.unsqueeze(0)

        parts = []
        if click_tok is not None:
            parts.append(click_tok.expand(B, -1, -1))
        parts.append(self.bg_tokens.unsqueeze(0).expand(B, -1, -1))
        tokens = torch.cat(parts, dim=1)

        for layer in self.layers:
            tokens, spatial = layer(tokens, spatial)

        if self.use_softmax_competition:
            token_values = self.value_proj(tokens)
            scores = torch.bmm(spatial, tokens.transpose(1, 2))
            weights = F.softmax(scores / self.softmax_temperature, dim=-1)
            delta_spatial = torch.bmm(weights, token_values)
            delta_spatial = delta_spatial.permute(0, 2, 1).reshape(
                B, self.internal_dim, D, H, W)
            delta = self.proj_out(delta_spatial)
        else:
            spatial = spatial.permute(0, 2, 1).reshape(
                B, self.internal_dim, D, H, W)
            delta = self.proj_out(spatial)

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


# ── Wrapper ──

class StageWithAttentionWrapper(nn.Module):
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


# ── LoRA ──

class LoRAConv3d(nn.Module):
    def __init__(self, original_conv: nn.Conv3d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self._bypass = False
        for p in self.original_conv.parameters():
            p.requires_grad_(False)
        in_ch = original_conv.in_channels
        out_ch = original_conv.out_channels
        k = original_conv.kernel_size
        s = original_conv.stride
        p = original_conv.padding
        self.lora_A = nn.Conv3d(in_ch, rank, kernel_size=k, stride=s,
                                padding=p, bias=False)
        self.lora_B = nn.Conv3d(rank, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        original_out = self.original_conv(x)
        if self._bypass:
            return original_out
        scale = getattr(self, '_external_scale', None)
        if scale is not None:
            lora_out = self.lora_B(self.lora_A(x)) * (scale / self.rank)
        else:
            lora_out = self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)
        return original_out + lora_out


# ── Position computation ──

def compute_token_pos(click_pos, patch_center, patch_size=PATCH_SIZE,
                      use_relative=False, spacing_dhw=None):
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


# ── Setup utilities ──

def _find_parent(root: nn.Module, dotted_name: str, target: nn.Module):
    parts = dotted_name.split('.')
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_decoder(decoder, target_stages, rank=4, alpha=1.0):
    for stage_idx in target_stages:
        if stage_idx >= len(decoder.stages):
            continue
        stage = decoder.stages[stage_idx]
        targets = []
        for block_name, block in stage.named_modules():
            if isinstance(block, nn.Conv3d) and block.kernel_size[0] > 1:
                parent, attr = _find_parent(stage, block_name, block)
                if parent is not None:
                    targets.append((block_name, block, parent, attr))
        for block_name, block, parent, attr in targets:
            lora_conv = LoRAConv3d(block, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_conv)
            if hasattr(parent, 'all_modules'):
                for idx, mod in enumerate(parent.all_modules):
                    if mod is block:
                        parent.all_modules[idx] = lora_conv
                        break


def set_lora_bypass(model, bypass):
    for module in model.modules():
        if isinstance(module, LoRAConv3d):
            module._bypass = bypass


def setup_attention(session, attn_ckpt_path, device):
    """Load attention checkpoint and wrap the session's decoder.

    Returns (wrapper, use_relative_pos) or (None, False) if no checkpoint.
    """
    if attn_ckpt_path is None or not os.path.exists(attn_ckpt_path):
        return None, False

    ckpt = torch.load(attn_ckpt_path, map_location=device, weights_only=False)

    stage_idx = ckpt.get('stage_idx', 1)
    stage_configs = {0: (320, 12), 1: (256, 24), 2: (128, 48),
                     3: (64, 96), 4: (32, 192)}
    input_dim, spatial_size = stage_configs[stage_idx]

    use_relative_pos = ckpt.get('use_relative_pos', False)
    use_learnable_scale = ckpt.get('use_learnable_scale', False)
    use_lora_scale = ckpt.get('use_lora_scale', False)

    attn_module = DecoderAttentionModule(
        input_dim=input_dim, spatial_size=spatial_size,
        internal_dim=ckpt.get('internal_dim', 128),
        num_layers=ckpt.get('num_layers', 1),
        num_heads=ckpt.get('num_heads', 4),
        num_bg_tokens=ckpt.get('num_bg_tokens', 1),
        use_tanh_gate=ckpt.get('use_tanh_gate', False),
        use_relative_pos=use_relative_pos,
        use_token_gate=ckpt.get('use_token_gate', False),
        use_voxel_gate=ckpt.get('use_voxel_gate', False),
        use_learnable_scale=use_learnable_scale,
        use_lora_scale=use_lora_scale,
    )
    attn_module.load_state_dict(ckpt['attention_state_dict'], strict=True)
    attn_module.to(device).eval()

    lora_state = ckpt.get('lora_state_dict', {})
    if lora_state:
        stages = sorted(set(int(k.split('.stages.')[1].split('.')[0])
                            for k in lora_state.keys()))
        lora_rank = ckpt.get('lora_rank', 4)
        apply_lora_to_decoder(session.network.decoder,
                              target_stages=stages, rank=lora_rank)
        session.network.decoder.to(device)
        for name, module in session.network.named_modules():
            if isinstance(module, LoRAConv3d):
                for key, val in lora_state.items():
                    if name in key:
                        if 'lora_A' in key:
                            module.lora_A.weight.data.copy_(val.to(device))
                        elif 'lora_B' in key:
                            module.lora_B.weight.data.copy_(val.to(device))

        if use_lora_scale:
            lora_s = torch.sigmoid(attn_module.lora_scale_param) * 2
            for m in session.network.modules():
                if isinstance(m, LoRAConv3d):
                    m._external_scale = lora_s

    wrapper = StageWithAttentionWrapper(
        session.network.decoder.stages[stage_idx], attn_module)
    session.network.decoder.stages[stage_idx] = wrapper

    return wrapper, use_relative_pos


def build_token_info_for_object(session, oid, num_objects, clicks, clicks_order,
                                 wrapper, use_relative_pos=False, spacing_dhw=None):
    """Build token_info and set bypass for one object during inference.

    For no-bbox cases only. Bbox cases should bypass entirely.
    """
    if wrapper is None:
        return

    center = (session.new_interaction_centers[-1]
              if session.new_interaction_centers else None)
    bbox_start = [0, 0, 0]
    if hasattr(session, 'preprocessed_props'):
        bbox = session.preprocessed_props.get('bbox_used_for_cropping', None)
        if bbox is not None:
            bbox_start = [int(b[0]) for b in bbox]

    if center is None:
        wrapper._bypass = True
        set_lora_bypass(session.network, True)
        wrapper.set_token_info({'clicks': []})
        return

    self_tokens = []
    if clicks is not None and oid - 1 < len(clicks):
        clicks_here = clicks[oid - 1]
        order_here = clicks_order[oid - 1] if clicks_order is not None else []
        fg_ptr = bg_ptr = 0
        for ri, kind in enumerate(order_here):
            if kind is None:
                continue
            if kind == 'fg':
                click = clicks_here['fg'][fg_ptr]; fg_ptr += 1
                role = ROLE_SELF_FG
            else:
                click = clicks_here['bg'][bg_ptr]; bg_ptr += 1
                role = ROLE_SELF_BG
            click_pre = [click[d] - bbox_start[d] for d in range(3)]
            if use_relative_pos:
                pos = compute_token_pos(click_pre, center, PATCH_SIZE,
                                        use_relative=True, spacing_dhw=spacing_dhw)
            else:
                pos = compute_token_pos(click_pre, center, PATCH_SIZE,
                                        use_relative=False).clamp(0, 1)
            self_tokens.append({'pos': pos, 'role': role, 'round': ri})

    other_tokens = []
    for j in range(num_objects):
        if j == oid - 1:
            continue
        if clicks is None or j >= len(clicks):
            continue
        clicks_j = clicks[j]
        order_j = clicks_order[j] if clicks_order is not None else []
        fg_ptr = bg_ptr = 0
        for ri, kind in enumerate(order_j):
            if kind is None:
                continue
            if kind == 'fg':
                click = clicks_j['fg'][fg_ptr]; fg_ptr += 1
                role = ROLE_OTHER_FG
            else:
                click = clicks_j['bg'][bg_ptr]; bg_ptr += 1
                role = ROLE_OTHER_BG
            click_pre = [click[d] - bbox_start[d] for d in range(3)]
            pos_abs = compute_token_pos(click_pre, center, PATCH_SIZE,
                                        use_relative=False)
            if not all(0.0 <= x <= 1.0 for x in pos_abs.tolist()):
                continue
            if use_relative_pos:
                pos = compute_token_pos(click_pre, center, PATCH_SIZE,
                                        use_relative=True, spacing_dhw=spacing_dhw)
            else:
                pos = pos_abs
            other_tokens.append({'pos': pos, 'role': role, 'round': ri})

    has_other = len(other_tokens) > 0

    if has_other:
        wrapper._bypass = False
        set_lora_bypass(session.network, False)
        if hasattr(wrapper.attention, 'use_lora_scale') \
                and wrapper.attention.use_lora_scale:
            lora_s = torch.sigmoid(wrapper.attention.lora_scale_param) * 2
            for m in session.network.modules():
                if isinstance(m, LoRAConv3d):
                    m._external_scale = lora_s
        wrapper.set_token_info({'clicks': self_tokens + other_tokens})
    else:
        wrapper._bypass = True
        set_lora_bypass(session.network, True)
        wrapper.set_token_info({'clicks': []})
