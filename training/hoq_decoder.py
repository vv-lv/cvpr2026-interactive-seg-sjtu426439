"""
HOQ-CA (Hierarchical Object Query + Cross-Attention Assembly) Decoder.

Replaces post-hoc assembly (last-wins / max_prob_sigmoid) with a learned
cross-attention decoder that operates on per-object encoder features.

Architecture:
  1. nnInteractive encoder (frozen) runs per-object → multi-scale features
  2. HOQ decoder takes per-object features + click/bbox embeddings
  3. Object queries cross-attend to features, self-attend to each other
  4. Mask head: dot product between refined queries and features

Designed for BraTS-style nested multi-class cases where assembly is the
primary failure mode (80% of AUC<2.0 cases).
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HOQDecoder(nn.Module):
    """Hierarchical Object Query decoder for learned multi-object assembly.

    Takes per-object encoder features from frozen nnInteractive and outputs
    a multi-class segmentation via cross-attention between object queries
    and spatial features.

    Args:
        feat_channels: channels of input features (256 for nnInteractive stage 3)
        hidden_dim: internal dimension for queries and attention
        num_heads: number of attention heads
        num_layers: number of decoder layers (cross-attn + self-attn + FFN)
        max_objects: maximum number of objects (queries)
    """

    def __init__(self, feat_channels=256, hidden_dim=256, num_heads=8,
                 num_layers=3, max_objects=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.num_layers = num_layers

        # Learnable object queries
        self.object_queries = nn.Embedding(max_objects, hidden_dim)

        # Feature projection (if feat_channels != hidden_dim)
        self.feat_proj = nn.Conv3d(feat_channels, hidden_dim, 1) \
            if feat_channels != hidden_dim else nn.Identity()

        # Positional encoding for spatial features (learnable)
        # Will be initialized on first forward based on spatial size
        self.pos_embed = None  # lazy init

        # Decoder layers
        self.self_attn = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm1 = nn.ModuleList()  # after self-attn
        self.norm2 = nn.ModuleList()  # after cross-attn
        self.norm3 = nn.ModuleList()  # after FFN

        for _ in range(num_layers):
            self.self_attn.append(
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
            )
            self.cross_attn.append(
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
            )
            self.ffn.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1),
            ))
            self.norm1.append(nn.LayerNorm(hidden_dim))
            self.norm2.append(nn.LayerNorm(hidden_dim))
            self.norm3.append(nn.LayerNorm(hidden_dim))

        # Mask head: projects queries to spatial masks via dot product
        self.mask_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-object logit bias (helps with class imbalance)
        self.logit_bias = nn.Parameter(torch.zeros(max_objects))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize object queries with small random values
        nn.init.normal_(self.object_queries.weight, std=0.02)

    def _get_pos_embed(self, D, H, W, device):
        """Learnable 3D positional embedding, lazily initialized."""
        if self.pos_embed is None or self.pos_embed.shape[1] != D * H * W:
            self.pos_embed = nn.Parameter(
                torch.randn(1, D * H * W, self.hidden_dim, device=device) * 0.02
            )
        return self.pos_embed

    def forward(self, per_object_features: List[torch.Tensor],
                num_objects: int,
                per_object_logits: Optional[List[torch.Tensor]] = None):
        """
        Args:
            per_object_features: list of N tensors, each (C, D, H, W)
                from nnInteractive encoder stage 3 (256ch @ 24³)
            num_objects: number of actual objects (N <= max_objects)
            per_object_logits: optional list of N tensors, each (2, D', H', W')
                per-object decoder logits (for additional context)

        Returns:
            multi_logits: (N, D, H, W) per-object logits at feature resolution
                Apply argmax(dim=0) for final multi-class mask
        """
        device = per_object_features[0].device
        C, D, H, W = per_object_features[0].shape
        N = num_objects

        # 1. Project and combine per-object features into a shared feature map
        # Strategy: concatenate along a new "object" dimension, then flatten
        # This lets cross-attention see ALL objects' spatial features
        stacked = torch.stack(per_object_features[:N])  # (N, C, D, H, W)

        # Project features
        projected = []
        for i in range(N):
            feat = self.feat_proj(stacked[i:i+1])  # (1, hidden_dim, D, H, W)
            projected.append(feat)
        projected = torch.cat(projected, dim=0)  # (N, hidden_dim, D, H, W)

        # Flatten spatial: (N, hidden_dim, D, H, W) → (1, N*D*H*W, hidden_dim)
        feat_flat = projected.reshape(N, self.hidden_dim, -1)  # (N, hidden_dim, DHW)
        feat_flat = feat_flat.permute(0, 2, 1)  # (N, DHW, hidden_dim)
        feat_flat = feat_flat.reshape(1, N * D * H * W, self.hidden_dim)  # (1, N*DHW, hidden_dim)

        # 2. Object queries
        queries = self.object_queries.weight[:N].unsqueeze(0)  # (1, N, hidden_dim)

        # 3. Decoder layers
        for layer_idx in range(self.num_layers):
            # Self-attention among object queries
            q_res = self.self_attn[layer_idx](queries, queries, queries)[0]
            queries = self.norm1[layer_idx](queries + q_res)

            # Cross-attention: queries attend to spatial features
            q_res = self.cross_attn[layer_idx](queries, feat_flat, feat_flat)[0]
            queries = self.norm2[layer_idx](queries + q_res)

            # FFN
            q_res = self.ffn[layer_idx](queries)
            queries = self.norm3[layer_idx](queries + q_res)

        # 4. Mask prediction via dot product
        mask_queries = self.mask_proj(queries)  # (1, N, hidden_dim)

        # Use the MEAN feature map for dot product (not per-object)
        mean_feat = projected.mean(0, keepdim=True)  # (1, hidden_dim, D, H, W)
        mean_feat_flat = mean_feat.reshape(1, self.hidden_dim, -1)  # (1, hidden_dim, DHW)

        # (1, N, hidden_dim) × (1, hidden_dim, DHW) → (1, N, DHW)
        logits = torch.bmm(mask_queries, mean_feat_flat)
        logits = logits / math.sqrt(self.hidden_dim)  # scale

        # Add per-object bias
        logits = logits + self.logit_bias[:N].unsqueeze(0).unsqueeze(-1)

        # Reshape to spatial
        logits = logits.reshape(N, D, H, W)

        return logits


class HOQAssembly(nn.Module):
    """Complete HOQ assembly pipeline: frozen nnInteractive + HOQ decoder.

    For training: extracts per-object encoder features, runs HOQ decoder.
    For inference: wraps around existing nnInteractive inference session.
    """

    def __init__(self, network, hoq_decoder, feature_stage=3):
        """
        Args:
            network: frozen nnInteractive ResidualEncoderUNet
            hoq_decoder: HOQDecoder instance (trainable)
            feature_stage: which encoder stage to use (3 = 256ch @ 24³)
        """
        super().__init__()
        self.network = network
        self.hoq_decoder = hoq_decoder
        self.feature_stage = feature_stage

        # Freeze nnInteractive
        for param in self.network.parameters():
            param.requires_grad = False

    def extract_features(self, input_8ch: torch.Tensor) -> tuple:
        """Run frozen encoder and return features + decoder logits.

        Args:
            input_8ch: (B, 8, D, H, W) nnInteractive input for ONE object

        Returns:
            features: (B, C, D', H', W') encoder features at target stage
            logits: (B, 2, D, H, W) decoder output logits
        """
        with torch.no_grad():
            skips = self.network.encoder(input_8ch)
            features = skips[self.feature_stage]  # (B, C, D', H', W')
            decoder_out = self.network.decoder(skips)
            if isinstance(decoder_out, (list, tuple)):
                logits = decoder_out[0]  # highest resolution
            else:
                logits = decoder_out
        return features, logits

    def forward(self, image: torch.Tensor,
                interactions_per_object: List[torch.Tensor],
                target_multiclass: Optional[torch.Tensor] = None):
        """
        Args:
            image: (1, 1, D, H, W) preprocessed image
            interactions_per_object: list of N tensors, each (1, 7, D, H, W)
                interaction channels for each object
            target_multiclass: optional (1, 1, D, H, W) multi-class GT
                for loss computation

        Returns:
            multi_logits: (N, D', H', W') per-object logits at feature resolution
            per_obj_logits: list of N tensors, each (1, 2, D, H, W) from nnInt decoder
        """
        N = len(interactions_per_object)

        # Extract per-object features from frozen encoder
        all_features = []
        all_logits = []
        for i in range(N):
            input_8ch = torch.cat([image, interactions_per_object[i]], dim=1)
            features, logits = self.extract_features(input_8ch)
            all_features.append(features[0])  # remove batch dim → (C, D', H', W')
            all_logits.append(logits)

        # Run HOQ decoder
        multi_logits = self.hoq_decoder(all_features, N)  # (N, D', H', W')

        return multi_logits, all_logits


def compute_multiclass_loss(multi_logits, target_multiclass, feature_spatial_shape):
    """Compute multi-class assembly loss.

    Args:
        multi_logits: (N, D', H', W') per-object logits at feature resolution
        target_multiclass: (D, H, W) multi-class GT (0=bg, 1..N=objects)
        feature_spatial_shape: (D', H', W') target spatial size

    Returns:
        loss: scalar
    """
    N = multi_logits.shape[0]
    D_, H_, W_ = feature_spatial_shape

    # Downsample GT to feature resolution
    gt = target_multiclass.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    gt_ds = F.interpolate(gt, size=(D_, H_, W_), mode='nearest')
    gt_ds = gt_ds.squeeze(0).squeeze(0).long()  # (D', H', W')

    # Convert to per-object binary targets
    per_obj_targets = []
    for i in range(N):
        obj_target = (gt_ds == (i + 1)).float()  # (D', H', W')
        per_obj_targets.append(obj_target)
    per_obj_targets = torch.stack(per_obj_targets)  # (N, D', H', W')

    # Dice loss (per-object, then mean)
    pred_sigmoid = torch.sigmoid(multi_logits)  # (N, D', H', W')
    intersection = (pred_sigmoid * per_obj_targets).sum(dim=(1, 2, 3))
    union = pred_sigmoid.sum(dim=(1, 2, 3)) + per_obj_targets.sum(dim=(1, 2, 3))
    dice_per_obj = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
    dice_loss = dice_per_obj.mean()

    # CE loss (multi-class: argmax over objects vs GT)
    # Add background channel (logit=0)
    bg_logit = torch.zeros(1, D_, H_, W_, device=multi_logits.device)
    all_logits = torch.cat([bg_logit, multi_logits], dim=0)  # (N+1, D', H', W')
    all_logits = all_logits.unsqueeze(0)  # (1, N+1, D', H', W')
    gt_ds_batch = gt_ds.unsqueeze(0)  # (1, D', H', W')
    ce_loss = F.cross_entropy(all_logits, gt_ds_batch)

    return dice_loss + ce_loss
