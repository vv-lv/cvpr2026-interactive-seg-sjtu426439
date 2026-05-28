#!/usr/bin/env python3
"""
VISTA3D bbox 微调：方案二（冻结训练，只训练 bbox embedding）。

修改 PointMappingSAM：新增 2 个 bbox corner embedding
  - point_label=4: bbox min corner (z_min, y_min, x_min)
  - point_label=5: bbox max corner (z_max, y_max, x_max)

冻结 encoder + 现有 point embeddings + cross-attention，只训练新增的 2 个 embedding。

训练数据：从 GT mask 自动生成 bbox + jitter。
Prompt 采样：bbox-only 60%, bbox+point 30%, point-only 10%

用法:
  conda activate lby_seg_vista3d
  python -u training/vista3d_bbox_finetune.py --epochs 50 --gpu 0
"""
import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import monai
import monai.transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.apps.vista3d.sampler import sample_prompt_pairs
from monai.losses import DiceCELoss
from monai.networks.nets import vista3d132
from torch.utils.data import DataLoader, Dataset

warnings.simplefilter("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "3D_train_npz_random_10percent_16G"
DEFAULT_CKPT = Path(
    "/media/sjtu426/lby_t/seg_models/VISTA/vista3d/cvpr_workshop/"
    "alldata_extracted/model_final.pth"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "vista3d_bbox_ft"

NUM_PATCHES = 4
PATCH_SIZE = 128


# ─── Patch PointMappingSAM to support bbox ──────────────────────────────────

def patch_point_head_for_bbox(model):
    """Add bbox corner embeddings to PointMappingSAM.

    New point_label values:
      4 = bbox min corner (z_min, y_min, x_min) → positive embedding
      5 = bbox max corner (z_max, y_max, x_max) → positive embedding
    """
    point_head = model.point_head

    # Add 2 new embeddings for bbox corners
    dim = point_head.point_embeddings[0].weight.shape[1]
    point_head.bbox_min_embed = nn.Embedding(1, dim).to(
        point_head.point_embeddings[0].weight.device)
    point_head.bbox_max_embed = nn.Embedding(1, dim).to(
        point_head.point_embeddings[0].weight.device)

    # Initialize near zero (so initial behavior ≈ positive point)
    nn.init.normal_(point_head.bbox_min_embed.weight, std=0.01)
    nn.init.normal_(point_head.bbox_max_embed.weight, std=0.01)

    # Save original forward
    original_forward = point_head.forward

    def new_forward(self, out, point_coords, point_labels, class_vector=None):
        """Modified forward that handles bbox labels (4 and 5)."""
        # Remap bbox labels to positive (1) for PE, then add bbox-specific embed
        has_bbox = (point_labels == 4).any() or (point_labels == 5).any()

        if has_bbox:
            # Save bbox masks
            bbox_min_mask = point_labels == 4
            bbox_max_mask = point_labels == 5

            # Temporarily remap to label 1 (positive) for standard PE + embedding
            point_labels_mod = point_labels.clone()
            point_labels_mod[bbox_min_mask] = 1
            point_labels_mod[bbox_max_mask] = 1

            # Call original forward logic manually (up to embedding)
            out_low = self.feat_downsample(out)
            out_shape = tuple(out.shape[-3:])
            out = None
            torch.cuda.empty_cache()

            points = point_coords + 0.5
            point_embedding = self.pe_layer.forward_with_coords(
                points, out_shape)
            point_embedding[point_labels_mod == -1] = 0.0
            point_embedding[point_labels_mod == -1] += \
                self.not_a_point_embed.weight
            point_embedding[point_labels_mod == 0] += \
                self.point_embeddings[0].weight
            point_embedding[point_labels_mod == 1] += \
                self.point_embeddings[1].weight
            point_embedding[point_labels_mod == 2] += \
                self.point_embeddings[0].weight + \
                self.special_class_embed.weight
            point_embedding[point_labels_mod == 3] += \
                self.point_embeddings[1].weight + \
                self.special_class_embed.weight

            # Add bbox-specific embeddings
            point_embedding[bbox_min_mask] += self.bbox_min_embed.weight
            point_embedding[bbox_max_mask] += self.bbox_max_embed.weight

            output_tokens = self.mask_tokens.weight
            output_tokens = output_tokens.unsqueeze(0).expand(
                point_embedding.size(0), -1, -1)

            if class_vector is None:
                tokens_all = torch.cat(
                    (output_tokens, point_embedding,
                     self.supported_embed.weight.unsqueeze(0).expand(
                         point_embedding.size(0), -1, -1)),
                    dim=1)
            else:
                class_embeddings = []
                for i in class_vector:
                    if i > self.last_supported:
                        class_embeddings.append(self.zeroshot_embed.weight)
                    else:
                        class_embeddings.append(self.supported_embed.weight)
                tokens_all = torch.cat(
                    (output_tokens, point_embedding,
                     torch.stack(class_embeddings)), dim=1)

            # Cross attention (same as original)
            masks = []
            max_prompt = self.max_prompt
            for i in range(int(np.ceil(
                    tokens_all.shape[0] / max_prompt))):
                src, upscaled_embedding, hyper_in = None, None, None
                torch.cuda.empty_cache()
                idx = (i * max_prompt,
                       min((i + 1) * max_prompt, tokens_all.shape[0]))
                tokens = tokens_all[idx[0]:idx[1]]
                src = torch.repeat_interleave(
                    out_low, tokens.shape[0], dim=0)
                pos_src = torch.repeat_interleave(
                    self.pe_layer(out_low.shape[-3:]).unsqueeze(0),
                    tokens.shape[0], dim=0)
                b, c, h, w, d = src.shape
                hs, src = self.transformer(src, pos_src, tokens)
                mask_tokens_out = hs[:, :1, :]
                hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
                src = src.transpose(1, 2).view(b, c, h, w, d)
                upscaled_embedding = self.output_upscaling(src)
                b, c, h, w, d = upscaled_embedding.shape
                mask = hyper_in @ upscaled_embedding.view(b, c, h * w * d)
                masks.append(mask.view(-1, 1, h, w, d))

            return torch.vstack(masks)
        else:
            return original_forward(out, point_coords, point_labels,
                                    class_vector)

    import types
    point_head.forward = types.MethodType(new_forward, point_head)
    return model


# ─── Bbox generation from GT ────────────────────────────────────────────────

def generate_bbox_from_mask(mask, jitter_ratio=0.2):
    """Generate a 3D bounding box from binary mask with random jitter.

    Returns: (min_corner, max_corner) each [z, y, x]
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None, None

    mins = coords.min(0)
    maxs = coords.max(0) + 1  # exclusive

    # Add jitter
    sizes = maxs - mins
    for d in range(3):
        jit = int(sizes[d] * jitter_ratio)
        mins[d] = max(0, mins[d] - random.randint(0, jit))
        maxs[d] = min(mask.shape[d], maxs[d] + random.randint(0, jit))

    return mins.tolist(), maxs.tolist()


# ─── Dataset ────────────────────────────────────────────────────────────────

class BboxTrainDataset(Dataset):
    def __init__(self, data_dir, max_files=500):
        data_dir = Path(data_dir)
        all_files = sorted(data_dir.rglob("*.npz"))
        # Filter for files with foreground
        random.shuffle(all_files)
        self.files = all_files[:max_files]
        print(f"BboxTrainDataset: {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        img = torch.from_numpy(
            data['imgs'].astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(
            data['gts'].astype(np.int32)).unsqueeze(0)

        transforms = monai.transforms.Compose([
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys="image", lower=1, upper=99, b_min=0, b_max=1,
                clip=True),
            monai.transforms.SpatialPadd(
                mode=["constant", "constant"],
                keys=["image", "label"],
                spatial_size=[PATCH_SIZE, PATCH_SIZE, PATCH_SIZE]),
            monai.transforms.RandCropByLabelClassesd(
                spatial_size=[PATCH_SIZE, PATCH_SIZE, PATCH_SIZE],
                keys=["image", "label"],
                label_key="label",
                num_classes=label.max().item() + 1,
                num_samples=NUM_PATCHES),
            monai.transforms.RandFlipd(
                spatial_axis=0, prob=0.2, keys=["image", "label"]),
            monai.transforms.RandFlipd(
                spatial_axis=1, prob=0.2, keys=["image", "label"]),
            monai.transforms.RandFlipd(
                spatial_axis=2, prob=0.2, keys=["image", "label"]),
        ])

        out = transforms({"image": img, "label": label})
        return out


# ─── Training ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_files", type=int, default=500)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", default="encoder",
                        choices=["all", "encoder", "none"],
                        help="all=only bbox embed; encoder=bbox+point_head; "
                             "none=everything trainable")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (initially on CPU, split later)
    model = vista3d132(in_channels=1)
    ckpt = torch.load(args.checkpoint, map_location='cpu',
                      weights_only=False)
    # Handle DDP checkpoint
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print("Loaded VISTA3D alldata weights", flush=True)

    # Patch model for bbox support
    model = patch_point_head_for_bbox(model)
    print("Patched PointMappingSAM with bbox embeddings", flush=True)

    # Freeze based on --freeze setting
    if args.freeze == "all":
        # Only bbox embeddings trainable
        for param in model.parameters():
            param.requires_grad = False
        model.point_head.bbox_min_embed.weight.requires_grad = True
        model.point_head.bbox_max_embed.weight.requires_grad = True
    elif args.freeze == "encoder":
        # Freeze image encoder, unfreeze point_head (cross-attention + embeddings)
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        for param in model.class_head.parameters():
            param.requires_grad = False
        for param in model.point_head.parameters():
            param.requires_grad = True
    elif args.freeze == "none":
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # All on single GPU, encoder in fp16 to save memory
    model.image_encoder.half().to(device)
    model.point_head.to(device)
    print(f"Freeze mode: {args.freeze}", flush=True)
    print(f"All on {device}, encoder fp16", flush=True)
    print(f"Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.4f}%)", flush=True)

    # Optimizer
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        train_params, lr=args.lr, weight_decay=1e-5)

    loss_fn = DiceCELoss(
        include_background=False, sigmoid=True,
        smooth_dr=1e-5, smooth_nr=0, squared_pred=True)

    # Dataset: split train/val
    dataset = BboxTrainDataset(args.data_dir, max_files=args.max_files)
    n_val = min(20, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                        num_workers=4, collate_fn=lambda x: x[0])
    print(f"Train: {n_train}, Val: {n_val}", flush=True)

    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, "
          f"bbox embedding only", flush=True)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch in loader:
            # batch is a list of dicts (from RandCropByLabelClassesd)
            if isinstance(batch, dict):
                patch_list = [batch]
            elif isinstance(batch, list) and isinstance(batch[0], dict):
                patch_list = batch
            else:
                continue

            for patch_data in patch_list:
                inputs = patch_data['image'].unsqueeze(0)  # keep on CPU
                labels = patch_data['label'].unsqueeze(0).to(device)

                fg_classes = list(
                    set(labels.unique().tolist()) - {0})
                if not fg_classes:
                    continue

                # Decide prompt type
                r = random.random()
                if r < 0.3:
                    # Bbox only
                    prompt_mode = 'bbox'
                elif r < 0.6:
                    # Bbox + point
                    prompt_mode = 'bbox_point'
                else:
                    # Point only (preserve existing capability)
                    prompt_mode = 'point'

                if prompt_mode == 'point':
                    # Standard point sampling
                    label_prompt, point, point_label, prompt_class = \
                        sample_prompt_pairs(
                            labels, fg_classes,
                            max_point=3, max_prompt=2,
                            drop_label_prob=1, drop_point_prob=0)
                    if point is None:
                        continue
                else:
                    # Generate bbox prompts
                    points_list = []
                    labels_list = []
                    prompt_classes = []

                    # Limit to 2 objects per step (memory)
                    selected_classes = fg_classes[:2] if len(fg_classes) > 2 \
                        else fg_classes
                    for cls in selected_classes:
                        mask_np = (labels[0, 0].cpu().numpy() == cls)
                        bbox_min, bbox_max = generate_bbox_from_mask(
                            mask_np, jitter_ratio=0.2)
                        if bbox_min is None:
                            continue

                        obj_points = []
                        obj_labels = []

                        # Bbox corners
                        obj_points.append(bbox_min)  # [z, y, x]
                        obj_labels.append(4)  # bbox min
                        obj_points.append(bbox_max)
                        obj_labels.append(5)  # bbox max

                        if prompt_mode == 'bbox_point':
                            # Add 1 random fg point inside mask
                            fg_coords = np.argwhere(mask_np)
                            if len(fg_coords) > 0:
                                pt = fg_coords[
                                    random.randint(
                                        0, len(fg_coords) - 1)].tolist()
                                obj_points.append(pt)
                                obj_labels.append(1)  # fg point

                        points_list.append(obj_points)
                        labels_list.append(obj_labels)
                        prompt_classes.append(cls)

                    if not points_list:
                        continue

                    # Pad to same length
                    max_n = max(len(p) for p in points_list)
                    B = len(points_list)
                    point = torch.zeros(B, max_n, 3, dtype=torch.long,
                                        device=device)
                    point_label = torch.full(
                        (B, max_n), -1, dtype=torch.long, device=device)

                    for i in range(B):
                        for j, (pt, lb) in enumerate(
                                zip(points_list[i], labels_list[i])):
                            point[i, j] = torch.tensor(pt)
                            point_label[i, j] = lb

                    prompt_class = prompt_classes

                # Forward: encoder on GPU1 (no_grad), point_head on GPU0
                optimizer.zero_grad()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    features, _ = model.image_encoder(
                        inputs.half().to(device),
                        with_point=True, with_label=False)
                del inputs  # free image tensor immediately
                torch.cuda.empty_cache()
                features = features.detach().float() \
                    .requires_grad_(True)
                outputs = torch.utils.checkpoint.checkpoint(
                    model.point_head, features,
                    point.to(device), point_label.to(device),
                    None,  # class_vector
                    use_reentrant=False)

                # Loss
                loss_val = torch.tensor(0.0, device=device)
                n_valid = 0
                for idx in range(len(prompt_class)):
                    cls = prompt_class[idx]
                    if cls == 0:
                        continue
                    gt = (labels == cls).float()
                    loss_val += loss_fn(outputs[[idx]].float(), gt)
                    n_valid += 1

                if n_valid > 0:
                    loss_val = loss_val / n_valid
                    loss_val.backward()
                    optimizer.step()
                    epoch_losses.append(loss_val.item())

        elapsed = time.time() - t0
        mean_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f"Epoch {epoch:02d}: loss={mean_loss:.4f}, "
              f"steps={len(epoch_losses)}, time={elapsed:.1f}s",
              flush=True)

        # Validation (every 5 epochs): bbox-only Dice on val set
        if (epoch + 1) % 5 == 0 and n_val > 0:
            model.eval()
            val_dices = []
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=0, collate_fn=lambda x: x[0])
            with torch.no_grad():
                for vb in val_loader:
                    if isinstance(vb, dict):
                        vpatches = [vb]
                    elif isinstance(vb, list) and isinstance(vb[0], dict):
                        vpatches = vb
                    else:
                        continue
                    for vp in vpatches[:1]:  # 1 patch per val file
                        vinputs = vp['image'].unsqueeze(0)
                        vlabels = vp['label'].unsqueeze(0).to(device)
                        vfg = list(set(vlabels.unique().tolist()) - {0})
                        if not vfg:
                            continue
                        cls = vfg[0]
                        mask_np = (vlabels[0, 0].cpu().numpy() == cls)
                        bmin, bmax = generate_bbox_from_mask(
                            mask_np, jitter_ratio=0.1)
                        if bmin is None:
                            continue
                        vpt = torch.tensor(
                            [[bmin, bmax]], device=device)
                        vlbl = torch.tensor([[4, 5]], device=device)
                        with torch.cuda.amp.autocast():
                            vfeat, _ = model.image_encoder(
                                vinputs.half().to(device),
                                with_point=True, with_label=False)
                        vout = model.point_head(
                            vfeat.float(), vpt, vlbl)
                        vpred = (vout[0, 0] > 0).float()
                        vgt = (vlabels[0, 0] == cls).float()
                        inter = (vpred * vgt).sum()
                        dice = (2 * inter / (vpred.sum() + vgt.sum()
                                + 1e-5)).item()
                        val_dices.append(dice)
                        del vfeat, vout, vpred, vgt
                        torch.cuda.empty_cache()
            if val_dices:
                vd = np.mean(val_dices)
                print(f"  Val bbox-only Dice: {vd:.3f} "
                      f"({len(val_dices)} patches)", flush=True)
            model.train()

        # Save
        ckpt_data = {
            'epoch': epoch,
            'point_head_state': model.point_head.state_dict(),
            'loss': mean_loss,
        }
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(ckpt_data, args.output_dir / "best.pth")
            print(f"  -> New best loss: {mean_loss:.4f}", flush=True)

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data,
                       args.output_dir / f"epoch{epoch:02d}.pth")

    print(f"\nDone. Best loss: {best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
