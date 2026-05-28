#!/usr/bin/env python3
"""
VISTA3D fine-tuning: decoder-end injection of bbox + prev_pred.

Architecture: frozen encoder + SpatialPromptEncoder injected at decoder output.
Input: 3ch [image, bbox_mask, prev_pred], split inside forward.

Usage:
  python training/vista3d_finetune.py --sanity_check --gpu 1
  python training/vista3d_finetune.py --epochs 5 --max_files 500 --gpu 1
  python training/vista3d_finetune.py --epochs 20 --gpu 1  # full 44K
"""
from __future__ import annotations

import argparse
import random
import time
import types
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
from monai.networks.nets.vista3d import vista3d132
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_ROOT = Path("/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all")
DEFAULT_PRETRAINED = Path("/media/sjtu426/lby_t/seg_models/VISTA/vista3d/cvpr_workshop/alldata_extracted/model_final.pth")
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "vista3d_finetune"

ROI_SIZE = [128, 128, 128]
NUM_PATCHES_PER_IMAGE = 4


# ─── SpatialPromptEncoder ────────────────────────────────────────────────────
class SpatialPromptEncoder(nn.Module):
    """Encode bbox_mask + prev_pred (2ch) → out_low feature space (48ch@64³).
    Zero-init last layer → initial output = 0.
    """
    def __init__(self, in_ch=2, feat_ch=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, feat_ch, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# ─── Model setup ─────────────────────────────────────────────────────────────
def build_model(pretrained_path: Path, device: torch.device = None):
    """Build VISTA3D + SpatialPromptEncoder injected inside point_head."""
    model = vista3d132(in_channels=1)
    ckpt = torch.load(str(pretrained_path), map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    # Add prompt encoder to point_head (so it's inside the right module)
    model.point_head.prompt_encoder = SpatialPromptEncoder(in_ch=2, feat_ch=48)
    model.point_head._prompt_mask = None  # will be set before forward

    # Freeze encoder
    for p in model.image_encoder.encoder.parameters():
        p.requires_grad = False

    # Patch point_head.forward: inject prompt_embed into out_low
    original_ph_forward = model.point_head.forward

    def patched_ph_forward(self, out, point_coords, point_labels, class_vector=None):
        # Step 1: feat_downsample (frozen, same as original)
        out_low = self.feat_downsample(out)
        out_shape = tuple(out.shape[-3:])
        out = None
        torch.cuda.empty_cache()

        # Step 2: inject prompt embedding into out_low
        if self._prompt_mask is not None:
            pm = self._prompt_mask  # [1, 2, H, W, D]
            # Downsample to out_low resolution
            pm_low = F.interpolate(pm, size=out_low.shape[-3:], mode='trilinear', align_corners=False)
            prompt_embed = self.prompt_encoder(pm_low)  # [1, 48, h, w, d]
            out_low = out_low + prompt_embed

        # Step 3: rest of original point_head forward (point embedding + cross-attention)
        import numpy as np_
        points = point_coords + 0.5
        point_embedding = self.pe_layer.forward_with_coords(points, out_shape)
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        point_embedding[point_labels == 2] += self.point_embeddings[0].weight + self.special_class_embed.weight
        point_embedding[point_labels == 3] += self.point_embeddings[1].weight + self.special_class_embed.weight
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(point_embedding.size(0), -1, -1)
        if class_vector is None:
            tokens_all = torch.cat((
                output_tokens, point_embedding,
                self.supported_embed.weight.unsqueeze(0).expand(point_embedding.size(0), -1, -1),
            ), dim=1)
        else:
            class_embeddings = []
            for i in class_vector:
                if i > self.last_supported:
                    class_embeddings.append(self.zeroshot_embed.weight)
                else:
                    class_embeddings.append(self.supported_embed.weight)
            tokens_all = torch.cat((output_tokens, point_embedding, torch.stack(class_embeddings)), dim=1)

        masks = []
        max_prompt = self.max_prompt
        for i in range(int(np_.ceil(tokens_all.shape[0] / max_prompt))):
            src, upscaled_embedding, hyper_in = None, None, None
            torch.cuda.empty_cache()
            idx = (i * max_prompt, min((i + 1) * max_prompt, tokens_all.shape[0]))
            tokens = tokens_all[idx[0]:idx[1]]
            src = torch.repeat_interleave(out_low, tokens.shape[0], dim=0)
            pos_src = torch.repeat_interleave(
                self.pe_layer(out_low.shape[-3:]).unsqueeze(0), tokens.shape[0], dim=0)
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

    model.point_head.forward = types.MethodType(patched_ph_forward, model.point_head)

    # Patch model.forward to split 3ch input and set prompt_mask
    original_model_forward = model.forward.__func__ if hasattr(model.forward, '__func__') else model.forward

    def patched_model_forward(self, input_images, **kwargs):
        if input_images.shape[1] == 3:
            kwargs_new = dict(kwargs)
            kwargs_new.pop('input_images', None)
            image = input_images[:, :1]
            self.point_head._prompt_mask = input_images[:, 1:]
            return original_model_forward(self, image, **kwargs_new)
        else:
            self.point_head._prompt_mask = None
            return original_model_forward(self, input_images, **kwargs)

    model.forward = types.MethodType(patched_model_forward, model)

    if device is not None:
        model = model.to(device)
    return model


def build_param_groups(model, base_lr: float):
    """Trainable: prompt_encoder + up_layers + point_head."""
    param_groups = []
    for p in model.parameters():
        p.requires_grad = False

    # 1. SpatialPromptEncoder inside point_head (highest LR)
    for p in model.point_head.prompt_encoder.parameters():
        p.requires_grad = True
    param_groups.append({
        'params': list(model.point_head.prompt_encoder.parameters()),
        'lr': base_lr * 10, 'name': 'prompt_encoder'
    })

    # 2. Interactive decoder
    for p in model.image_encoder.up_layers.parameters():
        p.requires_grad = True
    param_groups.append({
        'params': list(model.image_encoder.up_layers.parameters()),
        'lr': base_lr, 'name': 'up_layers'
    })

    # 3. Point head (excluding prompt_encoder which is already in group 1)
    pe_param_ids = {id(p) for p in model.point_head.prompt_encoder.parameters()}
    ph_params = [p for p in model.point_head.parameters() if id(p) not in pe_param_ids]
    for p in ph_params:
        p.requires_grad = True
    param_groups.append({
        'params': ph_params,
        'lr': base_lr, 'name': 'point_head'
    })

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']:25s}: {n:>10,} params, lr={pg['lr']:.2e}")
    return param_groups


# ─── Prompt generation ───────────────────────────────────────────────────────
def generate_bbox_mask(gt, oid):
    """Generate bbox mask following competition evaluation protocol (get_boxes.py).

    1. z-axis: exact GT z_min/z_max, no jitter
    2. xy-axis: 2D bbox on z_middle slice, outward-only jitter (randint(0,6) * scale/256)
    3. Render as 3D mask: z_min:z_max × y_min:y_max × x_min:x_max
    """
    obj = (gt == oid)
    coords = np.argwhere(obj)
    if len(coords) == 0:
        return np.zeros(gt.shape, dtype=np.float32)

    D, H, W = gt.shape

    # z-axis: exact GT boundary
    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()

    # z_middle: median z slice (same as competition)
    z_indices = np.unique(coords[:, 0])
    z_middle = z_indices[len(z_indices) // 2]

    # 2D bbox on z_middle slice
    gt_mid = (gt[z_middle] == oid)
    if gt_mid.sum() == 0:
        # Fallback: use full 3D bbox
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_min, x_max = coords[:, 2].min(), coords[:, 2].max()
    else:
        y_indices, x_indices = np.where(gt_mid)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

    # Outward-only jitter (matching competition: randint(0,6) * scale/256)
    bbox_shift = np.random.randint(0, 6)
    bbox_shift_x = int(bbox_shift * W / 256)
    bbox_shift_y = int(bbox_shift * H / 256)
    x_min = max(0, x_min - bbox_shift_x)
    x_max = min(W - 1, x_max + bbox_shift_x)
    y_min = max(0, y_min - bbox_shift_y)
    y_max = min(H - 1, y_max + bbox_shift_y)

    # Render 3D mask
    mask = np.zeros(gt.shape, dtype=np.float32)
    mask[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1] = 1.0
    return mask


def generate_prev_pred(gt, oid, all_oids, wrong_prob=0.25):
    if random.random() < 0.1:
        return np.zeros(gt.shape, dtype=np.float32)
    if random.random() < wrong_prob and len(all_oids) > 1:
        wrong_oid = random.choice([o for o in all_oids if o != oid])
        obj = (gt == wrong_oid).astype(np.float32)
    else:
        obj = (gt == oid).astype(np.float32)
        iters = random.randint(1, 5)
        if random.random() > 0.5:
            obj = binary_dilation(obj, iterations=iters).astype(np.float32)
        else:
            obj = binary_erosion(obj, iterations=iters).astype(np.float32)
        obj = gaussian_filter(obj, sigma=random.uniform(1.0, 3.0))
    return np.clip(obj, 0, 1).astype(np.float32)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class Vista3DFinetuneDataset(Dataset):
    def __init__(self, file_list, bbox_drop=0.4, prev_drop=0.4):
        self.file_list = file_list
        self.bbox_drop = bbox_drop
        self.prev_drop = prev_drop

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz = np.load(self.file_list[idx], allow_pickle=True)
        img = torch.from_numpy(npz['imgs'].astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(npz['gts'].astype(np.int32)).unsqueeze(0)

        unique_labels = list(set(label.unique().tolist()) - {0})
        if not unique_labels:
            unique_labels = [1]
        oid = random.choice(unique_labels)
        gt_np = npz['gts']

        bbox = np.zeros(gt_np.shape, np.float32) if random.random() < self.bbox_drop else generate_bbox_mask(gt_np, oid)
        prev = np.zeros(gt_np.shape, np.float32) if random.random() < self.prev_drop else generate_prev_pred(gt_np, oid, unique_labels)

        bbox_t = torch.from_numpy(bbox).unsqueeze(0)
        prev_t = torch.from_numpy(prev).unsqueeze(0)

        all_keys = ["image", "label", "bbox_mask", "prev_pred"]
        data = {"image": img, "label": label, "bbox_mask": bbox_t, "prev_pred": prev_t}

        transforms = monai.transforms.Compose([
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys="image", lower=1, upper=99, b_min=0, b_max=1, clip=True),
            monai.transforms.SpatialPadd(
                keys=all_keys, spatial_size=ROI_SIZE, mode=["constant"] * 4),
            monai.transforms.RandCropByLabelClassesd(
                keys=all_keys, label_key="label", spatial_size=ROI_SIZE,
                num_classes=max(unique_labels) + 1, num_samples=NUM_PATCHES_PER_IMAGE),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
            monai.transforms.RandGaussianNoised(keys="image", mean=0, std=0.2, prob=0.2),
            monai.transforms.RandFlipd(keys=all_keys, spatial_axis=0, prob=0.2),
            monai.transforms.RandFlipd(keys=all_keys, spatial_axis=1, prob=0.2),
            monai.transforms.RandFlipd(keys=all_keys, spatial_axis=2, prob=0.2),
            monai.transforms.RandRotate90d(keys=all_keys, max_k=3, prob=0.2),
        ])
        return transforms(data)


# ─── Training ────────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    model.image_encoder.encoder.eval()
    scaler = torch.amp.GradScaler('cuda')
    total_loss, n_steps = 0.0, 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
        image_l, label_l = batch["image"], batch["label"]
        bbox_l, prev_l = batch["bbox_mask"], batch["prev_pred"]

        for k in range(image_l.shape[0]):
            img = image_l[[k]].to(device)
            lbl = label_l[[k]].to(device)
            bbox_m = bbox_l[[k]].to(device)
            prev_m = prev_l[[k]].to(device)

            # 3ch input: [image, bbox, prev]
            input_3ch = torch.cat([img, bbox_m, prev_m], dim=1)

            # Per-object: find target from bbox overlap or random
            target_oid = None
            if bbox_m.sum() > 0:
                bbox_region = (bbox_m[0, 0] > 0.5)
                lbl_in_bbox = lbl[0, 0][bbox_region]
                counts = torch.bincount(lbl_in_bbox.long().flatten())
                counts[0] = 0
                if counts.sum() > 0:
                    target_oid = counts.argmax().item()
            if target_oid is None or target_oid == 0:
                label_set = list(set(lbl.unique().tolist()) - {0})
                if not label_set:
                    continue
                target_oid = random.choice(label_set)

            _, point, point_label, prompt_class = sample_prompt_pairs(
                lbl, [target_oid], max_point=5, max_prompt=1,
                drop_label_prob=1, drop_point_prob=0)
            if point is None:
                continue

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(input_images=input_3ch,
                                point_coords=point, point_labels=point_label)
                gt_bin = (lbl == target_oid).float()
                loss = loss_fn(outputs[[0]].float(), gt_bin)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 12.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_steps += 1

            del outputs, input_3ch
            torch.cuda.empty_cache()

    return total_loss / max(n_steps, 1)


# ─── Sanity check ────────────────────────────────────────────────────────────
def run_sanity_checks(model, device, n_samples=20):
    model.eval()
    print("\n=== Sanity Checks ===")

    files = sorted(str(p) for p in TRAIN_DATA_ROOT.rglob("*.npz"))[:200]
    random.seed(42)
    random.shuffle(files)

    results = {'point_only': [], 'bbox': [], 'prev_gt': []}

    for fi in range(min(n_samples, len(files))):
        try:
            npz = np.load(files[fi], allow_pickle=True)
        except Exception:
            continue
        gt, img_np = npz['gts'], npz['imgs'].astype(np.float32)
        unique = list(set(np.unique(gt).tolist()) - {0})
        if not unique:
            continue
        oid = unique[0]
        coords = np.argwhere(gt == oid)
        if len(coords) < 10:
            continue

        # Center crop 128³
        center = coords.mean(0).astype(int)
        slices = []
        for d in range(3):
            lo = max(0, center[d] - 64)
            hi = lo + 128
            if hi > gt.shape[d]:
                hi = gt.shape[d]; lo = max(0, hi - 128)
            slices.append(slice(lo, hi))
        img_c = img_np[slices[0], slices[1], slices[2]]
        gt_c = gt[slices[0], slices[1], slices[2]]
        pad = [(0, max(0, 128 - img_c.shape[d])) for d in range(3)]
        if any(p[1] > 0 for p in pad):
            img_c = np.pad(img_c, pad); gt_c = np.pad(gt_c, pad)

        img_t = torch.from_numpy(img_c).unsqueeze(0).unsqueeze(0).float()
        p1, p99 = img_t.quantile(0.01), img_t.quantile(0.99)
        if p99 > p1:
            img_t = ((img_t - p1) / (p99 - p1)).clamp(0, 1)
        gt_t = torch.from_numpy(gt_c.astype(np.int32)).unsqueeze(0).unsqueeze(0).to(device)
        gt_bin = (gt_t == oid).float()
        if gt_bin.sum() == 0:
            continue

        fg = torch.argwhere(gt_bin[0, 0] > 0)
        pt = fg[len(fg) // 2]
        point = torch.tensor([[[pt[0], pt[1], pt[2]]]], device=device)
        plabel = torch.tensor([[1]], device=device)
        zeros = torch.zeros_like(img_t)

        with torch.no_grad():
            # 1) point only
            inp = torch.cat([img_t, zeros, zeros], dim=1).to(device)
            pred = (model(input_images=inp, point_coords=point, point_labels=plabel)[0:1] > 0).float()
            results['point_only'].append((2*(pred*gt_bin).sum()/(pred.sum()+gt_bin.sum()+1e-8)).item())

            # 2) bbox
            bbox_m = torch.zeros_like(img_t)
            mins, maxs = fg.min(0).values, fg.max(0).values
            bbox_m[0, 0, mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1] = 1.0
            inp = torch.cat([img_t, bbox_m, zeros], dim=1).to(device)
            pred = (model(input_images=inp, point_coords=point, point_labels=plabel)[0:1] > 0).float()
            results['bbox'].append((2*(pred*gt_bin).sum()/(pred.sum()+gt_bin.sum()+1e-8)).item())

            # 3) prev = GT
            inp = torch.cat([img_t, bbox_m, gt_bin.cpu()], dim=1).to(device)
            pred = (model(input_images=inp, point_coords=point, point_labels=plabel)[0:1] > 0).float()
            results['prev_gt'].append((2*(pred*gt_bin).sum()/(pred.sum()+gt_bin.sum()+1e-8)).item())

    for k, v in results.items():
        if v:
            print(f"  {k:15s}: mean DSC = {np.mean(v):.4f} +/- {np.std(v):.4f} (n={len(v)})")
    mp = np.mean(results['point_only']) if results['point_only'] else 0
    mb = np.mean(results['bbox']) if results['bbox'] else 0
    mg = np.mean(results['prev_gt']) if results['prev_gt'] else 0
    print(f"\n  [{'OK' if mp > 0.3 else 'FAIL'}] point_only={mp:.3f} > 0.3")
    print(f"  [{'OK' if abs(mb-mp)<0.02 else 'WARN'}] bbox={mb:.3f} ~ point {mp:.3f}")
    print(f"  [{'OK' if abs(mg-mp)<0.02 else 'WARN'}] prev=GT={mg:.3f} ~ point {mp:.3f} (upper bound after training)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--base_lr", type=float, default=2e-5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--sanity_check", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bbox_drop", type=float, default=0.4)
    p.add_argument("--prev_drop", type=float, default=0.4)
    return p.parse_args()


def main():
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', args.gpu)

    print(f"Building model from {args.pretrained}...")
    model = build_model(args.pretrained, device=device)

    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(str(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=True)

    if args.sanity_check:
        run_sanity_checks(model, device)
        return

    param_groups = build_param_groups(model, args.base_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    scheduler = monai.optimizers.WarmupCosineSchedule(
        optimizer, t_total=args.epochs + 1, warmup_multiplier=0.1, warmup_steps=0)
    loss_fn = DiceCELoss(include_background=False, sigmoid=True,
                         smooth_dr=1e-5, smooth_nr=0, squared_pred=True)

    file_list = sorted(str(p) for p in TRAIN_DATA_ROOT.rglob("*.npz"))
    if args.max_files > 0:
        random.seed(42); random.shuffle(file_list)
        file_list = file_list[:args.max_files]
    print(f"Training files: {len(file_list)}")

    dataset = Vista3DFinetuneDataset(file_list, args.bbox_drop, args.prev_drop)

    def collate_patches(batch):
        patches = []
        for item in batch:
            patches.extend(item) if isinstance(item, list) else patches.append(item)
        keys = [k for k in patches[0].keys() if isinstance(patches[0][k], torch.Tensor)]
        return {k: torch.stack([p[k] for p in patches]) for k in keys}

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate_patches)

    print(f"\nTraining | {args.epochs} epochs | {len(file_list)} files")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch)
        scheduler.step()

        pe_norm = sum(p.data.norm().item() for p in model.point_head.prompt_encoder.parameters())
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"time={time.time()-t0:.0f}s | prompt_enc_norm={pe_norm:.4f}")

        torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': avg_loss},
                    args.ckpt_dir / f"epoch{epoch:03d}.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': avg_loss},
                        args.ckpt_dir / "best.pth")
            print(f"  -> best (loss={best_loss:.4f})")

        if epoch == 0 or (epoch + 1) % 5 == 0:
            run_sanity_checks(model, device, n_samples=10)

    print(f"\nDone. Best loss={best_loss:.4f}. Checkpoints: {args.ckpt_dir}")


if __name__ == "__main__":
    main()
