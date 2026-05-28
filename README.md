# Bottleneck Attention, Maxprob Assembly, and Round-Aware TTA for Interactive 3D Medical Segmentation

**CVPR 2026 Workshop on Foundation Models for Medical Vision — Interactive Track**

Team: sjtu426439

## Method Overview

Our approach builds on [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) with three key innovations:

### 1. Cross-Object Bottleneck Attention
A lightweight attention module injected at the U-Net bottleneck that enables cross-object context sharing for click-only (no bounding box) multi-target cases. The module learns to disambiguate overlapping targets by attending to all objects' click positions and types simultaneously.

- Trained on a balanced 310-case subset (competitor/non-competitor 1:1)
- Bypassed for bounding-box cases to avoid distribution shift
- ~32MB checkpoint, negligible inference overhead

### 2. Maxprob Assembly with Safety Net
Multi-object overlap conflicts are resolved by comparing per-object foreground probabilities (raw maxprob) rather than sequential overwriting (last-wins). A safety net guarantees every prompted object retains at least 5 voxels, preventing empty predictions.

### 3. Round-Aware Test-Time Augmentation
Flip-based TTA (axis 2) is applied **only at the final interaction round (R5)** to improve boundary precision without cascading errors to subsequent clicks. A Dice agreement threshold (>0.90) between original and augmented predictions gates whether TTA is actually used per-object.

## Results

### 176-case Validation (DSC)
| Variant | DSC AUC | Final DSC | vs baseline |
|---------|---------|-----------|-------------|
| baseline (last-wins) | 3.110 | 0.806 | — |
| + maxprob assembly | 3.151 | 0.814 | +0.041 |
| + v9 attention | 3.147 | 0.813 | +0.037 |
| **+ all (Docker)** | **3.161** | **0.812** | **+0.051** |

### 10-case Validation (DSC + NSD)
| Variant | DSC AUC | NSD AUC | dDSC | dNSD |
|---------|---------|---------|------|------|
| baseline (last-wins) | 3.289 | 3.322 | — | — |
| **Full pipeline** | **3.337** | **3.395** | **+0.048** | **+0.073** |

NSD improvement (+0.073) is 1.5x larger than DSC improvement (+0.048), indicating better surface precision.

## Repository Structure

```
.
├── docker/                  # Docker submission files
│   ├── predict.py           # Main inference script (maxprob + v9 + TTA)
│   ├── predict.sh           # Docker entrypoint
│   ├── attention_inference.py  # Bottleneck attention integration
│   ├── Dockerfile           # Docker build file
│   ├── build_docker.sh      # Build script
│   └── prepare_jit_model.py # JIT model tracing for faster loading
├── training/                # Training code
│   ├── trainer.py           # Main training loop
│   ├── bottleneck_attention.py  # V9 bottleneck attention module
│   ├── dataset.py           # Data loading and interaction simulation
│   ├── lora.py              # LoRA bypass utilities
│   └── ...
├── scripts/                 # Evaluation scripts
│   ├── eval_comprehensive.py    # Full 6-round evaluation with DSC + NSD
│   ├── eval_bottleneck_attn.py  # Attention module evaluation
│   └── build_v9_train_split.py  # Training data split generation
├── evaluation/              # Official evaluation code
│   ├── CVPR25_iter_eval.py  # Official iterative evaluation script
│   └── SurfaceDice.py       # Surface Dice computation
└── data/splits/             # Data split definitions
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.6
- [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) v1.1.x
- nnU-Net v2 (2.6.x)
- connected-components-3d
- scipy, pandas, scikit-image

## Docker Submission

Build and export:
```bash
cd docker
bash build_docker.sh
docker save sjtu439426:latest | gzip > sjtu439426.tar.gz
```

Run (matches official evaluation command):
```bash
docker container run --gpus "device=0" -m 32G --name sjtu439426 --rm \
  -v $PWD/inputs/:/workspace/inputs/ \
  -v $PWD/outputs/:/workspace/outputs/ \
  sjtu439426:latest /bin/bash -c "sh predict.sh"
```

## Training

### Bottleneck Attention (v9)
```bash
# 1. Generate balanced training split
python scripts/build_v9_train_split.py

# 2. Train attention module (~2 epochs on 310 cases)
python training/trainer.py \
  --config v9_no_scale \
  --train_json data/splits/v9_train.json \
  --epochs 3 --lr 1e-4
```

### Evaluation
```bash
# Full 6-round evaluation with DSC + NSD
python scripts/eval_comprehensive.py \
  --gpu 0 \
  --val_json data/splits/expanded_eval_nolowlimb.json \
  --variants v9_tta_r5_ag90 \
  --compute_nsd
```

## Hardware

| Environment | GPU | torch | CUDA |
|-------------|-----|-------|------|
| Training | 2x RTX 3090 (24GB) | 2.6.0 | 11.8 |
| Docker | any GPU >= 16GB | 2.7.0 | 12.6 |
| Eval server | QUADRO RTX5000 (16GB) | — | 12.1 |

## Acknowledgments

- [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) by MIC-DKFZ (Apache-2.0 code, CC BY-NC-SA 4.0 weights)
- [CVPR 2026 FMV Workshop](https://www.codabench.org/competitions/5263/)

## License

Code: Apache-2.0
