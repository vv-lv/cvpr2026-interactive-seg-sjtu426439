#!/usr/bin/env python3
"""构建 v9 训练集 (~1050 files): Competitor:Non-competitor ≈ 1:1。

排除已在 v8_val.json 中的数据集家族。
只选多目标文件 (labels >= 2)。
"""
import json
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_ROOT = Path("/media/ssd/jz/CVPR-BiomedSegFM/3D_train_npz_all")
OUTPUT = Path("/media/sjtu426/lby_t/cvpr2026-interactive/data/splits/v9_train.json")

ALLOCATION = {
    # Competitor (~530)
    'MRI/MR_BraTS-T1c': 30,
    'MRI/MR_BraTS-T1n': 30,
    'MRI/MR_BraTS-T2f': 30,
    'MRI/MR_BraTS-T2w': 30,
    'CT/CT_TotalSeg_organs': 65,
    'CT/CT_TotalSeg_cardiac': 55,
    'CT/CT_Abdomen1K': 45,
    'CT/CT_AMOS': 45,
    'US3D/US_Cardiac': 35,
    'MRI/MR_Heart_ACDC': 25,
    'CT/CT_LymphNode': 25,
    'CT/CT_COVID19-Infection': 20,
    'CT/CT_LiverTumor': 25,
    'MRI/MR_ISLES_DWI': 20,
    'MRI/MR_T1c_crossMoDA_Tumor_Cochlea': 30,
    'MRI/MR_HNTS-MRG_HeadTumor': 20,
    # Non-competitor (~520)
    'CT/CT_TotalSeg_muscles': 90,
    'CT/CT_TotalSeg-vertebrae': 70,
    'MRI/MR_Spider_IVD': 55,
    'MRI/MR_Spider_Vertebrae': 55,
    'MRI/MR_TotalSeg': 50,
    'MRI/MR_WMH_FLAIR': 45,
    'MRI/MR_WMH_T1': 45,
    'Microscopy/Microscopy_nucmm': 40,
    'CT/CT_DeepLesion': 25,
    'Microscopy/Microscopy_SELMA3D_ADplaques': 20,
    'Microscopy/Microscopy_SELMA3D_neural_activity_marker': 15,
    'Microscopy/Microscopy_SELMA3D_nuceus': 10,
}

COMPETITOR_PREFIXES = [
    'MRI/MR_BraTS', 'CT/CT_TotalSeg_organs', 'CT/CT_TotalSeg_cardiac',
    'CT/CT_Abdomen1K', 'CT/CT_AMOS', 'US3D/US_Cardiac',
    'MRI/MR_Heart_ACDC', 'CT/CT_LymphNode', 'CT/CT_COVID19',
    'CT/CT_LiverTumor', 'MRI/MR_ISLES', 'MRI/MR_T1c_crossMoDA',
    'MRI/MR_HNTS',
]


def is_competitor(ds_path):
    return any(ds_path.startswith(p) for p in COMPETITOR_PREFIXES)


def get_multi_obj_files(ds_rel_path, n_needed):
    """Get n_needed multi-object files from a dataset."""
    ds_path = DATA_ROOT / ds_rel_path
    if not ds_path.exists():
        print(f"  WARNING: {ds_path} not found!")
        return []

    all_files = sorted([str(ds_path / f) for f in os.listdir(ds_path) if f.endswith('.npz')])
    random.shuffle(all_files)

    selected = []
    for fpath in all_files:
        if len(selected) >= n_needed:
            break
        try:
            data = np.load(fpath, allow_pickle=True)
            gt = data['gts']
            n_labels = len([l for l in np.unique(gt) if l > 0])
            if n_labels >= 2:
                selected.append(fpath)
        except Exception:
            continue

    return selected


def main():
    random.seed(42)
    np.random.seed(42)

    all_files = []
    stats = defaultdict(int)

    print("Building v9 training split...")
    print(f"Target: {sum(ALLOCATION.values())} files")
    print()

    for ds_rel, n_target in sorted(ALLOCATION.items()):
        files = get_multi_obj_files(ds_rel, n_target)
        n_got = len(files)
        ds_name = ds_rel.split('/')[-1]
        comp_tag = "COMP" if is_competitor(ds_rel) else "NONC"
        print(f"  [{comp_tag}] {ds_name}: {n_got}/{n_target}")

        for f in files:
            all_files.append({
                'path': f,
                'dataset': ds_rel,
            })
        stats[comp_tag] += n_got

    print()
    print(f"Total: {len(all_files)} files")
    print(f"  Competitor: {stats['COMP']}")
    print(f"  Non-competitor: {stats['NONC']}")
    print(f"  Ratio: {stats['COMP']/(stats['NONC']+1e-9):.2f}:1")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        'version': 'v9',
        'description': '~1050 balanced multi-obj training set (comp:noncomp ~1:1)',
        'seed': 42,
        'n_files': len(all_files),
        'n_competitor': stats['COMP'],
        'n_non_competitor': stats['NONC'],
        'files': all_files,
    }
    with open(OUTPUT, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {OUTPUT}")


if __name__ == '__main__':
    main()
