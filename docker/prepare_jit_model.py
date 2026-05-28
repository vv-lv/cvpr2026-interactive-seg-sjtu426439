#!/usr/bin/env python3
"""
One-time script: trace nnInteractive network and save as JIT model.
Run this BEFORE building Docker image.

Usage:
  python docker/prepare_jit_model.py \
    --checkpoint_dir /path/to/nnInteractive_v1.0_fold_all \
    --output model_traced.pt
"""
import argparse
import os
import sys
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", default=None,
                        help="Output path (default: checkpoint_dir/model_traced.pt)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = args.output or os.path.join(
        args.checkpoint_dir, "model_traced.pt")

    print(f"Loading model from {args.checkpoint_dir}...", flush=True)
    session = nnInteractiveInferenceSession(
        device=device, use_torch_compile=False, verbose=False,
        torch_n_threads=os.cpu_count(), do_autozoom=True,
        use_pinned_memory=True)
    session.initialize_from_trained_model_folder(
        model_training_output_dir=args.checkpoint_dir, use_fold='all')

    print("Tracing network...", flush=True)
    dummy = torch.randn(1, 8, 192, 192, 192, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(session.network, dummy)

    # Verify
    with torch.no_grad():
        out_orig = session.network(dummy)
        out_traced = traced(dummy)
    diff = (out_orig - out_traced).abs().max().item()
    print(f"Max output diff: {diff:.10f}", flush=True)
    assert diff < 1e-6, f"Trace verification failed! diff={diff}"

    torch.jit.save(traced, output)
    size_mb = os.path.getsize(output) / 1e6
    print(f"Saved JIT model: {output} ({size_mb:.0f}MB)", flush=True)

    del session, traced
    empty_cache(device)


if __name__ == "__main__":
    main()
