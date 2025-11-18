#!/usr/bin/env python3
"""
Compare V1 vs V2 intermediate outputs to find first divergence.

Usage:
    python compare_intermediates.py \\
        /tmp/v1_intermediates/ \\
        /tmp/v2_intermediates/
"""

import sys
import numpy as np
from pathlib import Path

def load_array(filepath):
    """Load binary float32 array."""
    return np.fromfile(filepath, dtype=np.float32)

def compare_stage(name, v1_data, v2_data, threshold=1e-5):
    """Compare two arrays and report differences."""
    if v1_data.shape != v2_data.shape:
        print(f"  [FAIL] {name}: SHAPE MISMATCH")
        print(f"      V1: {v1_data.shape}")
        print(f"      V2: {v2_data.shape}")
        return False

    diff = np.abs(v1_data - v2_data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    if rmse < threshold:
        print(f"  [OK] {name}: MATCH (RMSE={rmse:.2e})")
        return True
    else:
        print(f"  [FAIL] {name}: DIVERGENCE DETECTED")
        print(f"      Max diff:  {max_diff:.6e}")
        print(f"      Mean diff: {mean_diff:.6e}")
        print(f"      RMSE:      {rmse:.6e}")

        # Show sample values
        print(f"      V1[0:5]:   {v1_data.flat[0:5]}")
        print(f"      V2[0:5]:   {v2_data.flat[0:5]}")
        print(f"      Diff[0:5]: {diff.flat[0:5]}")
        return False

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <v1_dir> <v2_dir>")
        sys.exit(1)

    v1_dir = Path(sys.argv[1])
    v2_dir = Path(sys.argv[2])

    print("=" * 60)
    print("  V1 vs V2 Intermediate Comparison")
    print("=" * 60)

    # List all stages to compare
    stages = [
        "00_input_coords",
        "01_rbf_features",
        "02_edge_embedding",
        "03_edge_norm",
        "04_edge_after_we",
        "05_nodes_init",
        "10_layer0_messages",
        "11_layer0_aggregated",
        "12_layer0_after_norm1",
        "13_layer0_after_ffn",
        "14_layer0_after_norm2",
        "15_layer0_edges_updated",
        "20_layer1_messages",
        "21_layer1_aggregated",
        "22_layer1_after_norm1",
        "23_layer1_after_ffn",
        "24_layer1_after_norm2",
        "25_layer1_edges_updated",
        "30_layer2_messages",
        "31_layer2_aggregated",
        "32_layer2_after_norm1",
        "33_layer2_after_ffn",
        "34_layer2_after_norm2",
        "35_layer2_edges_updated",
        "99_final_output",
    ]

    first_divergence = None

    for stage in stages:
        v1_file = v1_dir / f"{stage}.bin"
        v2_file = v2_dir / f"{stage}.bin"

        if not v1_file.exists():
            print(f"  - {stage}: V1 file missing")
            continue
        if not v2_file.exists():
            print(f"  - {stage}: V2 file missing")
            continue

        v1_data = load_array(v1_file)
        v2_data = load_array(v2_file)

        match = compare_stage(stage, v1_data, v2_data)

        if not match and first_divergence is None:
            first_divergence = stage
            print(f"\n  ⚠ FIRST DIVERGENCE DETECTED AT: {stage}")
            print("  Stopping comparison (subsequent stages will compound error)\n")
            break

    print("=" * 60)
    if first_divergence:
        print(f"Result: First divergence at stage '{first_divergence}'")
    else:
        print("Result: All stages match!")
    print("=" * 60)

if __name__ == '__main__':
    main()
