#!/usr/bin/env python3
"""
Compare V1 (full reference) vs V2 intermediates to find divergence.

V1 stores edges as [1, L, L, D] (full adjacency)
V2 stores edges as [L, k, D] (k-nearest neighbors only)

Usage:
    python compare_v1_v2_intermediates.py \\
        /path/to/v1_intermediates/ \\
        /path/to/v2_intermediates/
"""

import sys
import numpy as np
from pathlib import Path

def load_v1_reference(base_dir):
    """Load V1 reference intermediates."""
    base = Path(base_dir)

    data = {}
    data['knn_indices'] = np.load(base / "knn_indices.npy")  # [1, L, k]
    data['rbf_features'] = np.load(base / "rbf_features_raw.npy")  # [1, L, L, 400]
    data['edge_features_raw'] = np.load(base / "edge_features_raw.npy")  # [1, L, L, 64]
    data['before_layernorm'] = np.load(base / "before_layernorm.npy")  # [1, L, L, 64]
    data['h_E_init'] = np.load(base / "h_E_init.npy")  # [1, L, L, 64] after W_e
    data['h_V_layer0'] = np.load(base / "h_V_layer0.npy")  # [1, L, 64]
    data['h_E_layer0'] = np.load(base / "h_E_layer0.npy")  # [1, L, L, 64]
    data['h_V_layer1'] = np.load(base / "h_V_layer1.npy")
    data['h_E_layer1'] = np.load(base / "h_E_layer1.npy")
    data['h_V_layer2'] = np.load(base / "h_V_layer2.npy")
    data['h_E_layer2'] = np.load(base / "h_E_layer2.npy")
    data['h_V_final'] = np.load(base / "h_V_final.npy")  # [1, L, 64]

    # Layer 0 detailed steps
    data['layer0_h_EV'] = np.load(base / "layer0_step2_h_EV.npy")  # [1, L, L, 192]
    data['layer0_messages'] = np.load(base / "layer0_step3_messages.npy")  # [1, L, L, 64]
    data['layer0_h_agg'] = np.load(base / "layer0_step4_h_agg.npy")  # [1, L, 64]
    data['layer0_h_V_norm1'] = np.load(base / "layer0_step5b_h_V_norm1.npy")
    data['layer0_ffn_out'] = np.load(base / "layer0_step5d_h_ffn_out.npy")

    return data

def extract_knn_edges(v1_edges, knn_indices):
    """
    Extract k-NN edges from V1's full adjacency matrix.

    v1_edges: [1, L, L, D]
    knn_indices: [1, L, k]
    returns: [L, k, D]
    """
    batch, L, L2, D = v1_edges.shape
    assert L == L2
    assert batch == 1

    _, _, k = knn_indices.shape

    # Extract edges
    v1_knn = np.zeros((L, k, D), dtype=np.float32)
    for i in range(L):
        for j in range(k):
            neighbor = knn_indices[0, i, j]
            v1_knn[i, j, :] = v1_edges[0, i, neighbor, :]

    return v1_knn

def compare_arrays(name, v1_arr, v2_arr, threshold=1e-5):
    """Compare two arrays and report differences."""
    if v1_arr.shape != v2_arr.shape:
        print(f"  [FAIL] {name}: SHAPE MISMATCH")
        print(f"      V1: {v1_arr.shape}")
        print(f"      V2: {v2_arr.shape}")
        return False

    diff = np.abs(v1_arr - v2_arr)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    # Compute relative error
    v1_norm = np.linalg.norm(v1_arr)
    rel_error = rmse / v1_norm if v1_norm > 0 else rmse

    if rmse < threshold:
        print(f"  [OK] {name:30} MATCH (RMSE={rmse:.2e}, rel={rel_error:.2e})")
        return True
    else:
        print(f"  [FAIL] {name:30} DIVERGENCE")
        print(f"      Max abs:   {max_diff:.6e}")
        print(f"      Mean abs:  {mean_diff:.6e}")
        print(f"      RMSE:      {rmse:.6e}")
        print(f"      Relative:  {rel_error:.6e}")

        # Show sample values
        flat_v1 = v1_arr.flatten()
        flat_v2 = v2_arr.flatten()
        flat_diff = diff.flatten()

        print(f"      V1[0:5]:   {flat_v1[0:5]}")
        print(f"      V2[0:5]:   {flat_v2[0:5]}")
        print(f"      Diff[0:5]: {flat_diff[0:5]}")
        return False

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <v1_dir> <v2_dir>")
        sys.exit(1)

    v1_dir = sys.argv[1]
    v2_dir = Path(sys.argv[2])

    print("=" * 70)
    print("  V1 vs V2 MPNN Intermediate Comparison")
    print("=" * 70)

    # Load V1 reference
    print("\nLoading V1 reference...")
    v1 = load_v1_reference(v1_dir)
    knn_indices = v1['knn_indices']
    print(f"  L = {knn_indices.shape[1]}, k = {knn_indices.shape[2]}")

    # Load V2 intermediates (TODO: V2 needs to dump these)
    print("\nLoading V2 intermediates...")
    error_msg = (
        "V2 intermediate dumping is not implemented. "
        "Rebuild the encoder with intermediate logging before running this script."
    )
    print(f"  ERROR: {error_msg}")
    raise SystemExit(1)

if __name__ == '__main__':
    main()
