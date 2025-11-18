#!/usr/bin/env python3
"""Print V1 intermediate values for manual comparison with V2."""

import numpy as np
from pathlib import Path

v1_dir = Path("/home/ubuntu/projects/pfalign/data/mpnn_reference/1CRN_crambin")

print("=== V1 Reference Values ===")

# h_E_init (after W_e)
h_E_init = np.load(v1_dir / "h_E_init.npy")  # [1, 46, 46, 64]
print(f"V1 h_E_init[0,0,0:5]: {h_E_init[0, 0, 0, :5]}")

# After each layer
for layer in [0, 1, 2]:
    h_V = np.load(v1_dir / f"h_V_layer{layer}.npy")  # [1, L, 64]
    h_E = np.load(v1_dir / f"h_E_layer{layer}.npy")  # [1, L, L, 64]
    print(f"After layer {layer}:")
    print(f"V1   h_V[0:5]: {h_V[0, 0, :5]}")
    print(f"V1   h_E[0,0,0:5]: {h_E[0, 0, 0, :5]}")

# Final
h_V_final = np.load(v1_dir / "h_V_final.npy")
print(f"V1 h_V_final[0:5]: {h_V_final[0, 0, :5]}")
print("===========================\n")
