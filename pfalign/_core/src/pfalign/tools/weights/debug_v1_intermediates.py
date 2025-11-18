#!/usr/bin/env python3
"""
Extract V1 MPNN intermediate outputs for debugging.

Runs V1 MPNN encoder on Crambin and dumps intermediates at each stage.

Usage:
    python debug_v1_intermediates.py \\
        /path/to/1CRN.pdb \\
        /tmp/v1_intermediates/
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add V1 to path
sys.path.insert(0, '/home/ubuntu/projects/pfalign/src/pfalign/align/python')

def save_array(filepath, data):
    """Save numpy array as binary."""
    data.astype(np.float32).tofile(filepath)
    print(f"  Saved: {filepath} {data.shape}")

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pdb> <output_dir>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  V1 MPNN Intermediate Dumper")
    print("=" * 50)

    # Import V1 (this will fail if not available, which is fine)
    try:
        from mpnn_encoder_wrapper import MPNNEncoderWrapper
    except ImportError:
        print("ERROR: V1 Python wrapper not available")
        print("This script requires V1 MPNN Python bindings")
        sys.exit(1)

    print("\nStep 1: Loading V1 MPNN encoder...")
    encoder = MPNNEncoderWrapper()
    print("[OK] Loaded V1 encoder")

    print("\nStep 2: Parsing PDB...")
    print("ERROR: V1 debug harness not implemented in this repository")
    raise SystemExit(
        "Implement PDB parsing and intermediate dumping (see docs/tools) before running this script."
    )

if __name__ == '__main__':
    main()
