#!/usr/bin/env python3
"""
Save MPNN weights to safetensors format for C++ loading.

Usage:
    python save_mpnn_weights.py \
        --checkpoint model.pt \
        --output weights.safetensors \
        --hidden-dim 64 \
        --num-layers 3 \
        --num-rbf 16

Or from Python:
    from save_mpnn_weights import save_mpnn_to_safetensors

    # From PyTorch state dict
    save_mpnn_to_safetensors(
        state_dict,
        "weights.safetensors",
        hidden_dim=64,
        num_layers=3,
        num_rbf=16
    )
"""

import struct
import json
import numpy as np
from typing import Dict, Any
from pathlib import Path


def save_mpnn_to_safetensors(
    weights: Dict[str, np.ndarray],
    output_path: str,
    hidden_dim: int,
    num_layers: int,
    num_rbf: int,
):
    """
    Save MPNN weights to safetensors format.

    Args:
        weights: Dictionary of weight tensors (numpy arrays or torch tensors)
        output_path: Output .safetensors file path
        hidden_dim: MPNN hidden dimension
        num_layers: Number of MPNN layers
        num_rbf: Number of RBF bins

    Expected keys in weights dict:
        - edge_embedding.weight
        - edge_embedding.norm.weight
        - edge_embedding.norm.bias
        - layers.{i}.W1.weight
        - layers.{i}.W1.bias
        - ... (see mpnn_weight_loader.h for full list)
    """

    # Convert all tensors to numpy float32
    tensors = {}
    for key, value in weights.items():
        # Handle PyTorch tensors
        if hasattr(value, 'cpu'):
            value = value.cpu().numpy()

        # Convert to float32
        tensors[key] = np.asarray(value, dtype=np.float32)

    # Build header metadata
    header = {}
    current_offset = 0

    # Sort keys for deterministic ordering
    for key in sorted(tensors.keys()):
        tensor = tensors[key]
        num_bytes = tensor.nbytes

        header[key] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [current_offset, current_offset + num_bytes]
        }

        current_offset += num_bytes

    # Add metadata
    header["__metadata__"] = {
        "hidden_dim": str(hidden_dim),
        "num_layers": str(num_layers),
        "num_rbf": str(num_rbf),
        "format": "mpnn_v2_weights"
    }

    # Serialize header to JSON
    header_json = json.dumps(header, separators=(',', ':'))
    header_bytes = header_json.encode('utf-8')
    header_size = len(header_bytes)

    # Write safetensors file
    with open(output_path, 'wb') as f:
        # Write header size (8 bytes, little-endian uint64)
        f.write(struct.pack('<Q', header_size))

        # Write header JSON
        f.write(header_bytes)

        # Write tensor data (in same order as header)
        for key in sorted(tensors.keys()):
            tensor = tensors[key]
            # Write as contiguous C-order (row-major)
            f.write(np.ascontiguousarray(tensor).tobytes())

    print(f"Saved MPNN weights to {output_path}")
    print(f"  Header size: {header_size} bytes")
    print(f"  Total size: {header_size + 8 + current_offset} bytes")
    print(f"  Num tensors: {len(tensors)}")


def save_pytorch_mpnn(checkpoint_path: str, output_path: str):
    """
    Load PyTorch checkpoint and save to safetensors.

    Assumes checkpoint is a state_dict with keys matching the expected format.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required to load .pt checkpoints")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        config = checkpoint.get('config', {})
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}

    # Infer config from state dict if not provided
    hidden_dim = config.get('hidden_dim')
    num_layers = config.get('num_layers')
    num_rbf = config.get('num_rbf', 16)

    if hidden_dim is None:
        # Infer from edge_embedding.norm.weight shape
        norm_key = 'edge_embedding.norm.weight'
        if norm_key in state_dict:
            hidden_dim = state_dict[norm_key].shape[0]
        else:
            raise ValueError("Cannot infer hidden_dim from checkpoint")

    if num_layers is None:
        # Count layers
        num_layers = 0
        for i in range(10):
            if f'layers.{i}.W1.weight' in state_dict:
                num_layers = i + 1
            else:
                break

    save_mpnn_to_safetensors(
        state_dict,
        output_path,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_rbf=num_rbf
    )


def create_random_weights(hidden_dim: int, num_layers: int, num_rbf: int) -> Dict[str, np.ndarray]:
    """
    Create random MPNN weights for testing.

    Uses small random values similar to test initialization.
    """
    np.random.seed(42)
    weights = {}

    edge_features = 25 * num_rbf

    # Edge embedding
    weights['edge_embedding.weight'] = np.random.randn(edge_features, hidden_dim).astype(np.float32) * 0.1
    weights['edge_embedding.norm.weight'] = np.ones(hidden_dim, dtype=np.float32)
    weights['edge_embedding.norm.bias'] = np.zeros(hidden_dim, dtype=np.float32)

    # Layers
    for i in range(num_layers):
        prefix = f'layers.{i}'

        input_dim = 2 * hidden_dim

        # Message MLP
        weights[f'{prefix}.W1.weight'] = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.W1.bias'] = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.W2.weight'] = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.W2.bias'] = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.W3.weight'] = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.W3.bias'] = np.random.randn(hidden_dim).astype(np.float32) * 0.1

        # Layer norms
        weights[f'{prefix}.norm1.weight'] = np.ones(hidden_dim, dtype=np.float32)
        weights[f'{prefix}.norm1.bias'] = np.zeros(hidden_dim, dtype=np.float32)
        weights[f'{prefix}.norm2.weight'] = np.ones(hidden_dim, dtype=np.float32)
        weights[f'{prefix}.norm2.bias'] = np.zeros(hidden_dim, dtype=np.float32)

        # FFN
        ffn_dim = 4 * hidden_dim
        weights[f'{prefix}.ffn.W_in.weight'] = np.random.randn(hidden_dim, ffn_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.ffn.W_in.bias'] = np.random.randn(ffn_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.ffn.W_out.weight'] = np.random.randn(ffn_dim, hidden_dim).astype(np.float32) * 0.1
        weights[f'{prefix}.ffn.W_out.bias'] = np.random.randn(hidden_dim).astype(np.float32) * 0.1

        # Norm3
        weights[f'{prefix}.norm3.weight'] = np.ones(hidden_dim, dtype=np.float32)
        weights[f'{prefix}.norm3.bias'] = np.zeros(hidden_dim, dtype=np.float32)

    return weights


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Save MPNN weights to safetensors format')
    parser.add_argument('--checkpoint', type=str, help='Input PyTorch checkpoint (.pt)')
    parser.add_argument('--output', type=str, required=True, help='Output safetensors file')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--num-rbf', type=int, default=16, help='Number of RBF bins')
    parser.add_argument('--random', action='store_true', help='Create random test weights')

    args = parser.parse_args()

    if args.random:
        print("Creating random test weights...")
        weights = create_random_weights(args.hidden_dim, args.num_layers, args.num_rbf)
        save_mpnn_to_safetensors(
            weights,
            args.output,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_rbf=args.num_rbf
        )
    elif args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        save_pytorch_mpnn(args.checkpoint, args.output)
    else:
        parser.error("Must specify either --checkpoint or --random")
