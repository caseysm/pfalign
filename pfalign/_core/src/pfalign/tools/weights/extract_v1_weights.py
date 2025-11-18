#!/usr/bin/env python3
"""
Extract V1 MPNN weights from C++ header file and convert to safetensors.

V1 weights are compiled into mpnn_weights.hpp as constexpr arrays.
This script parses that file and extracts them to safetensors format
for loading into V2.

Usage:
    python extract_v1_weights.py \
        --input /path/to/mpnn_weights.hpp \
        --output v1_weights.safetensors
"""

import re
import numpy as np
import argparse
from pathlib import Path
from save_mpnn_weights import save_mpnn_to_safetensors


def parse_cpp_array(content, array_name):
    """
    Extract a constexpr float array from C++ header file.

    Returns numpy array.
    """
    # Find the array declaration
    pattern = rf'constexpr\s+float\s+{re.escape(array_name)}\[\d+\]\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Array {array_name} not found in header file")

    # Extract the values
    values_str = match.group(1)

    # Parse float values (handle 'f' suffix)
    values = []
    for val in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?', values_str):
        val = val.rstrip('f')
        values.append(float(val))

    return np.array(values, dtype=np.float32)


def parse_shape_comment(content, array_name):
    """
    Extract shape from // Shape: comment before array.

    Returns tuple of dimensions.
    """
    # Find the shape comment before the array
    pattern = rf'//\s*Shape:\s*\[([^\]]+)\]\s+constexpr\s+float\s+{re.escape(array_name)}'
    match = re.search(pattern, content)

    if not match:
        return None

    shape_str = match.group(1)
    shape = tuple(int(d.strip()) for d in shape_str.split(','))
    return shape


def extract_v1_weights(header_path):
    """
    Extract all V1 MPNN weights from header file.

    Returns dict mapping V2 tensor names to numpy arrays.
    """
    with open(header_path, 'r') as f:
        content = f.read()

    weights = {}

    # V1 to V2 name mapping
    # V1 has 64D embeddings, 3 layers, 16 RBF bins

    print("Extracting edge embedding weights...")

    # Edge embedding (416 = 16 positional + 400 RBF features)
    # V1 concatenates [positional(16), RBF(400)] = 416 total
    edge_w_full = parse_cpp_array(content, 'protein_features__edge_embedding_w')
    edge_w_full = edge_w_full.reshape(416, 64)

    # Use FULL weight matrix (all 416 rows) to match V1
    weights['edge_embedding.weight'] = edge_w_full  # [416, 64]

    # Edge norms (scale = weight, offset = bias)
    weights['edge_embedding.norm.weight'] = parse_cpp_array(content, 'protein_features__norm_edges_scale')
    weights['edge_embedding.norm.bias'] = parse_cpp_array(content, 'protein_features__norm_edges_offset')

    # Positional encoding linear layer (66 -> 16)
    pos_w = parse_cpp_array(content, 'protein_features__positional_encodings__embedding_linear_w')
    weights['positional_encoding.weight'] = pos_w.reshape(66, 16)
    weights['positional_encoding.bias'] = parse_cpp_array(content, 'protein_features__positional_encodings__embedding_linear_b')

    # W_e: Initial edge transformation [64, 64]
    W_e_w = parse_cpp_array(content, 'W_e_w')
    weights['W_e.weight'] = W_e_w.reshape(64, 64)
    weights['W_e.bias'] = parse_cpp_array(content, 'W_e_b')

    # Extract 3 layers
    # Layer naming pattern in V1:
    #   Layer 0: enc_layer__enc0_*
    #   Layer 1: enc_layer_1__enc1_*
    #   Layer 2: enc_layer_2__enc2_*

    layer_configs = [
        ('enc_layer__', 'enc0'),
        ('enc_layer_1__', 'enc1'),
        ('enc_layer_2__', 'enc2')
    ]

    for i, (layer_prefix, layer_suffix) in enumerate(layer_configs):
        print(f"Extracting layer {i} weights...")

        layer_prefix_v2 = f'layers.{i}'
        v1_name_prefix = f'{layer_prefix}{layer_suffix}'

        # Message MLP (W1, W2, W3)
        # Both V1 and V2 concatenate [h_i, h_j, e_ij] = 3*hidden_dim
        # V1: enc_layer__enc0_W1_w shape [192, 64] = [3*64, 64]
        # V2 expects: [192, 64] for GEMM C = A @ B

        W1_w = parse_cpp_array(content, f'{v1_name_prefix}_W1_w')
        weights[f'{layer_prefix_v2}.W1.weight'] = W1_w.reshape(192, 64)

        weights[f'{layer_prefix_v2}.W1.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_W1_b')

        W2_w = parse_cpp_array(content, f'{v1_name_prefix}_W2_w')
        weights[f'{layer_prefix_v2}.W2.weight'] = W2_w.reshape(64, 64)

        weights[f'{layer_prefix_v2}.W2.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_W2_b')

        W3_w = parse_cpp_array(content, f'{v1_name_prefix}_W3_w')
        weights[f'{layer_prefix_v2}.W3.weight'] = W3_w.reshape(64, 64)

        weights[f'{layer_prefix_v2}.W3.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_W3_b')

        # Layer norms (scale = weight, offset = bias)
        weights[f'{layer_prefix_v2}.norm1.weight'] = parse_cpp_array(content, f'{v1_name_prefix}_norm1_scale')
        weights[f'{layer_prefix_v2}.norm1.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_norm1_offset')

        weights[f'{layer_prefix_v2}.norm2.weight'] = parse_cpp_array(content, f'{v1_name_prefix}_norm2_scale')
        weights[f'{layer_prefix_v2}.norm2.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_norm2_offset')

        # FFN (position-wise feed forward)
        ffn_name = f'{layer_prefix}position_wise_feed_forward__{layer_suffix}_dense'

        W_in_w = parse_cpp_array(content, f'{ffn_name}_W_in_w')
        weights[f'{layer_prefix_v2}.ffn.W_in.weight'] = W_in_w.reshape(64, 256)

        weights[f'{layer_prefix_v2}.ffn.W_in.bias'] = parse_cpp_array(content, f'{ffn_name}_W_in_b')

        W_out_w = parse_cpp_array(content, f'{ffn_name}_W_out_w')
        weights[f'{layer_prefix_v2}.ffn.W_out.weight'] = W_out_w.reshape(256, 64)

        weights[f'{layer_prefix_v2}.ffn.W_out.bias'] = parse_cpp_array(content, f'{ffn_name}_W_out_b')

        # Edge update MLP (W11, W12, W13)
        W11_w = parse_cpp_array(content, f'{v1_name_prefix}_W11_w')
        weights[f'{layer_prefix_v2}.W11.weight'] = W11_w.reshape(192, 64)

        weights[f'{layer_prefix_v2}.W11.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_W11_b')

        W12_w = parse_cpp_array(content, f'{v1_name_prefix}_W12_w')
        weights[f'{layer_prefix_v2}.W12.weight'] = W12_w.reshape(64, 64)

        weights[f'{layer_prefix_v2}.W12.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_W12_b')

        W13_w = parse_cpp_array(content, f'{v1_name_prefix}_W13_w')
        weights[f'{layer_prefix_v2}.W13.weight'] = W13_w.reshape(64, 64)

        weights[f'{layer_prefix_v2}.W13.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_W13_b')

        # Norm3 (edge layer norm)
        weights[f'{layer_prefix_v2}.norm3.weight'] = parse_cpp_array(content, f'{v1_name_prefix}_norm3_scale')
        weights[f'{layer_prefix_v2}.norm3.bias'] = parse_cpp_array(content, f'{v1_name_prefix}_norm3_offset')

    print(f"\nExtracted {len(weights)} weight tensors")

    # Verify shapes
    print("\nWeight shapes:")
    for name, arr in sorted(weights.items()):
        print(f"  {name}: {arr.shape}")

    return weights


def main():
    parser = argparse.ArgumentParser(description='Extract V1 MPNN weights to safetensors')
    parser.add_argument('--input', type=str,
                        default='/home/ubuntu/projects/pfalign/src/pfalign/align/kernels/mpnn/include/mpnn_weights.hpp',
                        help='Path to V1 mpnn_weights.hpp file')
    parser.add_argument('--output', type=str,
                        default='v1_mpnn_weights.safetensors',
                        help='Output safetensors file')

    args = parser.parse_args()

    print(f"Extracting V1 weights from: {args.input}")

    # Extract weights
    weights = extract_v1_weights(args.input)

    # Save to safetensors
    save_mpnn_to_safetensors(
        weights,
        args.output,
        hidden_dim=64,
        num_layers=3,
        num_rbf=16
    )

    print(f"\n[OK] Saved V1 weights to: {args.output}")
    print(f"\nYou can now load these in V2:")
    print(f"  auto [weights, config] = MPNNWeightLoader::load(\"{args.output}\");")


if __name__ == '__main__':
    main()
