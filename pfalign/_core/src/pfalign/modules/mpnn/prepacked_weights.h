#pragma once

#include "mpnn_encoder.h"
#include "pfalign/primitives/gemm/gemm_impl.h"
#include <cstring>
#include <algorithm>

namespace pfalign {
namespace mpnn {

/**
 * Pre-packed weight matrix storage.
 *
 * Stores a weight matrix in the packed format expected by BLIS-style GEMM,
 * eliminating runtime packing overhead.
 *
 * Layout: NR-wide strips (NR=4 for 8*4 micro-kernel)
 *   packed[k * NR + j] = original[k * N + j_strip + j]
 *
 * This matches the format produced by pack_B_panel() in gemm_neon.cpp.
 */
struct PrepackedWeight {
    float* packed_data;  // Pre-packed matrix data
    int K;               // Rows of original matrix
    int N;               // Cols of original matrix
    int packed_size;     // Total size in floats

    PrepackedWeight() : packed_data(nullptr), K(0), N(0), packed_size(0) {
    }

    ~PrepackedWeight() {
        delete[] packed_data;
    }

    // Disable copy
    PrepackedWeight(const PrepackedWeight&) = delete;
    PrepackedWeight& operator=(const PrepackedWeight&) = delete;

    // Enable move
    PrepackedWeight(PrepackedWeight&& other) noexcept
        : packed_data(other.packed_data), K(other.K), N(other.N), packed_size(other.packed_size) {
        other.packed_data = nullptr;
        other.K = 0;
        other.N = 0;
        other.packed_size = 0;
    }

    PrepackedWeight& operator=(PrepackedWeight&& other) noexcept {
        if (this != &other) {
            delete[] packed_data;
            packed_data = other.packed_data;
            K = other.K;
            N = other.N;
            packed_size = other.packed_size;
            other.packed_data = nullptr;
            other.K = 0;
            other.N = 0;
            other.packed_size = 0;
        }
        return *this;
    }
};

/**
 * Pack a weight matrix into BLIS-style panel format.
 *
 * Converts B[K*N] (row-major) into packed NR-wide strips for 8*4 micro-kernel.
 * This is identical to pack_B_panel() logic from gemm_neon.cpp, but operates
 * on the full matrix at once.
 *
 * @param B Original weight matrix [K*N] (row-major)
 * @param K Rows of B
 * @param N Cols of B
 * @param NR Micro-tile width (4 for 8*4 kernel)
 * @return Pre-packed weight matrix
 */
inline PrepackedWeight pack_weight_matrix(const float* B, int K, int N, int NR = 4) {
    PrepackedWeight result;
    result.K = K;
    result.N = N;

    // Calculate packed size: ceil(N / NR) strips * K * NR
    int num_strips = (N + NR - 1) / NR;
    result.packed_size = num_strips * K * NR;
    result.packed_data = new float[result.packed_size];

    // Pack into NR-wide strips (same logic as pack_B_panel)
    float* pack_ptr = result.packed_data;

    for (int j_strip = 0; j_strip < N; j_strip += NR) {
        const int j_end = std::min(NR, N - j_strip);

        for (int k = 0; k < K; k++) {
            // Pack NR cols of B[k, :] (row slice)
            for (int j = 0; j < NR; j++) {
                if (j < j_end) {
                    *pack_ptr++ = B[k * N + j_strip + j];
                } else {
                    *pack_ptr++ = 0.0f;  // Pad to NR boundary
                }
            }
        }
    }

    return result;
}

/**
 * Collection of all pre-packed MPNN weights.
 *
 * Each weight matrix used as the RHS of a GEMM operation is pre-packed
 * at load time to eliminate runtime packing overhead.
 *
 * Memory overhead: ~2* (original + packed), but saves 10-20% runtime.
 */
struct PrepackedMPNNWeights {
    // Edge embedding
    PrepackedWeight edge_embedding;  // [edge_features * hidden_dim]
    PrepackedWeight positional;      // [66 * 16]
    PrepackedWeight W_e;             // [hidden_dim * hidden_dim]

    // Per-layer weights
    struct LayerPrepack {
        // Node message MLP
        PrepackedWeight W1;  // [3*hidden_dim * hidden_dim]
        PrepackedWeight W2;  // [hidden_dim * hidden_dim]
        PrepackedWeight W3;  // [hidden_dim * hidden_dim]

        // Position-wise FFN
        PrepackedWeight ffn_W_in;   // [hidden_dim * 4*hidden_dim]
        PrepackedWeight ffn_W_out;  // [4*hidden_dim * hidden_dim]

        // Edge update MLP
        PrepackedWeight W11;  // [3*hidden_dim * hidden_dim]
        PrepackedWeight W12;  // [hidden_dim * hidden_dim]
        PrepackedWeight W13;  // [hidden_dim * hidden_dim]
    };

    LayerPrepack* layers;
    int num_layers;

    PrepackedMPNNWeights(int num_layers_) : num_layers(num_layers_) {
        layers = new LayerPrepack[num_layers];
    }

    ~PrepackedMPNNWeights() {
        delete[] layers;
    }

    // Disable copy
    PrepackedMPNNWeights(const PrepackedMPNNWeights&) = delete;
    PrepackedMPNNWeights& operator=(const PrepackedMPNNWeights&) = delete;

    // Enable move
    PrepackedMPNNWeights(PrepackedMPNNWeights&& other) noexcept
        : layers(other.layers),
          num_layers(other.num_layers),
          edge_embedding(std::move(other.edge_embedding)),
          positional(std::move(other.positional)),
          W_e(std::move(other.W_e)) {
        other.layers = nullptr;
        other.num_layers = 0;
    }

    PrepackedMPNNWeights& operator=(PrepackedMPNNWeights&& other) noexcept {
        if (this != &other) {
            delete[] layers;
            layers = other.layers;
            num_layers = other.num_layers;
            edge_embedding = std::move(other.edge_embedding);
            positional = std::move(other.positional);
            W_e = std::move(other.W_e);
            other.layers = nullptr;
            other.num_layers = 0;
        }
        return *this;
    }
};

/**
 * Pre-pack all MPNN weights at load time.
 *
 * Converts all weight matrices used in GEMM operations into pre-packed format.
 * Should be called once after loading weights from checkpoint.
 *
 * @param weights Original MPNN weights
 * @param config MPNN configuration (for dimensions)
 * @return Pre-packed weights ready for zero-overhead GEMM
 */
inline PrepackedMPNNWeights prepack_mpnn_weights(const MPNNWeights& weights,
                                                 const MPNNConfig& config) {
    const int hidden_dim = config.hidden_dim;
    const int num_rbf = config.num_rbf;
    const int edge_features = 25 * num_rbf;  // 25 atom pairs * num_rbf

    PrepackedMPNNWeights prepacked(weights.num_layers);

    // Edge embedding: [edge_features * hidden_dim]
    // Note: V1 has no bias, so this is matrix-only
    prepacked.edge_embedding =
        pack_weight_matrix(weights.edge_embedding_weight, edge_features, hidden_dim);

    // Positional encoding: [66 * 16]
    prepacked.positional = pack_weight_matrix(weights.positional_weight, 66, 16);

    // W_e: [hidden_dim * hidden_dim]
    prepacked.W_e = pack_weight_matrix(weights.W_e_weight, hidden_dim, hidden_dim);

    // Per-layer weights
    for (int layer = 0; layer < weights.num_layers; layer++) {
        const auto& layer_weights = weights.layers[layer];
        auto& layer_prepacked = prepacked.layers[layer];

        // Node message MLP
        layer_prepacked.W1 =
            pack_weight_matrix(layer_weights.W1_weight, 3 * hidden_dim, hidden_dim);
        layer_prepacked.W2 = pack_weight_matrix(layer_weights.W2_weight, hidden_dim, hidden_dim);
        layer_prepacked.W3 = pack_weight_matrix(layer_weights.W3_weight, hidden_dim, hidden_dim);

        // Position-wise FFN
        layer_prepacked.ffn_W_in =
            pack_weight_matrix(layer_weights.ffn_W_in_weight, hidden_dim, 4 * hidden_dim);
        layer_prepacked.ffn_W_out =
            pack_weight_matrix(layer_weights.ffn_W_out_weight, 4 * hidden_dim, hidden_dim);

        // Edge update MLP
        layer_prepacked.W11 =
            pack_weight_matrix(layer_weights.W11_weight, 3 * hidden_dim, hidden_dim);
        layer_prepacked.W12 = pack_weight_matrix(layer_weights.W12_weight, hidden_dim, hidden_dim);
        layer_prepacked.W13 = pack_weight_matrix(layer_weights.W13_weight, hidden_dim, hidden_dim);
    }

    return prepacked;
}

}  // namespace mpnn
}  // namespace pfalign
