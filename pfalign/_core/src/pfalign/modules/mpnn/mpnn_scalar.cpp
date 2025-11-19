/**
 * Scalar implementation of MPNN encoder.
 *
 * Assembles primitives into full message passing neural network.
 */

#include "mpnn_encoder.h"
#include "blocked_gemm.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/primitives/knn/knn_impl.h"
#include "pfalign/primitives/rbf/rbf_impl.h"
#include "pfalign/primitives/gemm/gemm_impl.h"
#include "pfalign/primitives/layer_norm/layer_norm_impl.h"
#include "pfalign/primitives/gather/gather_impl.h"
#include "pfalign/common/profiling.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

// Debug: print first 5 values (disabled for benchmarking)
#if 0
    #define DEBUG_PRINT_ARRAY(name, arr, n)                                                     \
        do {                                                                                    \
            std::cout << "V2 " << name << "[0:5]: ";                                           \
            for (int _i = 0; _i < std::min(n, 5); _i++) {                                      \
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) << arr[_i] << " "; \
            }                                                                                   \
            std::cout << std::endl;                                                            \
        } while(0)
#else
    #define DEBUG_PRINT_ARRAY(name, arr, n) \
        do {                                \
        } while (0)
#endif

namespace {

inline void validate_finite_array(const float* data, std::size_t count, const char* label) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            std::ostringstream oss;
            oss << label << " contains non-finite value at index " << i;
            throw std::invalid_argument(oss.str());
        }
    }
}

}  // namespace

namespace pfalign {
namespace mpnn {

constexpr float CB_COEFF_A = -0.58273431f;
constexpr float CB_COEFF_B = 0.56802827f;
constexpr float CB_COEFF_C = -0.54067466f;

static inline void cross_product(const float* u, const float* v, float* out) {
    out[0] = u[1] * v[2] - u[2] * v[1];
    out[1] = u[2] * v[0] - u[0] * v[2];
    out[2] = u[0] * v[1] - u[1] * v[0];
}

/**
 * Helper: Extract atom coordinates from backbone.
 *
 * Input:  coords [L * 4 * 3] = (N, Ca, C, O) for each residue
 * Output: Ca, N, C, O arrays [L * 3]
 *         Cb [L * 3] computed as virtual beta carbon
 */
static void extract_atoms(const float* coords, int L, float* Ca, float* N, float* C, float* O,
                          float* Cb) {
    for (int i = 0; i < L; i++) {
        // Extract backbone atoms
        const float* residue = coords + i * 4 * 3;

        // N atom
        N[i * 3 + 0] = residue[0 * 3 + 0];
        N[i * 3 + 1] = residue[0 * 3 + 1];
        N[i * 3 + 2] = residue[0 * 3 + 2];

        // Ca atom
        Ca[i * 3 + 0] = residue[1 * 3 + 0];
        Ca[i * 3 + 1] = residue[1 * 3 + 1];
        Ca[i * 3 + 2] = residue[1 * 3 + 2];

        // C atom
        C[i * 3 + 0] = residue[2 * 3 + 0];
        C[i * 3 + 1] = residue[2 * 3 + 1];
        C[i * 3 + 2] = residue[2 * 3 + 2];

        // O atom
        O[i * 3 + 0] = residue[3 * 3 + 0];
        O[i * 3 + 1] = residue[3 * 3 + 1];
        O[i * 3 + 2] = residue[3 * 3 + 2];

        // Compute virtual Cb position using V1 coefficients
        float b_vec[3] = {Ca[i * 3 + 0] - N[i * 3 + 0], Ca[i * 3 + 1] - N[i * 3 + 1],
                          Ca[i * 3 + 2] - N[i * 3 + 2]};
        float c_vec[3] = {C[i * 3 + 0] - Ca[i * 3 + 0], C[i * 3 + 1] - Ca[i * 3 + 1],
                          C[i * 3 + 2] - Ca[i * 3 + 2]};
        float a_vec[3];
        cross_product(b_vec, c_vec, a_vec);

        for (int d = 0; d < 3; d++) {
            Cb[i * 3 + d] = CB_COEFF_A * a_vec[d] + CB_COEFF_B * b_vec[d] + CB_COEFF_C * c_vec[d] +
                            Ca[i * 3 + d];
        }
    }
}

/**
 * Helper: Compute 25 RBF feature sets for atom pairs.
 *
 * Atom pairs (5 atoms * 5 atoms):
 *   Ca-Ca, Ca-N, Ca-C, Ca-O, Ca-Cb,
 *   N-Ca,  N-N,  N-C,  N-O,  N-Cb,
 *   C-Ca,  C-N,  C-C,  C-O,  C-Cb,
 *   O-Ca,  O-N,  O-C,  O-O,  O-Cb,
 *   Cb-Ca, Cb-N, Cb-C, Cb-O, Cb-Cb
 *
 * For each edge (i, j), compute distance for each pair and apply RBF.
 */
static void compute_edge_rbf_features(const float* Ca, const float* N, const float* C,
                                      const float* O, const float* Cb, const int* neighbor_indices,
                                      int L, int k, const MPNNConfig& config,
                                      float* rbf_features  // [L * k * (25 * num_rbf)]
) {
    const float* atom_arrays[5] = {Ca, N, C, O, Cb};
    struct AtomPair {
        int atom_i;
        int atom_j;
    };
    static constexpr AtomPair atom_pairs[25] = {
        {0, 0},  // Ca-Ca
        {1, 1},  // N-N
        {2, 2},  // C-C
        {3, 3},  // O-O
        {4, 4},  // Cb-Cb
        {0, 1},  // Ca-N
        {0, 2},  // Ca-C
        {0, 3},  // Ca-O
        {0, 4},  // Ca-Cb
        {1, 2},  // N-C
        {1, 3},  // N-O
        {1, 4},  // N-Cb
        {4, 2},  // Cb-C
        {4, 3},  // Cb-O
        {3, 2},  // O-C
        {1, 0},  // N-Ca
        {2, 0},  // C-Ca
        {3, 0},  // O-Ca
        {4, 0},  // Cb-Ca
        {2, 1},  // C-N
        {3, 1},  // O-N
        {4, 1},  // Cb-N
        {2, 4},  // C-Cb
        {3, 4},  // O-Cb
        {2, 3}   // C-O
    };

    // Precompute RBF centers (once for all atom pairs)
    float centers[16];  // Max num_rbf
    pfalign::rbf::rbf_initialize_centers(centers, config.num_rbf, config.rbf_min_dist,
                                         config.rbf_max_dist);
    float inv_sigma_sq = pfalign::rbf::rbf_compute_inv_sigma_sq(
        config.rbf_min_dist, config.rbf_max_dist, config.num_rbf);

    // For each query residue
    for (int i = 0; i < L; i++) {
        // For each neighbor
        for (int j = 0; j < k; j++) {
            int neighbor_idx = neighbor_indices[i * k + j];

            // Skip invalid neighbors
            if (neighbor_idx < 0) {
                std::memset(rbf_features + (i * k + j) * 25 * config.num_rbf, 0,
                            static_cast<size_t>(25 * config.num_rbf) * sizeof(float));
                continue;
            }

            float* feature_out = rbf_features + (i * k + j) * 25 * config.num_rbf;

            // Compute 25 atom-pair distances and RBFs
            for (int pair_idx = 0; pair_idx < 25; pair_idx++) {
                int atom_i = atom_pairs[pair_idx].atom_i;
                int atom_j = atom_pairs[pair_idx].atom_j;

                const float* atom1 = atom_arrays[atom_i] + i * 3;
                const float* atom2 = atom_arrays[atom_j] + neighbor_idx * 3;

                float dx = atom2[0] - atom1[0];
                float dy = atom2[1] - atom1[1];
                float dz = atom2[2] - atom1[2];
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                pfalign::rbf::rbf_single<ScalarBackend>(dist, centers, inv_sigma_sq,
                                                        feature_out + pair_idx * config.num_rbf,
                                                        config.num_rbf);
            }
        }
    }
}

/**
 * Helper: GELU activation (EXACT formula matching V1/JAX).
 */
static void gelu(float* x, int size) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // This is the EXACT formula (approximate=False in JAX), matching V1
    constexpr float sqrt_2 = 1.41421356237f;
    for (int i = 0; i < size; i++) {
        x[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] / sqrt_2));
    }
}

/**
 * Helper: Edge update layer.
 *
 * For each edge:
 * 1. Gather neighbor embeddings (with updated h_V)
 * 2. Compute edge updates: MLP(concat(h_i, h_j, e_ij))
 * 3. Apply residual + LayerNorm
 */
/**
 * Helper: Compute positional encodings for all edges in batch.
 *
 * For each edge (i,j), computes positional encoding based on residue offset or chain difference.
 * Output: [L*k, 16] positional features (bias + weight[class_idx])
 */
static void compute_positional_batch(const int* neighbor_indices, const int* residue_idx,
                                     const int* chain_labels, int L, int k,
                                     const float* positional_weight, const float* positional_bias,
                                     float* positional_batch  // Output: [L*k, 16]
) {
    PROFILE_SCOPE("MPNN_batched_positional_encoding");

    const int positional_features = 16;
    const int max_relative = 32;
    const int positional_classes = 2 * max_relative + 2;  // 66

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < k; j++) {
            int edge_idx = i * k + j;
            int neighbor = neighbor_indices[edge_idx];
            float* output = positional_batch + edge_idx * positional_features;

            if (neighbor < 0 || neighbor >= L) {
                // Invalid edge - zero out
                std::memset(output, 0, static_cast<size_t>(positional_features) * sizeof(float));
                continue;
            }

            // Compute class_idx based on chain and residue offset
            int class_idx;
            if (chain_labels[i] != chain_labels[neighbor]) {
                // Different chains
                class_idx = positional_classes - 1;  // 65
            } else {
                // Same chain - use residue offset
                int offset = residue_idx[i] - residue_idx[neighbor];
                offset = std::max(-max_relative, std::min(max_relative, offset));
                class_idx = offset + max_relative;  // Maps [-32, 32] -> [0, 64]
            }

            // Lookup: output = bias + weight[class_idx]
            const float* weight_row = positional_weight + class_idx * positional_features;
            for (int d = 0; d < positional_features; d++) {
                output[d] = positional_bias[d] + weight_row[d];
            }
        }
    }
}

/**
 * Helper: Prepare batched edge inputs by concatenating positional + RBF features.
 *
 * Output: [L*k, 416] where 416 = 16 (positional) + 400 (RBF)
 */
static void prepare_edge_inputs_batch(const float* positional_batch,  // [L*k, 16]
                                      const float* rbf_features,      // [L*k, 400]
                                      int L, int k,
                                      float* edge_inputs  // Output: [L*k, 416]
) {
    PROFILE_SCOPE("MPNN_batched_edge_concat");

    const int positional_features = 16;
    const int rbf_features_size = 400;
    const int total_features = positional_features + rbf_features_size;

    for (int edge = 0; edge < L * k; edge++) {
        float* output = edge_inputs + edge * total_features;

        // Copy positional [16]
        std::memcpy(output, positional_batch + edge * positional_features,
                    static_cast<size_t>(positional_features) * sizeof(float));

        // Copy RBF [400]
        std::memcpy(output + positional_features, rbf_features + edge * rbf_features_size,
                    static_cast<size_t>(rbf_features_size) * sizeof(float));
    }
}

/**
 * Batched edge embedding computation.
 *
 * Processes all L*k edges in a single batched operation:
 * 1. Compute positional encodings for all edges
 * 2. Concatenate [positional + RBF] features
 * 3. Single batched GEMM to produce edge embeddings
 *
 * Replaces 5,940 tiny GEMMs with 1 large GEMM.
 */
static void batched_edge_embedding(const float* rbf_features,    // [L*k, 400]
                                   const int* neighbor_indices,  // [L*k]
                                   const int* residue_idx,       // [L]
                                   const int* chain_labels,      // [L]
                                   int L, int k, int hidden_dim,
                                   const float* edge_embedding_weight,  // [416, hidden_dim]
                                   const float* positional_weight,      // [66, 16]
                                   const float* positional_bias,        // [16]
                                   float* edge_embeddings,              // Output: [L*k, hidden_dim]
                                   pfalign::memory::Arena* temp_arena) {
    PROFILE_SCOPE("MPNN_batched_edge_embedding");

    const int num_edges = L * k;
    const int positional_features = 16;
    const int rbf_features_size = 400;
    const int total_features = positional_features + rbf_features_size;

    // Step 1: Compute all positional encodings [L*k, 16]
    float* positional_batch = temp_arena->allocate<float>(num_edges * positional_features);
    compute_positional_batch(neighbor_indices, residue_idx, chain_labels, L, k, positional_weight,
                             positional_bias, positional_batch);

    // Step 2: Prepare all edge inputs [L*k, 416]
    float* edge_inputs = temp_arena->allocate<float>(num_edges * total_features);
    prepare_edge_inputs_batch(positional_batch, rbf_features, L, k, edge_inputs);

    // Step 3: Cache-blocked batched GEMM [L*k, 416] * [416, hidden_dim] -> [L*k, hidden_dim]
    // Optimal block size for [416, 64]: 192 rows (fits in 256KB L2 cache)
    blocked_gemm<ScalarBackend>(edge_inputs,            // [num_edges, 416]
                                edge_embedding_weight,  // [416, hidden_dim]
                                edge_embeddings,        // [num_edges, hidden_dim]
                                num_edges,              // M
                                hidden_dim,             // N
                                total_features,         // K
                                1.0f, 0.0f,
                                256  // block_size (conservative, fits all cases)
    );
}

[[maybe_unused]] static void
edge_update_layer(const float* h_V,  // Updated node embeddings [L * hidden_dim]
                  float* h_E,        // Edge embeddings [L * k * hidden_dim] (will be updated)
                  const int* neighbor_indices, int L, int k, int hidden_dim,
                  const MPNNWeights::LayerWeights& weights,
                  float* edge_input,  // Workspace [3 * hidden_dim]
                  float* edge_temp    // Workspace [hidden_dim]
) {
    // For each edge
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < k; j++) {
            int neighbor_idx = neighbor_indices[i * k + j];
            if (neighbor_idx < 0)
                continue;

            float* h_E_ij = h_E + (i * k + j) * hidden_dim;

            // Concatenate: [h_i, e_ij, h_j] (same order as node messages)
            {
                PROFILE_SCOPE("MPNN_edge_input_concat");
                std::memcpy(edge_input, h_V + i * hidden_dim,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
                std::memcpy(edge_input + hidden_dim, h_E_ij,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
                std::memcpy(edge_input + 2 * hidden_dim, h_V + neighbor_idx * hidden_dim,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
            }

            // MLP: W11 -> GELU -> W12 -> GELU -> W13
            // W11
            {
                PROFILE_SCOPE("MPNN_edge_W11_gemm");
                pfalign::gemm::gemm<ScalarBackend>(edge_input, weights.W11_weight.data(), edge_temp,
                                                   1, hidden_dim, 3 * hidden_dim, 1.0f, 0.0f);
            }
            {
                PROFILE_SCOPE("MPNN_edge_W11_bias");
                for (int d = 0; d < hidden_dim; d++) {
                    edge_temp[d] += weights.W11_bias[d];
                }
            }
            {
                PROFILE_SCOPE("MPNN_edge_W11_gelu");
                gelu(edge_temp, hidden_dim);
            }

            // W12
            float* w12_out = edge_input;  // Reuse edge_input as workspace
            {
                PROFILE_SCOPE("MPNN_edge_W12_memcpy");
                std::memcpy(w12_out, edge_temp, static_cast<size_t>(hidden_dim) * sizeof(float));
            }
            {
                PROFILE_SCOPE("MPNN_edge_W12_gemm");
                pfalign::gemm::gemm<ScalarBackend>(w12_out, weights.W12_weight.data(), edge_temp, 1,
                                                   hidden_dim, hidden_dim, 1.0f, 0.0f);
            }
            {
                PROFILE_SCOPE("MPNN_edge_W12_bias");
                for (int d = 0; d < hidden_dim; d++) {
                    edge_temp[d] += weights.W12_bias[d];
                }
            }
            {
                PROFILE_SCOPE("MPNN_edge_W12_gelu");
                gelu(edge_temp, hidden_dim);
            }

            // W13
            {
                PROFILE_SCOPE("MPNN_edge_W13_memcpy");
                std::memcpy(w12_out, edge_temp, static_cast<size_t>(hidden_dim) * sizeof(float));
            }
            {
                PROFILE_SCOPE("MPNN_edge_W13_gemm");
                pfalign::gemm::gemm<ScalarBackend>(w12_out, weights.W13_weight.data(), edge_temp, 1,
                                                   hidden_dim, hidden_dim, 1.0f, 0.0f);
            }
            {
                PROFILE_SCOPE("MPNN_edge_W13_bias");
                for (int d = 0; d < hidden_dim; d++) {
                    edge_temp[d] += weights.W13_bias[d];
                }
            }

            // Residual connection: h_E + edge_update
            {
                PROFILE_SCOPE("MPNN_edge_residual");
                for (int d = 0; d < hidden_dim; d++) {
                    h_E_ij[d] += edge_temp[d];
                }
            }

            // LayerNorm (norm3)
            {
                PROFILE_SCOPE("MPNN_edge_layer_norm");
                pfalign::layer_norm::layer_norm_forward<ScalarBackend>(
                    h_E_ij, h_E_ij, weights.norm3_gamma.data(), weights.norm3_beta.data(),
                    hidden_dim);
            }
        }
    }
}

/**
 * Helper: Batched edge update computation.
 *
 * Processes all L*k edges in a single batched operation instead of per-edge loops.
 * Updates edge embeddings using: MLP(h_i, e_ij, h_j) + residual + layer_norm
 *
 * @param h_V             Node embeddings [L * hidden_dim]
 * @param h_E             Edge embeddings [L * k * hidden_dim] (updated in-place)
 * @param neighbor_indices Neighbor indices [L * k]
 * @param L               Number of residues
 * @param k               Number of neighbors per residue
 * @param hidden_dim      Hidden dimension
 * @param weights         Layer weights (W11, W12, W13, biases, layer norm)
 * @param temp_arena      Arena for temporary allocations
 */
static void batched_edge_update(const float* h_V, float* h_E, const int* neighbor_indices, int L,
                                int k, int hidden_dim, const MPNNWeights::LayerWeights& weights,
                                pfalign::memory::Arena* temp_arena) {
    PROFILE_SCOPE("MPNN_batched_edge_update");

    const int num_edges = L * k;
    const int input_dim = 3 * hidden_dim;

    // Step 1: Prepare batched inputs [L*k, 3*hidden_dim]
    // Input: [h_i, e_ij, h_j] for each edge
    float* batched_inputs = temp_arena->allocate<float>(num_edges * input_dim);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < k; j++) {
            int edge_idx = i * k + j;
            int neighbor_idx = neighbor_indices[edge_idx];

            if (neighbor_idx < 0) {
                // Invalid neighbor - zero out input
                std::memset(batched_inputs + edge_idx * input_dim, 0,
                            static_cast<size_t>(input_dim) * sizeof(float));
                continue;
            }

            float* input_row = batched_inputs + edge_idx * input_dim;

            // Concatenate [h_i, e_ij, h_j]
            std::memcpy(input_row, h_V + i * hidden_dim,
                        static_cast<size_t>(hidden_dim) * sizeof(float));
            std::memcpy(input_row + hidden_dim, h_E + edge_idx * hidden_dim,
                        static_cast<size_t>(hidden_dim) * sizeof(float));
            std::memcpy(input_row + 2 * hidden_dim, h_V + neighbor_idx * hidden_dim,
                        static_cast<size_t>(hidden_dim) * sizeof(float));
        }
    }

    // Allocate intermediate buffer (double-buffering optimization)
    // Reuse batched_inputs after W11, and use single intermediate buffer
    float* buffer_a = temp_arena->allocate<float>(num_edges * hidden_dim);

    // Step 2+3: W11 - Fused blocked GEMM + bias + GELU
    // [L*k, 3*hidden_dim] * [3*hidden_dim, hidden_dim] -> [L*k, hidden_dim]
    // Optimal block size for [192, 64]: 256 rows (fits in 256KB L2 cache)
    {
        PROFILE_SCOPE("MPNN_batched_edge_W11_blocked");
        blocked_gemm_bias_gelu<ScalarBackend>(
            batched_inputs,             // [num_edges, 3*hidden_dim]
            weights.W11_weight.data(),  // [3*hidden_dim, hidden_dim]
            weights.W11_bias.data(),    // [hidden_dim]
            buffer_a,                   // [num_edges, hidden_dim]
            num_edges,                  // M
            hidden_dim,                 // N
            input_dim,                  // K = 3*hidden_dim = 192
            256                         // block_size
        );
    }

    // Step 4+5: W12 - Fused blocked GEMM + bias + GELU
    // [L*k, hidden_dim] * [hidden_dim, hidden_dim] -> [L*k, hidden_dim]
    // Reuse batched_inputs buffer (no longer needed after W11)
    {
        PROFILE_SCOPE("MPNN_batched_edge_W12_blocked");
        blocked_gemm_bias_gelu<ScalarBackend>(
            buffer_a,                   // [num_edges, hidden_dim]
            weights.W12_weight.data(),  // [hidden_dim, hidden_dim]
            weights.W12_bias.data(),    // [hidden_dim]
            batched_inputs,             // [num_edges, hidden_dim] (reuse buffer!)
            num_edges,                  // M
            hidden_dim,                 // N
            hidden_dim,                 // K
            256                         // block_size
        );
    }

    // Step 6+7: W13 - Fused blocked GEMM + bias (no GELU)
    // [L*k, hidden_dim] * [hidden_dim, hidden_dim] -> [L*k, hidden_dim]
    // Reuse buffer_a for final output
    {
        PROFILE_SCOPE("MPNN_batched_edge_W13_blocked");
        blocked_gemm_bias<ScalarBackend>(batched_inputs,  // [num_edges, hidden_dim] (from W12)
                                         weights.W13_weight.data(),  // [hidden_dim, hidden_dim]
                                         weights.W13_bias.data(),    // [hidden_dim]
                                         buffer_a,    // [num_edges, hidden_dim] (reuse buffer_a!)
                                         num_edges,   // M
                                         hidden_dim,  // N
                                         hidden_dim,  // K
                                         256          // block_size
        );
    }

    // Step 8: Residual connection + LayerNorm (batched)
    {
        PROFILE_SCOPE("MPNN_batched_edge_residual_norm");
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < k; j++) {
                int edge_idx = i * k + j;
                int neighbor_idx = neighbor_indices[edge_idx];

                if (neighbor_idx < 0)
                    continue;  // Skip invalid edges

                float* h_E_ij = h_E + edge_idx * hidden_dim;
                float* edge_update = buffer_a + edge_idx * hidden_dim;

                // Residual: h_E += edge_update
                for (int d = 0; d < hidden_dim; d++) {
                    h_E_ij[d] += edge_update[d];
                }

                // LayerNorm (in-place)
                pfalign::layer_norm::layer_norm_forward<ScalarBackend>(
                    h_E_ij, h_E_ij, weights.norm3_gamma.data(), weights.norm3_beta.data(),
                    hidden_dim);
            }
        }
    }
}

/**
 * Helper: Batched message MLP computation.
 *
 * Processes all L*k edges in a single batched operation instead of per-edge loops.
 * This transforms 36,000 tiny GEMMs into 3 large GEMMs per layer.
 *
 * @param h_V             Node embeddings [L * hidden_dim]
 * @param neighbor_embs   Neighbor embeddings [L * k * hidden_dim]
 * @param h_E             Edge embeddings [L * k * hidden_dim]
 * @param L               Number of nodes
 * @param k               Number of neighbors per node
 * @param hidden_dim      Hidden dimension size
 * @param weights         Layer weights
 * @param messages        Output messages [L * k * hidden_dim]
 * @param temp_arena      Arena for temporary allocations
 */
static void batched_message_mlp(const float* h_V, const float* neighbor_embs, const float* h_E,
                                int L, int k, int hidden_dim,
                                const MPNNWeights::LayerWeights& weights,
                                float* messages,  // Output
                                pfalign::memory::Arena* temp_arena) {
    PROFILE_SCOPE("MPNN_batched_message_mlp");

    const int num_edges = L * k;
    const int input_dim = 3 * hidden_dim;

    // Step 1: Prepare batched inputs [L*k, 3*hidden_dim]
    // Format: [h_i, h_j, e_ij] for each edge
    float* batched_inputs = temp_arena->allocate<float>(num_edges * input_dim);

    {
        PROFILE_SCOPE("MPNN_batched_input_preparation");
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < k; j++) {
                int edge_idx = i * k + j;
                float* input_row = batched_inputs + edge_idx * input_dim;

                // Concatenate [h_i, h_j, e_ij]
                // Order matches JAX reference: [h_i, e_ij, h_j]
                std::memcpy(input_row, h_V + i * hidden_dim,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
                std::memcpy(input_row + hidden_dim, h_E + edge_idx * hidden_dim,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
                std::memcpy(input_row + 2 * hidden_dim, neighbor_embs + edge_idx * hidden_dim,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
            }
        }
    }

    // Debug: Print first edge input (matches original code)
    static bool printed_first = false;
    if (!printed_first) {
        //        std::cout << "Message input segments (batched):\n";
        //        std::cout << "  h_V: ";
        for (int d = 0; d < 5; d++) {
            //            std::cout << batched_inputs[d] << " ";
        }
        //        std::cout << "\n  h_E: ";
        for (int d = 0; d < 5; d++) {
            //            std::cout << batched_inputs[2 * hidden_dim + d] << " ";
        }
        //        std::cout << "\n  h_neighbor: ";
        for (int d = 0; d < 5; d++) {
            //            std::cout << batched_inputs[hidden_dim + d] << " ";
        }
        //        std::cout << std::endl;
        printed_first = true;
    }

    // Step 2: Allocate intermediate buffer (double-buffering optimization)
    // Reuse batched_inputs after W1, and use single intermediate buffer
    float* buffer_a = temp_arena->allocate<float>(num_edges * hidden_dim);

    // Step 3+4: W1 - Fused blocked GEMM + bias + GELU
    // [L*k, 192] * [192, 64] -> [L*k, 64]
    // Optimal block size for [192, 64]: 256 rows (fits in 256KB L2 cache)
    {
        PROFILE_SCOPE("MPNN_batched_W1_blocked");
        blocked_gemm_bias_gelu<ScalarBackend>(batched_inputs,            // [num_edges, 192]
                                              weights.W1_weight.data(),  // [192, 64]
                                              weights.W1_bias.data(),    // [64]
                                              buffer_a,                  // [num_edges, 64]
                                              num_edges,                 // M
                                              hidden_dim,                // N
                                              input_dim,                 // K = 3*hidden_dim = 192
                                              256                        // block_size
        );
    }

    // Step 5+6: W2 - Fused blocked GEMM + bias + GELU
    // [L*k, 64] * [64, 64] -> [L*k, 64]
    // Reuse batched_inputs buffer (no longer needed after W1)
    {
        PROFILE_SCOPE("MPNN_batched_W2_blocked");
        blocked_gemm_bias_gelu<ScalarBackend>(buffer_a,                  // [num_edges, 64]
                                              weights.W2_weight.data(),  // [64, 64]
                                              weights.W2_bias.data(),    // [64]
                                              batched_inputs,  // [num_edges, 64] (reuse buffer!)
                                              num_edges,       // M
                                              hidden_dim,      // N
                                              hidden_dim,      // K
                                              256              // block_size
        );
    }

    // Step 7+8: W3 - Fused blocked GEMM + bias (no GELU)
    // [L*k, 64] * [64, 64] -> [L*k, 64]
    {
        PROFILE_SCOPE("MPNN_batched_W3_blocked");
        blocked_gemm_bias<ScalarBackend>(batched_inputs,            // [num_edges, 64] (from W2)
                                         weights.W3_weight.data(),  // [64, 64]
                                         weights.W3_bias.data(),    // [64]
                                         messages,    // [num_edges, 64] (final output)
                                         num_edges,   // M
                                         hidden_dim,  // N
                                         hidden_dim,  // K
                                         256          // block_size
        );
    }

    // Debug: Print sample messages (matches original code)
    static bool printed_messages = false;
    if (!printed_messages) {
        for (int j : {0, 1, 2, 10, 20}) {
            if (j < k) {
                //                std::cout << "Messages[0," << j << "][:5]: ";
                [[maybe_unused]] float* msg = messages + j * hidden_dim;
                for (int d = 0; d < 5; d++) {
                    //                    std::cout << msg[d] << " ";
                }
                //                std::cout << std::endl;
            }
        }
        printed_messages = true;
    }
}

/**
 * Helper: Message passing layer.
 *
 * For each node:
 * 1. Gather neighbor embeddings
 * 2. Compute messages: MLP(concat(h_i, h_j, e_ij))
 * 3. Aggregate messages: sum over neighbors
 * 4. Update node: residual + LayerNorm
 * 5. Position-wise FFN: residual + LayerNorm
 * 6. Update edges: MLP(concat(h_i_new, e_ij, h_j_new)) + residual + LayerNorm
 */
static void
message_passing_layer(float* h_V,  // Node embeddings [L * hidden_dim]
                      float* h_E,  // Edge embeddings [L * k * hidden_dim] (will be updated)
                      const int* neighbor_indices, int L, int k, int hidden_dim,
                      const MPNNWeights::LayerWeights& weights, float message_scale,
                      float* messages,                    // Workspace [L * k * hidden_dim]
                      float* h_temp,                      // Workspace [L * hidden_dim]
                      pfalign::memory::Arena* temp_arena  // Arena for temporary allocations
) {
    static bool printed_scale = false;
    if (!printed_scale) {
        //        std::cout << "message_scale: " << message_scale << std::endl;
        printed_scale = true;
    }
    // Step 1: Gather neighbor embeddings
    float* neighbor_embs = temp_arena->allocate<float>(L * k * hidden_dim);
    {
        PROFILE_SCOPE("MPNN_layer_gather");
        pfalign::gather::gather<ScalarBackend>(h_V, neighbor_indices, neighbor_embs, L, k,
                                               hidden_dim);
    }

    // Step 2: Compute messages using batched MLP
    // Processes all L*k edges in 3 large GEMMs instead of per-edge loops
    batched_message_mlp(h_V,            // Node embeddings
                        neighbor_embs,  // Neighbor embeddings
                        h_E,            // Edge embeddings
                        L, k, hidden_dim, weights,
                        messages,  // Output
                        temp_arena);

    // Step 3: Aggregate messages (sum over neighbors)
    {
        PROFILE_SCOPE("MPNN_layer_aggregate");
        float scale = 1.0f / message_scale;
        for (int i = 0; i < L; i++) {
            std::memset(h_temp + i * hidden_dim, 0,
                        static_cast<size_t>(hidden_dim) * sizeof(float));

            for (int j = 0; j < k; j++) {
                int neighbor_idx = neighbor_indices[i * k + j];
                if (neighbor_idx < 0)
                    continue;

                const float* msg = messages + (i * k + j) * hidden_dim;
                float* out = h_temp + i * hidden_dim;

                for (int d = 0; d < hidden_dim; d++) {
                    out[d] += msg[d] * scale;
                }
            }

            if (i == 0) {
                for (int d = 0; d < 5; d++) {
                    [[maybe_unused]] float sum = 0.0f;
                    for (int j = 0; j < k; j++) {
                        const float* msg = messages + (i * k + j) * hidden_dim;
                        sum += msg[d];
                    }
                    //                std::cout << "Aggregate check dim " << d << ": sum=" << sum
                    //                          << " scaled=" << sum / message_scale
                    //                          << " stored=" << h_temp[d] << std::endl;
                }
            }
        }

#ifndef NDEBUG
        static bool dumped_messages = false;
        if (!dumped_messages) {
            std::ofstream out("layer0_messages.bin", std::ios::binary);
            out.write(reinterpret_cast<const char*>(messages), L * k * hidden_dim * sizeof(float));
            out.close();
            dumped_messages = true;
        }

        static bool dumped_residual = false;
        if (!dumped_residual) {
            std::ofstream out_res("layer0_h_V_res.bin", std::ios::binary);
            out_res.write(reinterpret_cast<const char*>(h_temp), L * hidden_dim * sizeof(float));
            out_res.close();
            dumped_residual = true;
        }
#endif
    }  // End MPNN_layer_aggregate

    //    std::cout << "Aggregated messages h_temp[0][:5]: ";
    for (int d = 0; d < 5 && d < hidden_dim; d++) {
        //        std::cout << h_temp[d] << " ";
    }
    //    std::cout << std::endl;

    // Step 4: Residual + LayerNorm
    {
        PROFILE_SCOPE("MPNN_layer_residual_norm");
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < hidden_dim; d++) {
                h_temp[i * hidden_dim + d] += h_V[i * hidden_dim + d];
            }

            pfalign::layer_norm::layer_norm_forward<ScalarBackend>(
                h_temp + i * hidden_dim, h_V + i * hidden_dim, weights.norm1_gamma.data(),
                weights.norm1_beta.data(), hidden_dim);
        }
    }

#ifndef NDEBUG
    static bool dumped_norm1 = false;
    if (!dumped_norm1) {
        std::ofstream out_norm("layer0_h_V_norm1.bin", std::ios::binary);
        out_norm.write(reinterpret_cast<const char*>(h_V), L * hidden_dim * sizeof(float));
        out_norm.close();
        dumped_norm1 = true;
    }
#endif

    // Step 5: Position-wise FFN
    {
        PROFILE_SCOPE("MPNN_layer_ffn");
        float* ffn_hidden = temp_arena->allocate<float>(L * 4 * hidden_dim);

        // FFN input layer: [hidden_dim -> 4*hidden_dim]
        pfalign::gemm::gemm<ScalarBackend>(h_V, weights.ffn_W_in_weight.data(), ffn_hidden, L,
                                           4 * hidden_dim, hidden_dim, 1.0f, 0.0f);
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < 4 * hidden_dim; d++) {
                ffn_hidden[i * 4 * hidden_dim + d] += weights.ffn_W_in_bias[d];
            }
        }
        gelu(ffn_hidden, L * 4 * hidden_dim);

#ifndef NDEBUG
        static bool dumped_ffn_in = false;
        if (!dumped_ffn_in) {
            std::ofstream out_ffn_in("layer0_ffn_in.bin", std::ios::binary);
            out_ffn_in.write(reinterpret_cast<const char*>(ffn_hidden),
                             L * 4 * hidden_dim * sizeof(float));
            out_ffn_in.close();
            //        std::cout << "Dumped layer0_ffn_in.bin" << std::endl;
            dumped_ffn_in = true;
        }
#endif

        // FFN output layer: [4*hidden_dim -> hidden_dim]
        pfalign::gemm::gemm<ScalarBackend>(ffn_hidden, weights.ffn_W_out_weight.data(), h_temp, L,
                                           hidden_dim, 4 * hidden_dim, 1.0f, 0.0f);

#ifndef NDEBUG
        static bool dumped_ffn_linear = false;
        if (!dumped_ffn_linear) {
            std::ofstream out_ffn_lin("layer0_ffn_linear.bin", std::ios::binary);
            out_ffn_lin.write(reinterpret_cast<const char*>(h_temp),
                              L * hidden_dim * sizeof(float));
            out_ffn_lin.close();
            dumped_ffn_linear = true;
        }
#endif
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < hidden_dim; d++) {
                h_temp[i * hidden_dim + d] += weights.ffn_W_out_bias[d] + h_V[i * hidden_dim + d];
            }

            // LayerNorm
            pfalign::layer_norm::layer_norm_forward<ScalarBackend>(
                h_temp + i * hidden_dim, h_V + i * hidden_dim, weights.norm2_gamma.data(),
                weights.norm2_beta.data(), hidden_dim);
        }
    }  // End MPNN_layer_ffn

#ifndef NDEBUG
    static bool dumped_ffn_out = false;
    if (!dumped_ffn_out) {
        std::ofstream out_ffn_out("layer0_ffn_out.bin", std::ios::binary);
        out_ffn_out.write(reinterpret_cast<const char*>(h_temp), L * hidden_dim * sizeof(float));
        out_ffn_out.close();
        //        std::cout << "Dumped layer0_ffn_out.bin" << std::endl;
        dumped_ffn_out = true;
    }

    static bool dumped_res2 = false;
    if (!dumped_res2) {
        std::ofstream out_res2("layer0_h_V_res2.bin", std::ios::binary);
        out_res2.write(reinterpret_cast<const char*>(h_V), L * hidden_dim * sizeof(float));
        out_res2.close();
        //        std::cout << "Dumped layer0_h_V_res2.bin" << std::endl;
        dumped_res2 = true;
    }
#endif

    // Step 6: Edge update (after node update is complete)
    // Use batched version - processes all L*k edges in 3 large GEMMs
    {
        PROFILE_SCOPE("MPNN_layer_edge_update");
        batched_edge_update(h_V, h_E, neighbor_indices, L, k, hidden_dim, weights, temp_arena);
    }

#ifndef NDEBUG
    static bool dumped_edge = false;
    if (!dumped_edge) {
        std::ofstream out_edge("layer0_h_E_after.bin", std::ios::binary);
        out_edge.write(reinterpret_cast<const char*>(h_E), L * k * hidden_dim * sizeof(float));
        out_edge.close();
        dumped_edge = true;
    }
#endif

    // No cleanup needed - Arena handles automatic deallocation
}

/**
 * MPNN forward pass (scalar implementation).
 */
template <>
void mpnn_forward<ScalarBackend>(const float* coords, int L, const MPNNWeights& weights,
                                 const MPNNConfig& config, float* node_emb, void* workspace_ptr) {
    if (coords == nullptr) {
        throw std::invalid_argument("mpnn_forward: coords pointer is null");
    }
    if (node_emb == nullptr) {
        throw std::invalid_argument("mpnn_forward: node_emb pointer is null");
    }
    if (workspace_ptr == nullptr) {
        throw std::invalid_argument("mpnn_forward: workspace pointer is null");
    }
    if (L <= 0) {
        throw std::invalid_argument("mpnn_forward: sequence length L must be positive");
    }
    if (config.hidden_dim <= 0) {
        throw std::invalid_argument("mpnn_forward: config.hidden_dim must be positive");
    }
    if (config.k_neighbors <= 0) {
        throw std::invalid_argument("mpnn_forward: config.k_neighbors must be positive");
    }
    if (config.num_layers <= 0) {
        throw std::invalid_argument("mpnn_forward: config.num_layers must be positive");
    }
    if (config.message_scale <= 0.0f) {
        throw std::invalid_argument("mpnn_forward: config.message_scale must be positive");
    }

    MPNNWorkspace* ws = static_cast<MPNNWorkspace*>(workspace_ptr);

    if (ws->L < L) {
        std::ostringstream oss;
        oss << "mpnn_forward: workspace length (" << ws->L
            << ") is smaller than requested sequence length (" << L << ")";
        throw std::invalid_argument(oss.str());
    }
    if (ws->hidden_dim != config.hidden_dim) {
        std::ostringstream oss;
        oss << "mpnn_forward: workspace hidden_dim (" << ws->hidden_dim
            << ") does not match config.hidden_dim (" << config.hidden_dim << ")";
        throw std::invalid_argument(oss.str());
    }
    if (ws->k < config.k_neighbors) {
        std::ostringstream oss;
        oss << "mpnn_forward: workspace k_neighbors (" << ws->k
            << ") is smaller than config.k_neighbors (" << config.k_neighbors << ")";
        throw std::invalid_argument(oss.str());
    }
    if (weights.num_layers != config.num_layers) {
        std::ostringstream oss;
        oss << "mpnn_forward: weights.num_layers (" << weights.num_layers
            << ") does not match config.num_layers (" << config.num_layers << ")";
        throw std::invalid_argument(oss.str());
    }

    // Validate coordinates are finite
    const std::size_t coord_count = static_cast<std::size_t>(L) * 4u * 3u;
    validate_finite_array(coords, coord_count, "mpnn_forward: coords");

    // Reset arena at start of forward pass (reuse memory from previous call)
    ws->temp_arena.reset();

    // Step 1: Extract atom coordinates
    {
        PROFILE_SCOPE("MPNN_extract_atoms");
        extract_atoms(coords, L, ws->Ca, ws->N, ws->C, ws->O, ws->Cb);
    }

    // Step 2: KNN search on Ca atoms
    {
        PROFILE_SCOPE("MPNN_knn_search");
        pfalign::knn::knn_search<ScalarBackend>(ws->Ca, L, config.k_neighbors, ws->neighbor_indices,
                                                ws->neighbor_distances_sq);
    }

    //    std::cout << "Neighbor indices row 0: ";
    for (int idx = 0; idx < config.k_neighbors; idx++) {
        //        std::cout << ws->neighbor_indices[idx] << " ";
    }
    //    std::cout << std::endl;

    // Step 3: Compute RBF features for all 25 atom pairs
    {
        PROFILE_SCOPE("MPNN_rbf_features");
        compute_edge_rbf_features(ws->Ca, ws->N, ws->C, ws->O, ws->Cb, ws->neighbor_indices, L,
                                  config.k_neighbors, config, ws->rbf_features);
    }

    //    std::cout << "\n=== V2 Early Stage Values ===" << std::endl;
    DEBUG_PRINT_ARRAY("RBF[0,0,0:5]", ws->rbf_features, 5);

    // Step 4: Edge embedding (batched)
    // Use batched version - processes all L*k edges in single GEMM
    // Input: [positional(16), RBF(25 * num_rbf)] = 416 features (matching V1)
    // Output: Edge embeddings [L * k * hidden_dim]
    {
        PROFILE_SCOPE("MPNN_edge_embedding");
        [[maybe_unused]] const int rbf_features = 25 * config.num_rbf;  // 400

        batched_edge_embedding(ws->rbf_features,      // [L*k, 400]
                               ws->neighbor_indices,  // [L*k]
                               ws->residue_idx,       // [L]
                               ws->chain_labels,      // [L]
                               L, config.k_neighbors, config.hidden_dim,
                               weights.edge_embedding_weight.data(),  // [416, hidden_dim]
                               weights.positional_weight.data(),      // [66, 16]
                               weights.positional_bias.data(),        // [16]
                               ws->edge_emb,                          // Output: [L*k, hidden_dim]
                               &ws->temp_arena);
    }  // End MPNN_edge_embedding

    // Debug: print edge_raw (after linear, before norm)
    DEBUG_PRINT_ARRAY("edge_raw[0,0,0:5]", ws->edge_emb, 5);

    // Apply LayerNorm to all edges
    {
        PROFILE_SCOPE("MPNN_edge_layer_norm");
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < config.k_neighbors; j++) {
                float* edge = ws->edge_emb + (i * config.k_neighbors + j) * config.hidden_dim;

                // LayerNorm
                pfalign::layer_norm::layer_norm_forward<ScalarBackend>(
                    edge, edge, weights.edge_norm_gamma.data(), weights.edge_norm_beta.data(),
                    config.hidden_dim);
            }
        }
    }

    // Debug: print after norm (before W_e)
    DEBUG_PRINT_ARRAY("after_norm[0,0,0:5]", ws->edge_emb, 5);

    // Step 4b: Apply W_e transformation to edge embeddings
    // This matches V1's h_E = W_e(E) step before message passing
    {
        PROFILE_SCOPE("MPNN_edge_transform");
        float* edge_temp = ws->temp_arena.allocate<float>(config.hidden_dim);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < config.k_neighbors; j++) {
                float* edge = ws->edge_emb + (i * config.k_neighbors + j) * config.hidden_dim;

                // W_e: [hidden_dim, hidden_dim]
                std::memcpy(edge_temp, edge,
                            static_cast<size_t>(config.hidden_dim) * sizeof(float));
                pfalign::gemm::gemm<ScalarBackend>(edge_temp, weights.W_e_weight.data(), edge, 1,
                                                   config.hidden_dim, config.hidden_dim, 1.0f,
                                                   0.0f);

                // Add bias
                for (int d = 0; d < config.hidden_dim; d++) {
                    edge[d] += weights.W_e_bias[d];
                }
            }
        }
    }

#ifndef NDEBUG
    static bool dumped_edge_before = false;
    if (!dumped_edge_before) {
        std::ofstream out_edge_before("layer0_h_E_before.bin", std::ios::binary);
        out_edge_before.write(reinterpret_cast<const char*>(ws->edge_emb),
                              L * config.k_neighbors * config.hidden_dim * sizeof(float));
        out_edge_before.close();
        dumped_edge_before = true;
    }
#endif

    //    std::cout << "\n=== V2 Intermediate Values ===" << std::endl;
    DEBUG_PRINT_ARRAY("h_E_init[0,0]", ws->edge_emb, config.hidden_dim);

    // Step 5: Initialize node embeddings to zero
    std::memset(ws->h_V, 0, static_cast<size_t>(L * config.hidden_dim) * sizeof(float));

    // Step 6: Message passing layers
    for (int layer = 0; layer < config.num_layers; layer++) {
        {
            PROFILE_SCOPE("MPNN_message_pass_layer");
            message_passing_layer(ws->h_V, ws->edge_emb, ws->neighbor_indices, L,
                                  config.k_neighbors, config.hidden_dim, weights.layers[layer],
                                  config.message_scale, ws->messages, ws->h_temp, &ws->temp_arena);
        }

        //        std::cout << "After layer " << layer << ":" << std::endl;
        DEBUG_PRINT_ARRAY("  h_V", ws->h_V, config.hidden_dim);
        DEBUG_PRINT_ARRAY("  h_E[0,0]", ws->edge_emb, config.hidden_dim);
    }

    // Step 7: Copy final embeddings to output
    std::memcpy(node_emb, ws->h_V, static_cast<size_t>(L * config.hidden_dim) * sizeof(float));

    DEBUG_PRINT_ARRAY("h_V_final", node_emb, config.hidden_dim);
    //    std::cout << "==============================\n" << std::endl;
}

}  // namespace mpnn
}  // namespace pfalign
