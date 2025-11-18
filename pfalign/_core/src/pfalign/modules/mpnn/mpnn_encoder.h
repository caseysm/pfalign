#pragma once

#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/common/growable_arena.h"
#include "pfalign/common/arena_allocator.h"
#include <vector>
#include <stdexcept>
#include <limits>
#include <string>

namespace pfalign {
namespace mpnn {

/**
 * MPNN (Message Passing Neural Network) Encoder for protein structure.
 *
 * Architecture:
 * 1. Extract atom coordinates (Ca, N, C, O, Cb)
 * 2. KNN search on Ca atoms -> find k nearest neighbors
 * 3. Compute RBF features for 25 atom-pair distances
 * 4. Edge embedding: Linear(RBF) + LayerNorm
 * 5. Message passing layers (3 layers):
 *    - Gather neighbor embeddings
 *    - MLP: W1 -> GELU -> W2 -> GELU -> W3
 *    - Sum messages -> residual -> LayerNorm
 *    - Position-wise FFN -> residual -> LayerNorm
 * 6. Output: node embeddings [L * hidden_dim]
 *
 * Reference:
 *   align/_jax_reference/mpnn.py
 */

/**
 * MPNN configuration parameters.
 */
struct MPNNConfig {
    int hidden_dim;       // Hidden dimension (default: 64)
    int num_layers;       // Number of message passing layers (default: 3)
    int num_rbf;          // Number of RBF bins (default: 16)
    int k_neighbors;      // Number of neighbors (default: 30)
    float rbf_min_dist;   // RBF minimum distance in Å (default: 2.0)
    float rbf_max_dist;   // RBF maximum distance in Å (default: 22.0)
    float message_scale;  // Message aggregation scale (default: 30.0)

    // Default constructor
    MPNNConfig()
        : hidden_dim(64),
          num_layers(3),
          num_rbf(16),
          k_neighbors(30),
          rbf_min_dist(2.0f),
          rbf_max_dist(22.0f),
          message_scale(30.0f) {
    }
};

/**
 * MPNN learned parameters (weights).
 *
 * All weights are in row-major format.
 * These would be loaded from a checkpoint or randomly initialized.
 *
 * Uses std::vector for automatic memory management (RAII).
 */
struct MPNNWeights {
    // Edge embedding (from RBF features)
    std::vector<float> edge_embedding_weight;  // [edge_features * hidden_dim] (NO bias in V1)
    std::vector<float> edge_norm_gamma;        // [hidden_dim]
    std::vector<float> edge_norm_beta;         // [hidden_dim]

    // Positional encoding linear layer (66 -> 16)
    std::vector<float> positional_weight;  // [66 * 16]
    std::vector<float> positional_bias;    // [16]

    // Initial edge transformation W_e (before message passing)
    std::vector<float> W_e_weight;  // [hidden_dim * hidden_dim]
    std::vector<float> W_e_bias;    // [hidden_dim]

    // Message passing layer (per layer)
    struct LayerWeights {
        // Node message MLP (W1/W2/W3)
        std::vector<float> W1_weight;  // [3*hidden_dim * hidden_dim]
        std::vector<float> W1_bias;    // [hidden_dim]
        std::vector<float> W2_weight;  // [hidden_dim * hidden_dim]
        std::vector<float> W2_bias;    // [hidden_dim]
        std::vector<float> W3_weight;  // [hidden_dim * hidden_dim]
        std::vector<float> W3_bias;    // [hidden_dim]

        // Node layer norms
        std::vector<float> norm1_gamma;  // [hidden_dim]
        std::vector<float> norm1_beta;   // [hidden_dim]
        std::vector<float> norm2_gamma;  // [hidden_dim]
        std::vector<float> norm2_beta;   // [hidden_dim]

        // Position-wise FFN
        std::vector<float> ffn_W_in_weight;   // [hidden_dim * (4*hidden_dim)]
        std::vector<float> ffn_W_in_bias;     // [4*hidden_dim]
        std::vector<float> ffn_W_out_weight;  // [(4*hidden_dim) * hidden_dim]
        std::vector<float> ffn_W_out_bias;    // [hidden_dim]

        // Edge update MLP (W11/W12/W13)
        std::vector<float> W11_weight;  // [3*hidden_dim * hidden_dim]
        std::vector<float> W11_bias;    // [hidden_dim]
        std::vector<float> W12_weight;  // [hidden_dim * hidden_dim]
        std::vector<float> W12_bias;    // [hidden_dim]
        std::vector<float> W13_weight;  // [hidden_dim * hidden_dim]
        std::vector<float> W13_bias;    // [hidden_dim]

        // Edge layer norm (norm3 in V1)
        std::vector<float> norm3_gamma;  // [hidden_dim]
        std::vector<float> norm3_beta;   // [hidden_dim]
    };

    std::vector<LayerWeights> layers;  // Vector of layer weights
    int num_layers;

    // Constructor: Initialize layers vector with num_layers elements
    MPNNWeights(int num_layers_)
        : layers(static_cast<size_t>(num_layers_))  // Resize vector to hold num_layers_ elements
          ,
          num_layers(num_layers_) {
        // Vectors are default-initialized (empty), will be populated by loader
    }

    // Rule of Zero: std::vector handles all memory management
    // Compiler-generated destructor, copy, and move operations work correctly
    // No manual memory management needed!
};

/**
 * MPNN encoder forward pass.
 *
 * Takes protein backbone coordinates and produces node embeddings.
 *
 * @param coords        Backbone coordinates [L * 4 * 3] (N, Ca, C, O for each residue)
 * @param L             Sequence length (number of residues)
 * @param weights       Learned MPNN weights
 * @param config        MPNN configuration
 * @param node_emb      Output node embeddings [L * hidden_dim]
 * @param workspace     Temporary workspace (preallocated)
 *
 * Example:
 *   float coords[100 * 4 * 3];  // 100 residues, 4 atoms each (N, Ca, C, O)
 *   float node_emb[100 * 64];
 *   MPNNWorkspace workspace(100, 30, 64);
 *   mpnn_forward<ScalarBackend>(coords, 100, weights, config, node_emb, workspace);
 */
template <typename Backend>
void mpnn_forward(const float* coords, int L, const MPNNWeights& weights, const MPNNConfig& config,
                  float* node_emb, void* workspace);

/**
 * Compute dynamic arena size based on protein length.
 *
 * Arena memory requirements scale with:
 * - Edge embedding: ~58 KB per residue
 * - Per layer (3 layers): ~94 KB per residue per layer
 * - Total: 58 + (94 * 3) = 340 KB per residue
 *
 * We add 15% safety margin -> 400 KB per residue
 *
 * @param L  Sequence length (number of residues)
 * @param k  Number of neighbors (typically 30)
 * @return   Arena size in bytes
 */
inline size_t compute_arena_size(int L, [[maybe_unused]] int k,
                                 [[maybe_unused]] int num_layers = 3) {
    // Empirical formula: 400 KB per residue covers all allocations with margin
    constexpr size_t bytes_per_residue = 400 * 1024;
    constexpr size_t min_size = 96 * 1024 * 1024;  // Minimum 96 MB for small proteins

    size_t required = static_cast<size_t>(L) * bytes_per_residue;
    return std::max(min_size, required);
}

/**
 * Helper: Validate MPNNWorkspace parameters to prevent overflow and invalid allocations.
 *
 * Checks for:
 * - Non-positive values
 * - Integer overflow when computing buffer sizes
 * - Exceeding std::vector::max_size()
 */
inline void validate_parameters(int L, int k, int hidden_dim, int num_rbf) {
    // Check for non-positive values
    if (L <= 0) {
        throw std::invalid_argument("MPNNWorkspace: sequence length L must be positive, got " +
                                    std::to_string(L));
    }
    if (k <= 0) {
        throw std::invalid_argument("MPNNWorkspace: k_neighbors must be positive, got " +
                                    std::to_string(k));
    }
    if (hidden_dim <= 0) {
        throw std::invalid_argument("MPNNWorkspace: hidden_dim must be positive, got " +
                                    std::to_string(hidden_dim));
    }
    if (num_rbf <= 0) {
        throw std::invalid_argument("MPNNWorkspace: num_rbf must be positive, got " +
                                    std::to_string(num_rbf));
    }

    // Check for reasonable upper bounds (prevent accidental huge allocations)
    constexpr int MAX_SEQUENCE_LENGTH = 100000;  // 100K residues is ~13 MB protein
    constexpr int MAX_K_NEIGHBORS = 1000;        // 1000 neighbors is unreasonably large
    constexpr int MAX_HIDDEN_DIM = 1024;         // 1024 is way larger than typical 64
    constexpr int MAX_NUM_RBF = 128;             // 128 RBF bins is way larger than typical 16

    if (L > MAX_SEQUENCE_LENGTH) {
        throw std::invalid_argument("MPNNWorkspace: sequence length L=" + std::to_string(L) +
                                    " exceeds maximum " + std::to_string(MAX_SEQUENCE_LENGTH));
    }
    if (k > MAX_K_NEIGHBORS) {
        throw std::invalid_argument("MPNNWorkspace: k_neighbors=" + std::to_string(k) +
                                    " exceeds maximum " + std::to_string(MAX_K_NEIGHBORS));
    }
    if (hidden_dim > MAX_HIDDEN_DIM) {
        throw std::invalid_argument("MPNNWorkspace: hidden_dim=" + std::to_string(hidden_dim) +
                                    " exceeds maximum " + std::to_string(MAX_HIDDEN_DIM));
    }
    if (num_rbf > MAX_NUM_RBF) {
        throw std::invalid_argument("MPNNWorkspace: num_rbf=" + std::to_string(num_rbf) +
                                    " exceeds maximum " + std::to_string(MAX_NUM_RBF));
    }
}

/**
 * Helper: Safely compute allocation size with overflow checking.
 *
 * Returns the size needed for a buffer, or throws if overflow would occur.
 */
inline size_t safe_size(int dim1, int dim2, const char* buffer_name) {
    // Use size_t for overflow detection
    size_t size = static_cast<size_t>(dim1) * static_cast<size_t>(dim2);

    // Check for overflow (if result wraps around to smaller value)
    if (size / dim2 != static_cast<size_t>(dim1)) {
        throw std::overflow_error(
            std::string("MPNNWorkspace: integer overflow computing size for ") + buffer_name +
            " (" + std::to_string(dim1) + " * " + std::to_string(dim2) + ")");
    }

    // Check against vector max_size (typically SIZE_MAX / sizeof(T))
    constexpr size_t MAX_FLOAT_VECTOR_SIZE = std::numeric_limits<size_t>::max() / sizeof(float);

    // Assume float buffer (more restrictive check)
    if (size > MAX_FLOAT_VECTOR_SIZE) {
        throw std::length_error(std::string("MPNNWorkspace: buffer size for ") + buffer_name +
                                " (" + std::to_string(size) +
                                ") exceeds std::vector<float>::max_size()");
    }

    return size;
}

/**
 * Helper: Safely compute RBF feature buffer size with overflow checking.
 *
 * RBF features have shape [L * k * 25 * num_rbf]
 */
inline size_t safe_size_rbf(int L, int k, int num_rbf) {
    // Compute step-by-step with explicit size_t multiplication to avoid int overflow
    size_t L_k = static_cast<size_t>(L) * static_cast<size_t>(k);
    size_t L_k_25 = L_k * 25;
    size_t total = L_k_25 * static_cast<size_t>(num_rbf);

    // Check for overflow
    if (total / num_rbf != L_k_25 || L_k_25 / 25 != L_k || L_k / k != static_cast<size_t>(L)) {
        throw std::overflow_error(
            "MPNNWorkspace: integer overflow computing size for rbf_features_vec");
    }

    return total;
}

/**
 * MPNN workspace for temporary allocations.
 *
 * Pre-allocates all buffers needed for MPNN forward pass.
 * Avoids dynamic allocation in hot path.
 */
struct MPNNWorkspace {
    // Dimensions (initialized before buffer allocations)
    int L;
    int k;
    int hidden_dim;
    int num_rbf;

    // Atom coordinates (RAII with std::vector)
    std::vector<float> Ca_vec;  // [L * 3]
    std::vector<float> N_vec;   // [L * 3]
    std::vector<float> C_vec;   // [L * 3]
    std::vector<float> O_vec;   // [L * 3]
    std::vector<float> Cb_vec;  // [L * 3]

    // KNN results
    std::vector<int> neighbor_indices_vec;         // [L * k]
    std::vector<float> neighbor_distances_sq_vec;  // [L * k]

    // RBF features
    std::vector<float> rbf_features_vec;  // [L * k * (25 * num_rbf)]

    // Edge embeddings
    std::vector<float> edge_emb_vec;  // [L * k * hidden_dim]

    // Message passing buffers
    std::vector<float> h_V_vec;       // Node embeddings [L * hidden_dim]
    std::vector<float> h_E_vec;       // Edge embeddings [L * k * hidden_dim]
    std::vector<float> messages_vec;  // Message buffer [L * k * hidden_dim]
    std::vector<float> h_temp_vec;    // Temporary buffer [L * hidden_dim]

    // Positional metadata
    std::vector<int> residue_idx_vec;   // [L]
    std::vector<int> chain_labels_vec;  // [L]

    // Arena for temporary allocations (RAII)
    pfalign::memory::Arena temp_arena;

    // Raw pointers for backward compatibility with existing code
    float* Ca = nullptr;
    float* N = nullptr;
    float* C = nullptr;
    float* O = nullptr;
    float* Cb = nullptr;
    int* neighbor_indices = nullptr;
    float* neighbor_distances_sq = nullptr;
    float* rbf_features = nullptr;
    float* edge_emb = nullptr;
    float* h_V = nullptr;
    float* h_E = nullptr;
    float* messages = nullptr;
    float* h_temp = nullptr;
    int* residue_idx = nullptr;
    int* chain_labels = nullptr;

    MPNNWorkspace(int L_, int k_, int hidden_dim_, int num_rbf_ = 16)
        : L([&]() {
              validate_parameters(L_, k_, hidden_dim_, num_rbf_);
              return L_;
          }()),
          k(k_),
          hidden_dim(hidden_dim_),
          num_rbf(num_rbf_)
          // Allocate and zero-initialize all vectors (RAII - no leaks!)
          // Use constructor parameters (L_, k_, hidden_dim_, num_rbf_) when computing sizes.
          ,
          Ca_vec(safe_size(L_, 3, "Ca_vec"), 0.0f),
          N_vec(safe_size(L_, 3, "N_vec"), 0.0f),
          C_vec(safe_size(L_, 3, "C_vec"), 0.0f),
          O_vec(safe_size(L_, 3, "O_vec"), 0.0f),
          Cb_vec(safe_size(L_, 3, "Cb_vec"), 0.0f),
          neighbor_indices_vec(safe_size(L_, k_, "neighbor_indices_vec"), 0),
          neighbor_distances_sq_vec(safe_size(L_, k_, "neighbor_distances_sq_vec"), 0.0f),
          rbf_features_vec(safe_size_rbf(L_, k_, num_rbf_), 0.0f),
          edge_emb_vec(safe_size(L_ * k_, hidden_dim_, "edge_emb_vec"), 0.0f),
          h_V_vec(safe_size(L_, hidden_dim_, "h_V_vec"), 0.0f),
          h_E_vec(safe_size(L_ * k_, hidden_dim_, "h_E_vec"), 0.0f),
          messages_vec(safe_size(L_ * k_, hidden_dim_, "messages_vec"), 0.0f),
          h_temp_vec(safe_size(L_, hidden_dim_, "h_temp_vec"), 0.0f),
          residue_idx_vec(safe_size(L_, 1, "residue_idx_vec")),
          chain_labels_vec(safe_size(L_, 1, "chain_labels_vec"), 0),
          temp_arena(compute_arena_size(L_, k_))  // Dynamic arena size based on protein length
    {
        // Initialize raw pointers to vector data for backward compatibility
        Ca = Ca_vec.data();
        N = N_vec.data();
        C = C_vec.data();
        O = O_vec.data();
        Cb = Cb_vec.data();
        neighbor_indices = neighbor_indices_vec.data();
        neighbor_distances_sq = neighbor_distances_sq_vec.data();
        rbf_features = rbf_features_vec.data();
        edge_emb = edge_emb_vec.data();
        h_V = h_V_vec.data();
        h_E = h_E_vec.data();
        messages = messages_vec.data();
        h_temp = h_temp_vec.data();
        residue_idx = residue_idx_vec.data();
        chain_labels = chain_labels_vec.data();

        // Initialize residue indices
        for (int i = 0; i < L; i++) {
            residue_idx[i] = i;
        }
    }

    // Rule of Zero: std::vector handles all memory management automatically
    // No need for custom destructor, copy constructor, or assignment operators
};

}  // namespace mpnn
}  // namespace pfalign
