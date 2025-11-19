/**
 * Pairwise Protein Alignment - End-to-End Pipeline
 *
 * Chains together:
 * 1. Protein Coordinates -> MPNN Encoder -> Embeddings
 * 2. Two Embeddings -> Similarity Computation -> Similarity Matrix
 * 3. Similarity Matrix -> Smith-Waterman -> Alignment Partition
 *
 * This module provides the complete pairwise alignment pipeline
 * for the SoftAlign algorithm.
 */

#pragma once

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/adapters/alignment_types.h"
#include "pfalign/common/growable_arena.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace pfalign {
namespace io {
class Protein;
}
}  // namespace pfalign

namespace pfalign {
namespace pairwise {

/**
 * Configuration for pairwise alignment pipeline.
 * Combines MPNN and Smith-Waterman configurations.
 */
struct PairwiseConfig {
    mpnn::MPNNConfig mpnn_config;
    smith_waterman::SWConfig sw_config;

    // Smith-Waterman mode selection
    enum class SWMode {
        DIRECT_REGULAR,          // Textbook SW, single gap penalty
        DIRECT_AFFINE,           // Textbook SW, standard affine gaps
        DIRECT_AFFINE_FLEXIBLE,  // Textbook SW, flexible affine (I->D transitions)
        JAX_REGULAR,             // JAX formulation, single gap penalty
        JAX_AFFINE_STANDARD,     // JAX formulation, standard affine gaps
        JAX_AFFINE_FLEXIBLE      // JAX formulation, flexible affine (JAX-compatible)
    };
    SWMode sw_mode = SWMode::JAX_REGULAR;  // Default to JAX regular for training parity

    // When true, run the two per-sequence MPNN encoders in parallel threads.
    bool parallel_mpnn = true;

    PairwiseConfig() = default;
};

/**
 * Full alignment result structure.
 *
 * Contains all outputs from pairwise alignment including:
 * - Partition function (log-space alignment score)
 * - Normalized similarity score [-1, +1] (typically [0, 0.9] for aligned proteins)
 * - Posterior probability matrix
 * - Decoded alignment path with gaps
 *
 * Memory management:
 * - Caller allocates all buffers before calling pairwise_align_full()
 * - posteriors: [L1 * L2] probability matrix
 * - alignment_path: [max_path_length] typically L1 + L2
 *
 * Usage:
 * ```cpp
 * AlignmentResult result;
 * result.posteriors = arena.allocate<float>(L1 * L2);
 * result.alignment_path = arena.allocate<AlignmentPair>(L1 + L2);
 * result.max_path_length = L1 + L2;
 *
 * pairwise_align_full<ScalarBackend>(
 *     coords1, L1, coords2, L2,
 *     config, weights, workspace, &result, &arena
 * );
 *
 * // Access results
 * printf("Score: %.3f\n", result.score);
 * printf("Path length: %d\n", result.path_length);
 * for (int k = 0; k < result.path_length; k++) {
 *     printf("%d -> %d (p=%.3f)\n",
 *            result.alignment_path[k].i,
 *            result.alignment_path[k].j,
 *            result.alignment_path[k].posterior);
 * }
 * ```
 */
struct AlignmentResult {
    // Scalars
    float partition;  ///< Log-partition function from Smith-Waterman
    float score;      ///< Normalized alignment score ∈ [-1, +1] (posterior-weighted cosine)
    int L1;           ///< Length of sequence 1
    int L2;           ///< Length of sequence 2
    int path_length;  ///< Actual length of decoded alignment path

    // Buffers (caller-allocated)
    float* posteriors;              ///< Posterior matrix [L1 * L2] (user-allocated)
    AlignmentPair* alignment_path;  ///< Decoded alignment path [max_path_length]
    int max_path_length;            ///< Capacity of alignment_path buffer

    // Coordinate preservation (set by pipeline)
    const float* coords1;         ///< Pointer to original coordinates for sequence 1 (may be null)
    const float* coords2;         ///< Pointer to original coordinates for sequence 2 (may be null)
    std::string id1;              ///< Optional identifier for sequence 1
    std::string id2;              ///< Optional identifier for sequence 2
    const io::Protein* protein1;  ///< Optional metadata for sequence 1
    const io::Protein* protein2;  ///< Optional metadata for sequence 2

    AlignmentResult()
        : partition(0.0f),
          score(0.0f),
          L1(0),
          L2(0),
          path_length(0),
          posteriors(nullptr),
          alignment_path(nullptr),
          max_path_length(0),
          coords1(nullptr),
          coords2(nullptr),
          protein1(nullptr),
          protein2(nullptr) {
    }

    /**
     * Write aligned sequences to FASTA format using the decoded alignment.
     *
     * @param output_path Destination FASTA file path
     * @param seq1 Original sequence 1 (ungapped)
     * @param seq2 Original sequence 2 (ungapped)
     * @return true on success, false on I/O error
     */
    bool write_fasta(const std::string& output_path, const std::string& seq1,
                     const std::string& seq2) const;

    /**
     * Write superposed structures to a PDB file by aligning CA atoms via Kabsch.
     *
     * @param output_path Destination PDB path
     * @param reference Which sequence is the reference (0 = first, 1 = second)
     * @param arena Optional arena for temporary allocations
     * @return true on success, false on error
     */
    bool write_superposed_pdb(const std::string& output_path, int reference = 0,
                              pfalign::memory::GrowableArena* arena = nullptr) const;
};

/**
 * Workspace for pairwise alignment pipeline.
 * Pre-allocates all intermediate buffers to avoid malloc in hot path.
 *
 * Memory layout:
 * - MPNN workspaces for both proteins
 * - Embeddings for both proteins [L1 * D], [L2 * D]
 * - Similarity matrix [L1 * L2]
 * - Smith-Waterman DP matrix (size depends on mode)
 *
 * Usage:
 *   PairwiseWorkspace workspace(L1_max, L2_max, config);
 *   pairwise_align<ScalarBackend>(..., &workspace, ...);
 *   // Workspace auto-cleans on destruction
 */
struct PairwiseWorkspace {
    // Capacities (MUST be declared first, used in vector initialization)
    int L1_max;
    int L2_max;
    int hidden_dim;

    // MPNN workspaces (RAII with unique_ptr)
    std::unique_ptr<mpnn::MPNNWorkspace> mpnn_ws1_owner;
    std::unique_ptr<mpnn::MPNNWorkspace> mpnn_ws2_owner;

    // RAII storage (std::vector automatically manages memory)
    std::vector<float> embeddings1_vec;
    std::vector<float> embeddings2_vec;
    std::vector<float> similarity_vec;
    std::vector<float> sw_matrix_vec;
    std::vector<float> posteriors_vec;

    // Raw pointers for backward compatibility with existing code
    mpnn::MPNNWorkspace* mpnn_ws1 = nullptr;
    mpnn::MPNNWorkspace* mpnn_ws2 = nullptr;
    float* embeddings1 = nullptr;
    float* embeddings2 = nullptr;
    float* similarity = nullptr;
    float* sw_matrix = nullptr;
    float* posteriors = nullptr;

    static int compute_sw_matrix_size(int L1_max, int L2_max, PairwiseConfig::SWMode mode) {
        switch (mode) {
            case PairwiseConfig::SWMode::DIRECT_REGULAR:
                return (L1_max + 1) * (L2_max + 1);
            case PairwiseConfig::SWMode::DIRECT_AFFINE:
            case PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE:
                return (L1_max + 1) * (L2_max + 1) * 3;
            case PairwiseConfig::SWMode::JAX_REGULAR:
                return L1_max * L2_max;
            case PairwiseConfig::SWMode::JAX_AFFINE_STANDARD:
            case PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE:
                return L1_max * L2_max * 3;
        }
        return (L1_max + 1) * (L2_max + 1);
    }

    /**
     * Constructor: Pre-allocate all buffers with RAII.
     *
     * @param L1_max Maximum length for protein 1
     * @param L2_max Maximum length for protein 2
     * @param config Pipeline configuration
     * @param allocate_mpnn If true, allocate MPNN workspaces (default: true)
     *                      Set to false when aligning pre-computed embeddings
     *                      to save ~136MB per thread (96% memory reduction!)
     *
     * Exception safety: All allocations use RAII (unique_ptr/vector).
     * If any allocation fails, previously allocated resources are
     * automatically cleaned up.
     */
    PairwiseWorkspace(int L1_max, int L2_max, const PairwiseConfig& config,
                      bool allocate_mpnn = true)
        : L1_max(L1_max),
          L2_max(L2_max),
          hidden_dim(config.mpnn_config.hidden_dim)
          // RAII initialization: If any throws, previously constructed members auto-destruct
          ,
          embeddings1_vec(static_cast<size_t>(L1_max * hidden_dim), 0.0f),
          embeddings2_vec(static_cast<size_t>(L2_max * hidden_dim), 0.0f),
          similarity_vec(static_cast<size_t>(L1_max * L2_max), 0.0f),
          sw_matrix_vec(static_cast<size_t>(compute_sw_matrix_size(L1_max, L2_max, config.sw_mode)),
                        0.0f),
          posteriors_vec(static_cast<size_t>(L1_max * L2_max), 0.0f) {
        // Lazy MPNN workspace allocation (Issue #1 fix)
        // Only allocate when encoding structures, not when aligning embeddings
        // This saves ~136MB per thread in distance matrix computation!
        if (allocate_mpnn) {
            mpnn_ws1_owner = std::make_unique<mpnn::MPNNWorkspace>(
                L1_max, config.mpnn_config.k_neighbors, config.mpnn_config.hidden_dim,
                config.mpnn_config.num_rbf);
            mpnn_ws2_owner = std::make_unique<mpnn::MPNNWorkspace>(
                L2_max, config.mpnn_config.k_neighbors, config.mpnn_config.hidden_dim,
                config.mpnn_config.num_rbf);

            // Initialize raw pointers for backward compatibility
            mpnn_ws1 = mpnn_ws1_owner.get();
            mpnn_ws2 = mpnn_ws2_owner.get();
        }
        // else: mpnn_ws1/mpnn_ws2 remain nullptr (safe for embedding-only alignment)

        // Initialize buffer pointers (always needed)
        embeddings1 = embeddings1_vec.data();
        embeddings2 = embeddings2_vec.data();
        similarity = similarity_vec.data();
        sw_matrix = sw_matrix_vec.data();
        posteriors = posteriors_vec.data();
    }

    // Rule of Zero: unique_ptr and vector handle all cleanup automatically
    // No manual destructor needed!

    // Disable copy (workspace is large and non-copyable)
    PairwiseWorkspace(const PairwiseWorkspace&) = delete;
    PairwiseWorkspace& operator=(const PairwiseWorkspace&) = delete;
};

/**
 * Pairwise protein alignment - full pipeline.
 *
 * Chains: Coords -> MPNN -> Similarity -> Smith-Waterman
 *
 * @tparam Backend Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param coords1 Protein 1 coordinates [L1 * 14 * 3] (14 atom types)
 * @param L1 Length of protein 1
 * @param coords2 Protein 2 coordinates [L2 * 14 * 3]
 * @param L2 Length of protein 2
 * @param config Pipeline configuration (MPNN + SW)
 * @param weights MPNN weights (shared for both proteins)
 * @param workspace Pre-allocated workspace (must be large enough)
 * @param partition Output: alignment partition function (log-probability)
 */
template <typename Backend>
void pairwise_align(const float* coords1, int L1, const float* coords2, int L2,
                    const PairwiseConfig& config, const mpnn::MPNNWeights& weights,
                    PairwiseWorkspace* workspace, float* partition);

/**
 * Pairwise protein alignment - embeddings already computed.
 *
 * Chains: Embeddings -> Similarity -> Smith-Waterman
 *
 * Useful when embeddings are cached or computed separately.
 *
 * @tparam Backend Computation backend
 * @param embeddings1 Protein 1 embeddings [L1 * D]
 * @param L1 Length of protein 1
 * @param embeddings2 Protein 2 embeddings [L2 * D]
 * @param L2 Length of protein 2
 * @param hidden_dim Embedding dimension
 * @param config Pipeline configuration (only SW config used)
 * @param workspace Pre-allocated workspace (for similarity + SW)
 * @param partition Output: alignment partition function
 */
template <typename Backend>
void pairwise_align_from_embeddings(const float* embeddings1, int L1, const float* embeddings2,
                                    int L2, int hidden_dim, const PairwiseConfig& config,
                                    PairwiseWorkspace* workspace, float* partition);

/**
 * Pairwise protein alignment with score computation.
 *
 * Chains: Coords -> MPNN -> Similarity -> Smith-Waterman (forward + backward) -> Score
 *
 * Computes alignment score using the formula:
 *   score = sum(cosine_similarity ⊙ posteriors)
 *
 * Where:
 * - cosine_similarity[i,j] = dot_product[i,j] / (||e1[i]|| * ||e2[j]||) ∈ [-1, +1]
 * - posteriors[i,j] = P(i aligns to j | all alignment paths), normalized to sum to 1
 * - ⊙ is element-wise multiplication
 * - Score ∈ [-1, +1], with score=1 for identical proteins with perfect diagonal alignment
 * - Empirically, scores are typically positive ([0, 0.9]) for aligned proteins
 *
 * Optionally computes magnitude-aware score if score_magnitude is provided:
 *   score_mag = Sigma P[i,j] * dot[i,j] / Sigma P[i,j] * ||e1[i]|| * ||e2[j]||
 *
 * @tparam Backend Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param coords1 Protein 1 coordinates [L1 * 14 * 3]
 * @param L1 Length of protein 1
 * @param coords2 Protein 2 coordinates [L2 * 14 * 3]
 * @param L2 Length of protein 2
 * @param config Pipeline configuration (MPNN + SW)
 * @param weights MPNN weights (shared for both proteins)
 * @param workspace Pre-allocated workspace (must be large enough)
 * @param partition Output: alignment partition function (log-probability)
 * @param score Output: normalized cosine-based alignment score ∈ [-1, +1]
 * @param arena Arena for temporary allocations (backward pass scratch)
 * @param score_magnitude Optional output: magnitude-aware score (nullptr to skip)
 */
template <typename Backend>
void pairwise_align_with_score(const float* coords1, int L1, const float* coords2, int L2,
                               const PairwiseConfig& config, const mpnn::MPNNWeights& weights,
                               PairwiseWorkspace* workspace, float* partition, float* score,
                               pfalign::memory::GrowableArena* arena,
                               float* score_magnitude = nullptr);

/**
 * Pairwise protein alignment with score - embeddings already computed.
 *
 * Chains: Embeddings -> Similarity -> Smith-Waterman (forward + backward) -> Score
 *
 * Computes cosine-based alignment score by default. Optionally computes
 * magnitude-aware score if score_magnitude is provided.
 *
 * @tparam Backend Computation backend
 * @param embeddings1 Protein 1 embeddings [L1 * D]
 * @param L1 Length of protein 1
 * @param embeddings2 Protein 2 embeddings [L2 * D]
 * @param L2 Length of protein 2
 * @param hidden_dim Embedding dimension
 * @param config Pipeline configuration (only SW config used)
 * @param workspace Pre-allocated workspace (for similarity + SW)
 * @param partition Output: alignment partition function
 * @param score Output: normalized cosine-based alignment score ∈ [-1, +1]
 * @param arena Arena for temporary allocations (backward pass scratch)
 * @param score_magnitude Optional output: magnitude-aware score (nullptr to skip)
 */
template <typename Backend>
void pairwise_align_from_embeddings_with_score(const float* embeddings1, int L1,
                                               const float* embeddings2, int L2, int hidden_dim,
                                               const PairwiseConfig& config,
                                               PairwiseWorkspace* workspace, float* partition,
                                               float* score, pfalign::memory::GrowableArena* arena,
                                               float* score_magnitude = nullptr);

/**
 * Pairwise protein alignment with dual scoring metrics.
 *
 * Computes both cosine-based and magnitude-aware scores for comparison:
 *
 * score_cos = Sigma P[i,j] * cos[i,j]
 *   where cos[i,j] = dot[i,j] / (||e1[i]|| * ||e2[j]||)
 *   -> Pure directional similarity (magnitude-invariant)
 *
 * score_mag = Sigma P[i,j] * dot[i,j] / Sigma P[i,j] * ||e1[i]|| * ||e2[j]||
 *   -> Magnitude-aware weighted average of cosines
 *   -> Gives more influence to high-magnitude aligned pairs
 *
 * Design rationale:
 * - Dot product posteriors: magnitude influences which positions align
 * - score_cos: measures directional quality (magnitude already affected alignment)
 * - score_mag: retains magnitude signal in final score
 *
 * Both scores ∈ [-1, +1] (posterior-weighted cosine similarity).
 * Empirically, scores are usually positive ([0, 0.9] range) for aligned proteins.
 * Use this to empirically compare the metrics.
 *
 * @tparam Backend Computation backend
 * @param embeddings1 Protein 1 embeddings [L1 * D]
 * @param L1 Length of protein 1
 * @param embeddings2 Protein 2 embeddings [L2 * D]
 * @param L2 Length of protein 2
 * @param hidden_dim Embedding dimension
 * @param config Pipeline configuration (only SW config used)
 * @param workspace Pre-allocated workspace (for similarity + SW)
 * @param partition Output: alignment partition function
 * @param score_cosine Output: cosine-based score (direction-only)
 * @param score_magnitude Output: magnitude-aware score
 * @param arena Arena for temporary allocations (backward pass scratch)
 */
template <typename Backend>
void pairwise_align_from_embeddings_with_dual_score(const float* embeddings1, int L1,
                                                    const float* embeddings2, int L2,
                                                    int hidden_dim, const PairwiseConfig& config,
                                                    PairwiseWorkspace* workspace, float* partition,
                                                    float* score_cosine, float* score_magnitude,
                                                    pfalign::memory::GrowableArena* arena);

/**
 * Pairwise protein alignment with dual scoring metrics - from coordinates.
 *
 * Chains: Coords -> MPNN -> Embeddings -> Similarity -> SW -> Dual Scores
 *
 * @tparam Backend Computation backend
 * @param coords1 Protein 1 coordinates [L1 * 14 * 3]
 * @param L1 Length of protein 1
 * @param coords2 Protein 2 coordinates [L2 * 14 * 3]
 * @param L2 Length of protein 2
 * @param config Pipeline configuration (MPNN + SW)
 * @param weights MPNN weights (shared for both proteins)
 * @param workspace Pre-allocated workspace (must be large enough)
 * @param partition Output: alignment partition function
 * @param score_cosine Output: cosine-based score (direction-only)
 * @param score_magnitude Output: magnitude-aware score
 * @param arena Arena for temporary allocations (backward pass scratch)
 */
template <typename Backend>
void pairwise_align_with_dual_score(const float* coords1, int L1, const float* coords2, int L2,
                                    const PairwiseConfig& config, const mpnn::MPNNWeights& weights,
                                    PairwiseWorkspace* workspace, float* partition,
                                    float* score_cosine, float* score_magnitude,
                                    pfalign::memory::GrowableArena* arena);

/**
 * Full pairwise alignment with posteriors and decoded path.
 *
 * Chains: Embeddings -> Similarity -> SW (fwd+bwd) -> Score -> Decode -> AlignmentResult
 *
 * This is the most complete alignment function, returning:
 * - Partition function (log-space score)
 * - Alignment score (normalized, ∈ [-1, +1])
 * - Full posterior matrix [L1 * L2]
 * - Decoded alignment path with gaps
 *
 * Memory requirements:
 * - result->posteriors must be pre-allocated [L1 * L2]
 * - result->alignment_path must be pre-allocated [max_path_length]
 * - result->max_path_length must be set (typically L1 + L2)
 * - Arena for temporary buffers (decode DP matrices)
 *
 * Gap penalty for decoding:
 * - Default: config.sw_config.gap from forward pass
 * - Can be overridden via gap_penalty parameter
 *
 * @tparam Backend Computation backend
 * @param embeddings1 Protein 1 embeddings [L1 * D]
 * @param L1 Length of protein 1
 * @param embeddings2 Protein 2 embeddings [L2 * D]
 * @param L2 Length of protein 2
 * @param hidden_dim Embedding dimension
 * @param config Pipeline configuration (SW config)
 * @param workspace Pre-allocated workspace (for similarity + SW)
 * @param result Output: full alignment result (posteriors + path)
 * @param arena Arena for temporary allocations (backward pass + decode scratch)
 * @param gap_penalty Gap penalty for decoding (default: use config.sw_config.gap)
 */
template <typename Backend>
void pairwise_align_from_embeddings_full(
    const float* embeddings1, int L1, const float* embeddings2, int L2, int hidden_dim,
    const PairwiseConfig& config, PairwiseWorkspace* workspace, AlignmentResult* result,
    pfalign::memory::GrowableArena* arena,
    float gap_penalty = -2.0f  // Default gap penalty for decoding
);

/**
 * Full pairwise alignment with posteriors and decoded path - from coordinates.
 *
 * Chains: Coords -> MPNN -> Embeddings -> Full Alignment (posteriors + path)
 *
 * @tparam Backend Computation backend
 * @param coords1 Protein 1 coordinates [L1 * 14 * 3]
 * @param L1 Length of protein 1
 * @param coords2 Protein 2 coordinates [L2 * 14 * 3]
 * @param L2 Length of protein 2
 * @param config Pipeline configuration (MPNN + SW)
 * @param weights MPNN weights (shared for both proteins)
 * @param workspace Pre-allocated workspace
 * @param result Output: full alignment result (posteriors + path)
 * @param arena Arena for temporary allocations
 * @param gap_penalty Gap penalty for decoding (default: -2.0)
 */
template <typename Backend>
void pairwise_align_full(const float* coords1, int L1, const float* coords2, int L2,
                         const PairwiseConfig& config, const mpnn::MPNNWeights& weights,
                         PairwiseWorkspace* workspace, AlignmentResult* result,
                         pfalign::memory::GrowableArena* arena, float gap_penalty = -2.0f);

}  // namespace pairwise
}  // namespace pfalign
