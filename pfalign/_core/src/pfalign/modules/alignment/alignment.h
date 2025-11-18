/**
 * Alignment module for protein sequence alignment.
 *
 * Provides clean interface to smith_waterman primitive with multiple modes:
 * - JAX-style (regular/affine/affine_flexible)
 * - Direct-style (regular/affine/affine_flexible)
 *
 * This module wraps the smith_waterman primitive similar to how
 * similarity wraps the gemm primitive.
 */

#pragma once

#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace alignment {

/**
 * Alignment mode selection.
 */
enum class AlignmentMode {
    JAX_REGULAR,            // JAX-style local alignment, regular gaps
    JAX_AFFINE,             // JAX-style local alignment, affine gaps (standard)
    JAX_AFFINE_FLEXIBLE,    // JAX-style local alignment, affine gaps (flexible)
    DIRECT_REGULAR,         // Direct local alignment, regular gaps
    DIRECT_AFFINE,          // Direct local alignment, affine gaps (standard)
    DIRECT_AFFINE_FLEXIBLE  // Direct local alignment, affine gaps (flexible)
};

/**
 * Alignment configuration.
 */
struct AlignmentConfig {
    AlignmentMode mode;
    smith_waterman::SWConfig sw_config;

    AlignmentConfig() : mode(AlignmentMode::JAX_AFFINE_FLEXIBLE) {
    }
};

/**
 * Compute alignment between two sequences given their similarity scores.
 *
 * This is a thin wrapper around the smith_waterman primitive that:
 * - Selects the appropriate SW variant based on mode
 * - Manages workspace allocation
 * - Provides clean interface for callers
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of first sequence
 * @param L2            Length of second sequence
 * @param config        Alignment configuration
 * @param dp_matrix     Output DP matrix (caller allocated)
 * @param partition     Output partition function (scalar)
 * @param temp_arena    Optional arena for temporary allocations
 *
 * DP matrix size depends on mode:
 * - JAX modes: L1 * L2 (regular) or L1 * L2 * 3 (affine)
 * - Direct modes: (L1+1) * (L2+1) (regular) or (L1+1) * (L2+1) * 3 (affine)
 *
 * Example:
 *   float scores[100 * 150];  // Similarity matrix
 *   float dp[100 * 150 * 3];  // DP matrix for affine mode
 *   float partition;
 *   AlignmentConfig config;
 *   config.mode = AlignmentMode::JAX_AFFINE_FLEXIBLE;
 *   compute_alignment<ScalarBackend>(scores, 100, 150, config, dp, &partition);
 */
template <typename Backend>
void compute_alignment(const float* scores, int L1, int L2, const AlignmentConfig& config,
                       float* dp_matrix, float* partition,
                       pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Compute alignment with backward pass (posteriors).
 *
 * Runs both forward (DP) and backward pass to compute posterior
 * alignment probabilities P(i aligns to j).
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of first sequence
 * @param L2            Length of second sequence
 * @param config        Alignment configuration
 * @param dp_matrix     Output DP matrix (caller allocated)
 * @param posteriors    Output posterior probabilities [L1 * L2]
 * @param partition     Output partition function (scalar)
 * @param temp_arena    Optional arena for temporary allocations
 *
 * Example:
 *   float scores[100 * 150];
 *   float dp[100 * 150 * 3];
 *   float posteriors[100 * 150];
 *   float partition;
 *   compute_alignment_with_posteriors<ScalarBackend>(
 *       scores, 100, 150, config, dp, posteriors, &partition);
 */
template <typename Backend>
void compute_alignment_with_posteriors(const float* scores, int L1, int L2,
                                       const AlignmentConfig& config, float* dp_matrix,
                                       float* posteriors, float* partition,
                                       pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Get required DP matrix size for a given mode.
 *
 * Utility function to help callers allocate the right amount of memory.
 *
 * @param L1    Length of first sequence
 * @param L2    Length of second sequence
 * @param mode  Alignment mode
 * @return      Required size in floats
 */
inline size_t get_dp_matrix_size(int L1, int L2, AlignmentMode mode) {
    bool is_jax = (mode == AlignmentMode::JAX_REGULAR || mode == AlignmentMode::JAX_AFFINE ||
                   mode == AlignmentMode::JAX_AFFINE_FLEXIBLE);
    bool is_affine = (mode != AlignmentMode::JAX_REGULAR && mode != AlignmentMode::DIRECT_REGULAR);

    if (is_jax) {
        return is_affine ? static_cast<size_t>(L1 * L2 * 3) : static_cast<size_t>(L1 * L2);
    } else {
        return is_affine ? static_cast<size_t>((L1 + 1) * (L2 + 1) * 3)
                         : static_cast<size_t>((L1 + 1) * (L2 + 1));
    }
}

}  // namespace alignment
}  // namespace pfalign
