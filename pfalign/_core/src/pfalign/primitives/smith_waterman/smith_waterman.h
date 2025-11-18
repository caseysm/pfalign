/**
 * Smith-Waterman alignment algorithm.
 *
 * Implementations:
 * 1. Direct regular: Standard LOCAL SW with single gap penalty
 * 2. Direct affine: Standard LOCAL SW with affine gaps (open + extend)
 * 3. JAX regular: SoftAlign formulation with single gap penalty
 * 4. JAX affine: SoftAlign formulation with affine gaps
 */

#pragma once

#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace smith_waterman {

constexpr float NINF = -1e30f;

/**
 * Smith-Waterman configuration.
 */
struct SWConfig {
    float gap;          // Gap penalty for regular mode (negative, e.g., -0.1)
    float gap_open;     // Gap opening penalty for affine mode (negative, e.g., -1.0)
    float gap_extend;   // Gap extension penalty for affine mode (negative, e.g., -0.1)
    float temperature;  // Temperature for soft max (default: 1.0)
    bool affine;        // Use affine gap penalties (default: false)

    SWConfig() : gap(-0.1f), gap_open(-1.0f), gap_extend(-0.1f), temperature(1.0f), affine(false) {
    }
};

/**
 * Standard Smith-Waterman (Direct mode) - Regular gap penalties.
 *
 * Single DP matrix with 4 transitions: Align, Gap_up, Gap_left, Sky.
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        SW configuration (uses config.gap)
 * @param alpha         Output DP matrix [L1+1 * L2+1] (caller allocated)
 * @param partition     Output partition function (scalar)
 *
 * Recurrence (1-indexed):
 *   alpha[i,j] = soft_max([
 *       alpha[i-1,j-1] + scores[i-1,j-1],  // Align
 *       alpha[i-1,j]   + gap,               // Gap_up
 *       alpha[i,j-1]   + gap,               // Gap_left
 *       scores[i-1,j-1]                     // Sky (LOCAL)
 *   ])
 */
template <typename Backend>
void smith_waterman_direct_regular(const float* scores, int L1, int L2, const SWConfig& config,
                                   float* alpha, float* partition);

/**
 * Standard Smith-Waterman (Direct mode) - Affine gap penalties.
 *
 * Three DP matrices: M (match), I (gap in seq1), D (gap in seq2).
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        SW configuration (uses config.gap_open, config.gap_extend)
 * @param alpha         Output DP matrices [L1+1 * L2+1 * 3] (M, I, D) (caller allocated)
 * @param partition     Output partition function (scalar)
 *
 * Recurrence (1-indexed):
 *   M[i,j] = scores[i-1,j-1] + soft_max([M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0])
 *   I[i,j] = soft_max([M[i-1,j] + gap_open, I[i-1,j] + gap_extend])
 *   D[i,j] = soft_max([M[i,j-1] + gap_open, D[i,j-1] + gap_extend])
 */
template <typename Backend>
void smith_waterman_direct_affine(const float* scores, int L1, int L2, const SWConfig& config,
                                  float* alpha, float* partition);

/**
 * Standard Smith-Waterman (Direct mode) - Affine gap penalties (FLEXIBLE model).
 *
 * Like smith_waterman_direct_affine but allows Right->Down transitions.
 * This enables switching from one gap type to another without returning
 * to the Match state first.
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        SW configuration (uses config.gap_open, config.gap_extend)
 * @param alpha         Output DP matrices [L1+1 * L2+1 * 3] (M, I, D) (caller allocated)
 * @param partition     Output partition function (scalar)
 *
 * Three states per position:
 * - M: Match/mismatch
 * - I: Gap in sequence 1 (moving right)
 * - D: Gap in sequence 2 (moving down)
 *
 * Flexible model allows:
 * - I->D transitions (Right->Down with gap_open penalty)
 */
template <typename Backend>
void smith_waterman_direct_affine_flexible(const float* scores, int L1, int L2,
                                           const SWConfig& config, float* alpha, float* partition);

/**
 * JAX-compatible Smith-Waterman - Regular gap penalties.
 *
 * Matches align/_jax_reference/smith_waterman.py::sw()
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        SW configuration (uses config.gap)
 * @param hij           Output DP matrix [L1 * L2] (caller allocated)
 * @param partition     Output partition function (scalar)
 *
 * Recurrence (0-indexed, for i∈[0,L1-1], j∈[0,L2-1]):
 *   hij[i,j] = soft_max([
 *       hij[i-1,j-1] + scores[i,j],  // Align (if i>0, j>0)
 *       hij[i-1,j]   + gap,           // Turn_0 (if i>0)
 *       hij[i,j-1]   + gap,           // Turn_1 (if j>0)
 *       scores[i,j]                   // Sky
 *   ])
 *
 *   Final: S = soft_max(hij[i,j] + scores[i+1,j+1]) for i<L1-1, j<L2-1
 */
template <typename Backend>
void smith_waterman_jax_regular(const float* scores, int L1, int L2, const SWConfig& config,
                                float* hij, float* partition);

/**
 * JAX-compatible Smith-Waterman - Affine gap penalties (STANDARD model).
 *
 * NOTE: This implements STANDARD affine gaps (textbook algorithm):
 * - Right state: Align->Right (open), Right->Right (extend)
 * - Down state: Align->Down (open), Down->Down (extend)
 * - NO cross-state transitions (Right↔Down)
 *
 * This does NOT match the JAX reference exactly (see jax_affine_flexible for that).
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        SW configuration (uses config.gap_open, config.gap_extend)
 * @param hij           Output DP matrices [L1 * L2 * 3] (Align, Right, Down) (caller allocated)
 * @param partition     Output partition function (scalar)
 *
 * Three states per position:
 * - Align: Match/mismatch
 * - Right: Gap in sequence 2 (moving right)
 * - Down: Gap in sequence 1 (moving down)
 *
 * Final: S = soft_max(hij[i,j,:] + scores[i+1,j+1]) for i<L1-1, j<L2-1
 */
template <typename Backend>
void smith_waterman_jax_affine(const float* scores, int L1, int L2, const SWConfig& config,
                               float* hij, float* partition);

/**
 * JAX-compatible Smith-Waterman - Affine gap penalties (FLEXIBLE model).
 *
 * This EXACTLY matches align/_jax_reference/smith_waterman.py::sw_affine()
 * with penalize_turns=True and restrict_turns=True (the defaults).
 *
 * FLEXIBLE model allows cross-state transitions:
 * - Right state: Align->Right (open), Right->Right (extend), [Down->Right restricted out]
 * - Down state: Align->Down (open), Right->Down (open), Down->Down (extend)
 *
 * This differs from standard affine gaps by allowing Right->Down transitions.
 *
 * @param scores        Similarity matrix [L1 * L2]
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        SW configuration (uses config.gap_open, config.gap_extend)
 * @param hij           Output DP matrices [L1 * L2 * 3] (Align, Right, Down) (caller allocated)
 * @param partition     Output partition function (scalar)
 *
 * Final: S = soft_max(hij[i,j,:] + scores[i+1,j+1]) for i<L1-1, j<L2-1
 */
template <typename Backend>
void smith_waterman_jax_affine_flexible(const float* scores, int L1, int L2, const SWConfig& config,
                                        float* hij, float* partition);

// ============================================================================
// BACKWARD PASSES (Compute Posteriors / Alignment Matrices)
// ============================================================================
//
// Each backward pass computes the gradient ∂partition/∂scores[i,j], which
// gives the posterior probability P(position i aligns to position j).
//
// This is what JAX's jax.value_and_grad() returns as the "grad" component.
// The output is always a 2D matrix [L1 * L2], regardless of the forward pass
// dimensionality (which may be 3D for affine modes).
//
// Usage:
//   1. Run forward pass to get DP values and partition
//   2. Run backward pass to get posteriors (2D alignment matrix)
//   3. Save posteriors as the final alignment output
//

/**
 * Backward pass for Direct Regular mode.
 *
 * Computes ∂partition/∂scores via backpropagation through the forward DP.
 *
 * @param alpha         [L1+1 * L2+1] DP values from forward pass
 * @param scores        [L1 * L2] input similarity matrix
 * @param L1, L2        Sequence lengths
 * @param config        SW configuration (temperature, gap)
 * @param partition     Scalar partition function from forward pass
 * @param posteriors    [L1 * L2] OUTPUT - posterior probabilities P(i~j)
 */
template <typename Backend>
void smith_waterman_direct_regular_backward(const float* alpha, const float* scores, int L1, int L2,
                                            const SWConfig& config, float partition,
                                            float* posteriors,
                                            pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Backward pass for Direct Affine mode.
 *
 * @param alpha         [L1+1 * L2+1 * 3] DP values (M, I, D) from forward pass
 * @param scores        [L1 * L2] input similarity matrix
 * @param L1, L2        Sequence lengths
 * @param config        SW configuration (temperature, gap_open, gap_extend)
 * @param partition     Scalar partition function from forward pass
 * @param posteriors    [L1 * L2] OUTPUT - posterior probabilities P(i~j)
 */
template <typename Backend>
void smith_waterman_direct_affine_backward(const float* alpha, const float* scores, int L1, int L2,
                                           const SWConfig& config, float partition,
                                           float* posteriors,
                                           pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Backward pass for Direct Affine Flexible mode.
 *
 * @param alpha         [L1+1 * L2+1 * 3] DP values (M, I, D) from forward pass
 * @param scores        [L1 * L2] input similarity matrix
 * @param L1, L2        Sequence lengths
 * @param config        SW configuration (temperature, gap_open, gap_extend)
 * @param partition     Scalar partition function from forward pass
 * @param posteriors    [L1 * L2] OUTPUT - posterior probabilities P(i~j)
 */
template <typename Backend>
void smith_waterman_direct_affine_flexible_backward(
    const float* alpha, const float* scores, int L1, int L2, const SWConfig& config,
    float partition, float* posteriors, pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Backward pass for JAX Regular mode.
 *
 * Matches the gradient computation from JAX's jax.value_and_grad(sco).
 *
 * @param hij           [L1 * L2] DP values from forward pass
 * @param scores        [L1 * L2] input similarity matrix
 * @param L1, L2        Sequence lengths
 * @param config        SW configuration (temperature, gap)
 * @param partition     Scalar partition function from forward pass
 * @param posteriors    [L1 * L2] OUTPUT - posterior probabilities P(i~j)
 */
template <typename Backend>
void smith_waterman_jax_regular_backward(const float* hij, const float* scores, int L1, int L2,
                                         const SWConfig& config, float partition, float* posteriors,
                                         pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Backward pass for JAX Affine mode.
 *
 * @param hij           [L1 * L2 * 3] DP values (Align, Right, Down) from forward
 * @param scores        [L1 * L2] input similarity matrix
 * @param L1, L2        Sequence lengths
 * @param config        SW configuration (temperature, gap_open, gap_extend)
 * @param partition     Scalar partition function from forward pass
 * @param posteriors    [L1 * L2] OUTPUT - posterior probabilities P(i~j)
 */
template <typename Backend>
void smith_waterman_jax_affine_backward(const float* hij, const float* scores, int L1, int L2,
                                        const SWConfig& config, float partition, float* posteriors,
                                        pfalign::memory::GrowableArena* temp_arena = nullptr);

/**
 * Backward pass for JAX Affine Flexible mode.
 *
 * Matches the gradient from JAX's jax.value_and_grad(sco) with affine gaps.
 * This is the mode used by default in the original JAX implementation.
 *
 * @param hij           [L1 * L2 * 3] DP values (Align, Right, Down) from forward
 * @param scores        [L1 * L2] input similarity matrix
 * @param L1, L2        Sequence lengths
 * @param config        SW configuration (temperature, gap_open, gap_extend)
 * @param partition     Scalar partition function from forward pass
 * @param posteriors    [L1 * L2] OUTPUT - posterior probabilities P(i~j)
 */
template <typename Backend>
void smith_waterman_jax_affine_flexible_backward(
    const float* hij, const float* scores, int L1, int L2, const SWConfig& config, float partition,
    float* posteriors, pfalign::memory::GrowableArena* temp_arena = nullptr);

}  // namespace smith_waterman
}  // namespace pfalign
