/**
 * Scalar implementations of Smith-Waterman alignment.
 *
 * Implements all 4 versions:
 * 1. Direct regular (standard LOCAL SW, single gap penalty)
 * 2. Direct affine (standard LOCAL SW, affine gaps)
 * 3. JAX regular (SoftAlign formulation, single gap penalty)
 * 4. JAX affine (SoftAlign formulation, affine gaps)
 */

#include "smith_waterman.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace pfalign {
namespace smith_waterman {

/**
 * Helper: Soft maximum (logsumexp) for numerical stability.
 *
 * IMPORTANT: All Smith-Waterman implementations use the Boltzmann approach:
 *   - Scores and gap penalties are kept as raw values (NOT divided by T)
 *   - soft_max is called with the actual temperature T
 *   - All DP recursions maintain proper Boltzmann statistics
 *   - Partition includes T factor: max + T*log(sum)
 *   - Backward weights include T factor: exp((z - partition)/T)
 *
 * Formula: soft_max(x, T) = max(x) + T * log(sum(exp((x_i - max(x)) / T)))
 *
 * This ensures temperature invariance in posterior probability sums.
 */
static inline float soft_max(const float* values, int n, float temperature) {
    // Find maximum for numerical stability
    float max_val = values[0];
    for (int i = 1; i < n; i++) {
        max_val = std::max(max_val, values[i]);
    }

    // Handle -inf case
    if (std::isinf(max_val) && max_val < 0) {
        return NINF;
    }

    // Compute sum of exp
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_exp += std::exp((values[i] - max_val) / temperature);
    }

    return temperature * (std::log(sum_exp) + max_val / temperature);
}

// ============================================================================
// 1. Direct Regular (Standard LOCAL SW, single gap penalty)
// ============================================================================

template <>
void smith_waterman_direct_regular<ScalarBackend>(const float* scores, int L1, int L2,
                                                  const SWConfig& config, float* alpha,
                                                  float* partition) {
    float T = config.temperature;
    float gap_penalty = config.gap;  // Raw gap, no pre-scaling

    // Initialize boundaries to -inf (LOCAL alignment)
    int size = (L1 + 1) * (L2 + 1);
    for (int i = 0; i < size; i++) {
        alpha[i] = NINF;
    }

    // Fill DP matrix (1-indexed)
    for (int i = 1; i <= L1; i++) {
        for (int j = 1; j <= L2; j++) {
            float score_ij = scores[(i - 1) * L2 + (j - 1)];  // Raw score, no pre-scaling

            // Four transitions
            float align = alpha[(i - 1) * (L2 + 1) + (j - 1)] + score_ij;
            float gap_up = alpha[(i - 1) * (L2 + 1) + j] + gap_penalty;
            float gap_left = alpha[i * (L2 + 1) + (j - 1)] + gap_penalty;
            float sky = score_ij;  // LOCAL: start fresh

            float values[4] = {align, gap_up, gap_left, sky};
            alpha[i * (L2 + 1) + j] = soft_max(values, 4, T);  // Use actual temperature
        }
    }

    // Compute partition function: soft_max over all alpha values
    float max_alpha = NINF;
    for (int i = 0; i < size; i++) {
        max_alpha = std::max(max_alpha, alpha[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        if (alpha[i] > NINF) {
            sum_exp += std::exp((alpha[i] - max_alpha) / T);  // Divide by T in exponent
        }
    }

    *partition = max_alpha + T * std::log(sum_exp);  // Boltzmann partition
}

// ============================================================================
// 2. Direct Affine (Standard LOCAL SW, affine gaps)
// ============================================================================

template <>
void smith_waterman_direct_affine<ScalarBackend>(const float* scores, int L1, int L2,
                                                 const SWConfig& config, float* alpha,
                                                 float* partition) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling

    // alpha has 3 states: [M, I, D] stored as [state, i, j]
    // M: Match/mismatch
    // I: Gap in seq1 (insertion in seq2)
    // D: Gap in seq2 (deletion from seq2)
    int size_per_state = (L1 + 1) * (L2 + 1);

    float* M = alpha;
    float* I = alpha + size_per_state;
    float* D = alpha + 2 * size_per_state;

    // Initialize all to -inf
    for (int i = 0; i < 3 * size_per_state; i++) {
        alpha[i] = NINF;
    }

    // Fill DP matrices
    for (int i = 1; i <= L1; i++) {
        for (int j = 1; j <= L2; j++) {
            float score_ij = scores[(i - 1) * L2 + (j - 1)];  // Raw score, no pre-scaling
            int idx = i * (L2 + 1) + j;
            int idx_diag = (i - 1) * (L2 + 1) + (j - 1);
            int idx_up = (i - 1) * (L2 + 1) + j;
            int idx_left = i * (L2 + 1) + (j - 1);

            // M[i,j]: Come from M, I, or D at (i-1,j-1), or start fresh (Sky)
            float m_from_m = M[idx_diag] + score_ij;
            float m_from_i = I[idx_diag] + score_ij;
            float m_from_d = D[idx_diag] + score_ij;
            float sky = score_ij;  // LOCAL: start fresh

            float m_values[4] = {m_from_m, m_from_i, m_from_d, sky};
            M[idx] = soft_max(m_values, 4, T);  // Use actual temperature

            // I[i,j]: Gap in seq1 (moving down) - come from M or extend I
            float i_from_m = M[idx_up] + gap_open;
            float i_from_i = I[idx_up] + gap_extend;

            float i_values[2] = {i_from_m, i_from_i};
            I[idx] = soft_max(i_values, 2, T);  // Use actual temperature

            // D[i,j]: Gap in seq2 (moving right) - come from M or extend D
            float d_from_m = M[idx_left] + gap_open;
            float d_from_d = D[idx_left] + gap_extend;

            float d_values[2] = {d_from_m, d_from_d};
            D[idx] = soft_max(d_values, 2, T);  // Use actual temperature
        }
    }

    // Partition: soft_max over all M, I, D values
    float max_val = NINF;
    for (int i = 0; i < 3 * size_per_state; i++) {
        max_val = std::max(max_val, alpha[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < 3 * size_per_state; i++) {
        if (alpha[i] > NINF) {
            sum_exp += std::exp((alpha[i] - max_val) / T);  // Divide by T in exponent
        }
    }

    *partition = max_val + T * std::log(sum_exp);  // Boltzmann partition
}

// ============================================================================
// 3. Direct Affine Flexible (Standard LOCAL SW, flexible affine gaps)
// ============================================================================

template <>
void smith_waterman_direct_affine_flexible<ScalarBackend>(const float* scores, int L1, int L2,
                                                          const SWConfig& config, float* alpha,
                                                          float* partition) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling

    // alpha has 3 states: [M, I, D] stored as [state, i, j]
    // M: Match/mismatch
    // I: Gap in seq1 (insertion in seq2, moving down)
    // D: Gap in seq2 (deletion from seq2, moving right)
    int size_per_state = (L1 + 1) * (L2 + 1);

    float* M = alpha;
    float* I = alpha + size_per_state;
    float* D = alpha + 2 * size_per_state;

    // Initialize all to -inf
    for (int i = 0; i < 3 * size_per_state; i++) {
        alpha[i] = NINF;
    }

    // Fill DP matrices
    for (int i = 1; i <= L1; i++) {
        for (int j = 1; j <= L2; j++) {
            float score_ij = scores[(i - 1) * L2 + (j - 1)];  // Raw score, no pre-scaling
            int idx = i * (L2 + 1) + j;
            int idx_diag = (i - 1) * (L2 + 1) + (j - 1);
            int idx_up = (i - 1) * (L2 + 1) + j;
            int idx_left = i * (L2 + 1) + (j - 1);

            // M[i,j]: Come from M, I, or D at (i-1,j-1), or start fresh (Sky)
            float m_from_m = M[idx_diag] + score_ij;
            float m_from_i = I[idx_diag] + score_ij;
            float m_from_d = D[idx_diag] + score_ij;
            float sky = score_ij;  // LOCAL: start fresh

            float m_values[4] = {m_from_m, m_from_i, m_from_d, sky};
            M[idx] = soft_max(m_values, 4, T);  // Use actual temperature

            // I[i,j]: Gap in seq1 (moving down) - come from M or extend I
            float i_from_m = M[idx_up] + gap_open;
            float i_from_i = I[idx_up] + gap_extend;

            float i_values[2] = {i_from_m, i_from_i};
            I[idx] = soft_max(i_values, 2, T);  // Use actual temperature

            // D[i,j]: Gap in seq2 (moving right) - FLEXIBLE: allow I->D transition
            float d_from_m = M[idx_left] + gap_open;
            float d_from_i = I[idx_left] + gap_open;  // NEW: I->D transition!
            float d_from_d = D[idx_left] + gap_extend;

            float d_values[3] = {d_from_m, d_from_i, d_from_d};
            D[idx] = soft_max(d_values, 3, T);  // Use actual temperature
        }
    }

    // Partition: soft_max over all M, I, D values
    float max_val = NINF;
    for (int i = 0; i < 3 * size_per_state; i++) {
        max_val = std::max(max_val, alpha[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < 3 * size_per_state; i++) {
        if (alpha[i] > NINF) {
            sum_exp += std::exp((alpha[i] - max_val) / T);  // Divide by T in exponent
        }
    }

    *partition = max_val + T * std::log(sum_exp);  // Boltzmann partition
}

// ============================================================================
// 4. JAX Regular (SoftAlign formulation, single gap penalty)
// ============================================================================

template <>
void smith_waterman_jax_regular<ScalarBackend>(const float* scores, int L1, int L2,
                                               const SWConfig& config, float* hij,
                                               float* partition) {
    float T = config.temperature;
    float gap_penalty = config.gap;  // Raw gap, no pre-scaling

    // Initialize hij to -inf
    for (int i = 0; i < L1 * L2; i++) {
        hij[i] = NINF;
    }

    // Fill hij matrix (0-indexed, conceptually on scores[:-1, :-1])
    // Process grid of size (L1-1) * (L2-1)
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_ij = scores[i * L2 + j];  // Raw score, no pre-scaling

            // Four transitions (matching CUDA JAX implementation)
            float align = NINF;
            float turn_0 = NINF;
            float turn_1 = NINF;
            float sky = score_ij;

            // Align: from (i-1, j-1)
            if (i > 0 && j > 0) {
                align = hij[(i - 1) * L2 + (j - 1)] + score_ij;
            }

            // Turn_0 (gap down): from (i-1, j)
            if (i > 0) {
                turn_0 = hij[(i - 1) * L2 + j] + gap_penalty;
            }

            // Turn_1 (gap right): from (i, j-1)
            if (j > 0) {
                turn_1 = hij[i * L2 + (j - 1)] + gap_penalty;
            }

            float values[4] = {align, turn_0, turn_1, sky};
            hij[i * L2 + j] = soft_max(values, 4, T);  // Use actual temperature
        }
    }

    // Final step: JAX formulation
    // S = soft_max(hij[i,j] + scores[i+1, j+1]) for i<L1-1, j<L2-1
    // This is the KEY DIFFERENCE from standard LOCAL SW!

    float max_val = NINF;
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float val = hij[i * L2 + j] + scores[(i + 1) * L2 + (j + 1)];  // Raw score
            max_val = std::max(max_val, val);
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float val = hij[i * L2 + j] + scores[(i + 1) * L2 + (j + 1)];  // Raw score
            if (val > NINF) {
                sum_exp += std::exp((val - max_val) / T);  // Divide by T in exponent
            }
        }
    }

    *partition = max_val + T * std::log(sum_exp);  // Boltzmann partition
}

// ============================================================================
// 4. JAX Affine (SoftAlign formulation, affine gaps)
// ============================================================================

template <>
void smith_waterman_jax_affine<ScalarBackend>(const float* scores, int L1, int L2,
                                              const SWConfig& config, float* hij,
                                              float* partition) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling

    // hij has 3 states: [Align, Right, Down] stored as [i, j, state]
    // Align: Match/mismatch
    // Right: Gap in seq2 (moving right - insertion in seq1)
    // Down: Gap in seq1 (moving down - insertion in seq2)

    // Initialize to -inf
    for (int i = 0; i < L1 * L2 * 3; i++) {
        hij[i] = NINF;
    }

    // Fill DP matrices (0-indexed, grid size (L1-1) * (L2-1))
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_ij = scores[i * L2 + j];  // Raw score, no pre-scaling

            // Align state: Come from any state at (i-1,j-1) or start fresh
            float align_from_align = NINF;
            float align_from_right = NINF;
            float align_from_down = NINF;
            float sky = score_ij;

            if (i > 0 && j > 0) {
                int prev_idx = (i - 1) * L2 * 3 + (j - 1) * 3;
                align_from_align = hij[prev_idx + 0] + score_ij;
                align_from_right = hij[prev_idx + 1] + score_ij;
                align_from_down = hij[prev_idx + 2] + score_ij;
            }

            float align_values[4] = {align_from_align, align_from_right, align_from_down, sky};
            hij[i * L2 * 3 + j * 3 + 0] = soft_max(align_values, 4, T);  // Use actual temperature

            // Right state: Gap in seq2 (moving right from same row)
            // Come from Align or extend Right at (i, j-1)
            float right_from_align = NINF;
            float right_from_right = NINF;

            if (j > 0) {
                int prev_idx = i * L2 * 3 + (j - 1) * 3;
                right_from_align = hij[prev_idx + 0] + gap_open;
                right_from_right = hij[prev_idx + 1] + gap_extend;
            }

            float right_values[2] = {right_from_align, right_from_right};
            hij[i * L2 * 3 + j * 3 + 1] = soft_max(right_values, 2, T);  // Use actual temperature

            // Down state: Gap in seq1 (moving down from same column)
            // Come from Align or extend Down at (i-1, j)
            float down_from_align = NINF;
            float down_from_down = NINF;

            if (i > 0) {
                int prev_idx = (i - 1) * L2 * 3 + j * 3;
                down_from_align = hij[prev_idx + 0] + gap_open;
                down_from_down = hij[prev_idx + 2] + gap_extend;
            }

            float down_values[2] = {down_from_align, down_from_down};
            hij[i * L2 * 3 + j * 3 + 2] = soft_max(down_values, 2, T);  // Use actual temperature
        }
    }

    // Final step: JAX affine formulation
    // S = soft_max(hij[i,j,:] + scores[i+1,j+1]) for i<L1-1, j<L2-1
    // Sum over all 3 states

    float max_val = NINF;
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score
            for (int state = 0; state < 3; state++) {
                float val = hij[i * L2 * 3 + j * 3 + state] + score_shifted;
                max_val = std::max(max_val, val);
            }
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score
            for (int state = 0; state < 3; state++) {
                float val = hij[i * L2 * 3 + j * 3 + state] + score_shifted;
                if (val > NINF) {
                    sum_exp += std::exp((val - max_val) / T);  // Divide by T in exponent
                }
            }
        }
    }

    *partition = max_val + T * std::log(sum_exp);  // Boltzmann partition
}

/**
 * JAX Affine - FLEXIBLE model (with cross-state transitions).
 *
 * This EXACTLY matches the JAX reference implementation with
 * penalize_turns=True and restrict_turns=True.
 *
 * Key difference from standard affine:
 * - Right state can come from: Align (open), Right (extend)
 * - Down state can come from: Align (open), Right (open), Down (extend)
 *
 * The Right->Down transition is what makes this "flexible".
 */
template <>
void smith_waterman_jax_affine_flexible<ScalarBackend>(const float* scores, int L1, int L2,
                                                       const SWConfig& config, float* hij,
                                                       float* partition) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling

    // Initialize to -inf
    for (int i = 0; i < L1 * L2 * 3; i++) {
        hij[i] = NINF;
    }

    // Fill DP matrices (0-indexed, grid size (L1-1) * (L2-1))
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_ij = scores[i * L2 + j];  // Raw score, no pre-scaling

            // Align state: Come from any state at (i-1,j-1) or start fresh
            float align_from_align = NINF;
            float align_from_right = NINF;
            float align_from_down = NINF;
            float sky = score_ij;

            if (i > 0 && j > 0) {
                int prev_idx = (i - 1) * L2 * 3 + (j - 1) * 3;
                align_from_align = hij[prev_idx + 0] + score_ij;
                align_from_right = hij[prev_idx + 1] + score_ij;
                align_from_down = hij[prev_idx + 2] + score_ij;
            }

            float align_values[4] = {align_from_align, align_from_right, align_from_down, sky};
            hij[i * L2 * 3 + j * 3 + 0] = soft_max(align_values, 4, T);  // Use actual temperature

            // Right state: Gap in seq2 (moving right from same row)
            // FLEXIBLE: Can come from Align or Right (standard)
            // (Down->Right would be allowed but is restricted out)
            float right_from_align = NINF;
            float right_from_right = NINF;

            if (j > 0) {
                int prev_idx = i * L2 * 3 + (j - 1) * 3;
                right_from_align = hij[prev_idx + 0] + gap_open;
                right_from_right = hij[prev_idx + 1] + gap_extend;
                // Note: Down->Right with gap_open would go here, but restrict_turns=True removes
                // it
            }

            float right_values[2] = {right_from_align, right_from_right};
            hij[i * L2 * 3 + j * 3 + 1] = soft_max(right_values, 2, T);  // Use actual temperature

            // Down state: Gap in seq1 (moving down from same column)
            // FLEXIBLE: Can come from Align, Right (with open), or Down (extend)
            // This Right->Down transition is the key difference!
            float down_from_align = NINF;
            float down_from_right = NINF;  // NEW: Right->Down transition!
            float down_from_down = NINF;

            if (i > 0) {
                int prev_idx = (i - 1) * L2 * 3 + j * 3;
                down_from_align = hij[prev_idx + 0] + gap_open;
                down_from_right =
                    hij[prev_idx + 1] + gap_open;  // JAX flexible: Right->Down with gap_open
                down_from_down = hij[prev_idx + 2] + gap_extend;
            }

            float down_values[3] = {down_from_align, down_from_right, down_from_down};
            hij[i * L2 * 3 + j * 3 + 2] = soft_max(down_values, 3, T);  // Use actual temperature
        }
    }

    // Final step: JAX affine formulation
    // S = soft_max(hij[i,j,:] + scores[i+1,j+1]) for i<L1-1, j<L2-1

    float max_val = NINF;
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score
            for (int state = 0; state < 3; state++) {
                float val = hij[i * L2 * 3 + j * 3 + state] + score_shifted;
                max_val = std::max(max_val, val);
            }
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score
            for (int state = 0; state < 3; state++) {
                float val = hij[i * L2 * 3 + j * 3 + state] + score_shifted;
                sum_exp += std::exp((val - max_val) / T);  // Divide by T in exponent
            }
        }
    }

    *partition = max_val + T * std::log(sum_exp);  // Boltzmann partition
}

// ============================================================================
// BACKWARD PASSES (Compute Posteriors / Alignment Matrices)
// ============================================================================

/**
 * JAX Regular Backward Pass
 *
 * Computes ∂partition/∂scores via backpropagation through the forward DP.
 * This gives the posterior probabilities P(position i aligns to position j).
 *
 * Algorithm:
 * 1. Backprop through final soft_max: S = soft_max(hij + scores[i+1,j+1])
 * 2. Backprop through forward DP: hij[i,j] = soft_max([align, turn_0, turn_1, sky])
 *
 * The gradient ∂S/∂scores is the alignment matrix that JAX returns.
 */
template <>
void smith_waterman_jax_regular_backward<ScalarBackend>(
    const float* hij, const float* scores, int L1, int L2, const SWConfig& config, float partition,
    float* posteriors, pfalign::memory::GrowableArena* temp_arena) {
    float T = config.temperature;
    float gap_penalty = config.gap;  // Raw gap, no pre-scaling
    float upstream = 1.0f;  // ∂Loss/∂S (assuming loss = S)

    // Initialize posteriors to 0
    std::memset(posteriors, 0, L1 * L2 * sizeof(float));

    // Allocate workspace for backward DP values (beta)
    float* beta;
    bool allocated_beta = false;
    if (temp_arena) {
        beta = temp_arena->allocate<float>(L1 * L2);
    } else {
        beta = new float[L1 * L2];
        allocated_beta = true;
    }
    for (int i = 0; i < L1 * L2; i++) {
        beta[i] = NINF;
    }

    // ========================================================================
    // STEP 1: Backprop through final soft_max
    // ========================================================================
    //
    // S = soft_max(hij[i,j] + scores[i+1,j+1]) for i∈[0,L1-1), j∈[0,L2-1)
    //
    // Let z[i,j] = hij[i,j] + scores[i+1,j+1]
    // Then: ∂S/∂z[i,j] = exp((z[i,j] - S) / T) = softmax_weight[i,j]
    //
    // Gradients flow to:
    //   ∂S/∂hij[i,j] = softmax_weight[i,j]
    //   ∂S/∂scores[i+1,j+1] = softmax_weight[i,j]  (SHIFTED!)

    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            // Compute z = hij[i,j] + scores[i+1,j+1]
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score, no pre-scaling
            float z_ij = hij[i * L2 + j] + score_shifted;

            // Compute softmax weight (Boltzmann)
            float weight = std::exp((z_ij - partition) / T);  // Divide by T

            // Gradient flows to hij[i,j]
            beta[i * L2 + j] = upstream * weight;

            // Gradient flows to scores[i+1,j+1] (SHIFTED!)
            posteriors[(i + 1) * L2 + (j + 1)] += upstream * weight;
        }
    }

    // ========================================================================
    // STEP 2: Backward DP through forward computation
    // ========================================================================
    //
    // Forward recurrence:
    //   hij[i,j] = soft_max([align, turn_0, turn_1, sky])
    //
    // Where:
    //   align = hij[i-1,j-1] + scores[i,j]  (if i>0, j>0)
    //   turn_0 = hij[i-1,j] + gap            (if i>0)
    //   turn_1 = hij[i,j-1] + gap            (if j>0)
    //   sky = scores[i,j]
    //
    // Given ∂Loss/∂hij[i,j] (stored in beta), compute:
    //   ∂Loss/∂hij[i-1,j-1], ∂Loss/∂hij[i-1,j], ∂Loss/∂hij[i,j-1]
    //   ∂Loss/∂scores[i,j]

    // Process in reverse order (i from L1-2 to 0, j from L2-2 to 0)
    for (int i = L1 - 2; i >= 0; i--) {
        for (int j = L2 - 2; j >= 0; j--) {
            float grad_hij_ij = beta[i * L2 + j];
            if (std::isinf(grad_hij_ij) || grad_hij_ij == 0.0f)
                continue;

            // Reconstruct forward computation
            float score_ij = scores[i * L2 + j];  // Raw score, no pre-scaling

            float align = NINF;
            float turn_0 = NINF;
            float turn_1 = NINF;
            float sky = score_ij;

            if (i > 0 && j > 0) {
                align = hij[(i - 1) * L2 + (j - 1)] + score_ij;
            }
            if (i > 0) {
                turn_0 = hij[(i - 1) * L2 + j] + gap_penalty;  // Raw gap
            }
            if (j > 0) {
                turn_1 = hij[i * L2 + (j - 1)] + gap_penalty;  // Raw gap
            }

            // Compute softmax weights for each transition
            float options[4] = {align, turn_0, turn_1, sky};
            float max_val = NINF;
            for (int t = 0; t < 4; t++) {
                max_val = std::max(max_val, options[t]);
            }

            float sum_exp = 0.0f;
            float weights[4];
            for (int t = 0; t < 4; t++) {
                weights[t] = std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                sum_exp += weights[t];
            }

            for (int t = 0; t < 4; t++) {
                weights[t] /= sum_exp;
            }

            // Backprop through soft_max
            // ∂Loss/∂align
            if (i > 0 && j > 0) {
                float grad_align = grad_hij_ij * weights[0];
                beta[(i - 1) * L2 + (j - 1)] += grad_align;  // -> hij[i-1,j-1]
                posteriors[i * L2 + j] += grad_align;        // -> scores[i,j]
            }

            // ∂Loss/∂turn_0
            if (i > 0) {
                float grad_turn0 = grad_hij_ij * weights[1];
                beta[(i - 1) * L2 + j] += grad_turn0;  // -> hij[i-1,j]
                // grad_gap += grad_turn0 (not computed here, only posteriors)
            }

            // ∂Loss/∂turn_1
            if (j > 0) {
                float grad_turn1 = grad_hij_ij * weights[2];
                beta[i * L2 + (j - 1)] += grad_turn1;  // -> hij[i,j-1]
                // grad_gap += grad_turn1 (not computed here)
            }

            // ∂Loss/∂sky
            float grad_sky = grad_hij_ij * weights[3];
            posteriors[i * L2 + j] += grad_sky;  // -> scores[i,j]
        }
    }

    // Cleanup only if manually allocated
    if (allocated_beta) {
        delete[] beta;
    }
}

/**
 * Direct Regular Backward Pass
 *
 * Computes ∂partition/∂scores via backpropagation through the Direct LOCAL SW formulation.
 * This produces posterior probabilities P(position i aligns to position j).
 *
 * Algorithm:
 * 1. Compute ∂S/∂alpha for all positions (including empty alignment)
 * 2. Backprop through DP recursion: alpha[i,j] = soft_max([align, gap_up, gap_left, sky])
 *
 * Key differences from JAX:
 * - Uses 1-indexed alpha [L1+1 * L2+1] instead of 0-indexed hij
 * - Has boundary condition at (0,0) for empty alignment
 * - No shifted final step
 */
template <>
void smith_waterman_direct_regular_backward<ScalarBackend>(
    const float* alpha, const float* scores, int L1, int L2, const SWConfig& config,
    float partition, float* posteriors, pfalign::memory::GrowableArena* temp_arena) {
    float T = config.temperature;
    float gap_penalty = config.gap;  // Raw gap, no pre-scaling
    float upstream = 1.0f;  // ∂Loss/∂S (assuming loss = S)

    // Initialize posteriors to 0
    std::memset(posteriors, 0, L1 * L2 * sizeof(float));

    // Allocate workspace for backward DP values (grad_alpha)
    int alpha_size = (L1 + 1) * (L2 + 1);
    float* grad_alpha;
    bool allocated_grad_alpha = false;
    if (temp_arena) {
        grad_alpha = temp_arena->allocate<float>(alpha_size);
    } else {
        grad_alpha = new float[alpha_size];
        allocated_grad_alpha = true;
    }
    std::memset(grad_alpha, 0, alpha_size * sizeof(float));

    // ========================================================================
    // STEP 1: Compute ∂S/∂alpha for all positions
    // ========================================================================
    //
    // S = soft_max(alpha[i,j]) for all i,j (including empty alignment at (0,0))
    // ∂S/∂alpha[i,j] = exp((alpha[i,j] - S) / T)

    for (int i = 0; i <= L1; i++) {
        for (int j = 0; j <= L2; j++) {
            int idx = i * (L2 + 1) + j;

            // Empty alignment at (0,0) contributes 0 to logsumexp
            if (i == 0 && j == 0) {
                float exponent = -partition;
                grad_alpha[idx] = upstream * std::exp(exponent / T);  // Divide by T (Approach B)
                continue;
            }

            // Skip positions outside valid range
            if (i < 1 || i > L1 || j < 1 || j > L2) {
                continue;
            }

            // Gradient from being an ending position
            float alpha_val = alpha[idx];
            float exponent = alpha_val - partition;
            grad_alpha[idx] = upstream * std::exp(exponent / T);  // Divide by T (Approach B)
        }
    }

    // ========================================================================
    // STEP 2: Backprop through DP recursion
    // ========================================================================
    //
    // Forward recurrence:
    //   alpha[i,j] = soft_max([align, gap_up, gap_left, sky])
    //
    // Where:
    //   align = alpha[i-1,j-1] + scores[i-1,j-1]/T  (if i>1, j>1)
    //   gap_up = alpha[i-1,j] + gap/T                (if i>1)
    //   gap_left = alpha[i,j-1] + gap/T              (if j>1)
    //   sky = scores[i-1,j-1]/T                  (LOCAL: start fresh)
    //
    // Process in reverse order

    for (int i = L1; i >= 1; i--) {
        for (int j = L2; j >= 1; j--) {
            int alpha_idx = i * (L2 + 1) + j;
            float grad_alpha_ij = grad_alpha[alpha_idx];

            if (grad_alpha_ij == 0.0f)
                continue;

            // Reconstruct forward computation
            int score_idx = (i - 1) * L2 + (j - 1);  // scores is 0-indexed
            float score_ij = scores[score_idx];  // Raw score (Approach B)

            // Four transitions
            float val_align = NINF;
            float val_up = NINF;
            float val_left = NINF;
            float val_sky = score_ij;

            if (i > 1 && j > 1) {
                float alpha_prev = alpha[(i - 1) * (L2 + 1) + (j - 1)];
                val_align = alpha_prev + score_ij;
            }
            if (i > 1) {
                float alpha_up = alpha[(i - 1) * (L2 + 1) + j];
                val_up = alpha_up + gap_penalty;  // Raw gap
            }
            if (j > 1) {
                float alpha_left = alpha[i * (L2 + 1) + (j - 1)];
                val_left = alpha_left + gap_penalty;  // Raw gap
            }

            // Compute softmax weights
            float options[4] = {val_align, val_up, val_left, val_sky};
            float max_val = NINF;
            for (int t = 0; t < 4; t++) {
                max_val = std::max(max_val, options[t]);
            }

            if (std::isinf(max_val) && max_val < 0)
                continue;

            float sum_exp = 0.0f;
            float weights[4];
            for (int t = 0; t < 4; t++) {
                if (!std::isinf(options[t])) {
                    weights[t] = std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                    sum_exp += weights[t];
                } else {
                    weights[t] = 0.0f;
                }
            }

            for (int t = 0; t < 4; t++) {
                weights[t] /= sum_exp;
            }

            // Backprop through soft_max
            // ∂Loss/∂align
            if (!std::isinf(val_align)) {
                float grad_align = grad_alpha_ij * weights[0];
                grad_alpha[(i - 1) * (L2 + 1) + (j - 1)] += grad_align;  // -> alpha[i-1,j-1]
                posteriors[score_idx] += grad_align;                     // -> scores[i-1,j-1]
            }

            // ∂Loss/∂gap_up
            if (!std::isinf(val_up)) {
                float grad_up = grad_alpha_ij * weights[1];
                grad_alpha[(i - 1) * (L2 + 1) + j] += grad_up;  // -> alpha[i-1,j]
                // grad_gap += grad_up (not computed here)
            }

            // ∂Loss/∂gap_left
            if (!std::isinf(val_left)) {
                float grad_left = grad_alpha_ij * weights[2];
                grad_alpha[i * (L2 + 1) + (j - 1)] += grad_left;  // -> alpha[i,j-1]
                // grad_gap += grad_left (not computed here)
            }

            // ∂Loss/∂sky
            if (!std::isinf(val_sky)) {
                float grad_sky = grad_alpha_ij * weights[3];
                posteriors[score_idx] += grad_sky;  // -> scores[i-1,j-1]
            }
        }
    }

    // Cleanup only if manually allocated
    if (allocated_grad_alpha) {
        delete[] grad_alpha;
    }
}

/**
 * JAX Affine Flexible Backward Pass
 *
 * Computes ∂partition/∂scores via backpropagation through the JAX affine flexible formulation.
 * This handles 3 states: Align, Right, Down with flexible transitions (Right->Down allowed).
 *
 * Algorithm:
 * 1. Backprop through final soft_max: S = soft_max(hij[i,j,:] + scores[i+1,j+1])
 * 2. Backprop through DP for each state:
 *    - Align: [align_from_align, align_from_right, align_from_down, sky]
 *    - Right: [right_from_align, right_from_right]
 *    - Down: [down_from_align, down_from_right, down_from_down] (FLEXIBLE!)
 */
template <>
void smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
    const float* hij, const float* scores, int L1, int L2, const SWConfig& config, float partition,
    float* posteriors, pfalign::memory::GrowableArena* temp_arena) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling
    float upstream = 1.0f;

    // Initialize posteriors to 0
    std::memset(posteriors, 0, L1 * L2 * sizeof(float));

    // Allocate workspace for backward DP values (beta) - 3 states
    float* beta;
    bool allocated_beta = false;
    if (temp_arena) {
        beta = temp_arena->allocate<float>(L1 * L2 * 3);
    } else {
        beta = new float[L1 * L2 * 3];
        allocated_beta = true;
    }
    for (int i = 0; i < L1 * L2 * 3; i++) {
        beta[i] = NINF;
    }

    // ========================================================================
    // STEP 1: Backprop through final soft_max
    // ========================================================================
    //
    // S = soft_max(hij[i,j,state] + scores[i+1,j+1]) for i∈[0,L1-1), j∈[0,L2-1), state∈{0,1,2}

    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score, no pre-scaling

            for (int state = 0; state < 3; state++) {
                int idx = i * L2 * 3 + j * 3 + state;
                float z_ij = hij[idx] + score_shifted;

                // Compute softmax weight (Boltzmann)
                float weight = std::exp((z_ij - partition) / T);  // Divide by T

                // Gradient flows to hij[i,j,state]
                beta[idx] = upstream * weight;

                // Gradient flows to scores[i+1,j+1] (accumulated from all states)
                posteriors[(i + 1) * L2 + (j + 1)] += upstream * weight;
            }
        }
    }

    // ========================================================================
    // STEP 2: Backward DP through forward computation
    // ========================================================================
    //
    // Process in reverse order
    for (int i = L1 - 2; i >= 0; i--) {
        for (int j = L2 - 2; j >= 0; j--) {
            float score_ij = scores[i * L2 + j];  // Raw score, no pre-scaling

            // ================================================================
            // Align state: hij[i,j,0]
            // ================================================================
            int align_idx = i * L2 * 3 + j * 3 + 0;
            float grad_align = beta[align_idx];

            if (!std::isinf(grad_align) && grad_align != 0.0f) {
                // Reconstruct forward computation
                float align_from_align = NINF;
                float align_from_right = NINF;
                float align_from_down = NINF;
                float sky = score_ij;

                if (i > 0 && j > 0) {
                    int prev_idx = (i - 1) * L2 * 3 + (j - 1) * 3;
                    align_from_align = hij[prev_idx + 0] + score_ij;
                    align_from_right = hij[prev_idx + 1] + score_ij;
                    align_from_down = hij[prev_idx + 2] + score_ij;
                }

                // Compute softmax weights
                float options[4] = {align_from_align, align_from_right, align_from_down, sky};
                float max_val = NINF;
                for (int t = 0; t < 4; t++) {
                    max_val = std::max(max_val, options[t]);
                }

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[4];
                    for (int t = 0; t < 4; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 4; t++) {
                        weights[t] /= sum_exp;
                    }

                    // Backprop
                    if (i > 0 && j > 0) {
                        int prev_idx = (i - 1) * L2 * 3 + (j - 1) * 3;
                        beta[prev_idx + 0] += grad_align * weights[0];  // -> align[i-1,j-1]
                        beta[prev_idx + 1] += grad_align * weights[1];  // -> right[i-1,j-1]
                        beta[prev_idx + 2] += grad_align * weights[2];  // -> down[i-1,j-1]
                        posteriors[i * L2 + j] +=
                            grad_align * (weights[0] + weights[1] + weights[2]);
                    }
                    posteriors[i * L2 + j] += grad_align * weights[3];  // sky
                }
            }

            // ================================================================
            // Right state: hij[i,j,1]
            // ================================================================
            int right_idx = i * L2 * 3 + j * 3 + 1;
            float grad_right = beta[right_idx];

            if (!std::isinf(grad_right) && grad_right != 0.0f && j > 0) {
                // Reconstruct forward computation
                int prev_idx = i * L2 * 3 + (j - 1) * 3;
                float right_from_align = hij[prev_idx + 0] + gap_open;     // Raw gap
                float right_from_right = hij[prev_idx + 1] + gap_extend;   // Raw gap

                // Compute softmax weights
                float options[2] = {right_from_align, right_from_right};
                float max_val = std::max(options[0], options[1]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[2];
                    for (int t = 0; t < 2; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 2; t++) {
                        weights[t] /= sum_exp;
                    }

                    // Backprop
                    beta[prev_idx + 0] += grad_right * weights[0];  // -> align[i,j-1]
                    beta[prev_idx + 1] += grad_right * weights[1];  // -> right[i,j-1]
                }
            }

            // ================================================================
            // Down state: hij[i,j,2] - FLEXIBLE (3 transitions)
            // ================================================================
            int down_idx = i * L2 * 3 + j * 3 + 2;
            float grad_down = beta[down_idx];

            if (!std::isinf(grad_down) && grad_down != 0.0f && i > 0) {
                // Reconstruct forward computation
                int prev_idx = (i - 1) * L2 * 3 + j * 3;
                float down_from_align = hij[prev_idx + 0] + gap_open;    // Raw gap
                float down_from_right = hij[prev_idx + 1] + gap_open;    // FLEXIBLE! Raw gap
                float down_from_down = hij[prev_idx + 2] + gap_extend;   // Raw gap

                // Compute softmax weights
                float options[3] = {down_from_align, down_from_right, down_from_down};
                float max_val = NINF;
                for (int t = 0; t < 3; t++) {
                    max_val = std::max(max_val, options[t]);
                }

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[3];
                    for (int t = 0; t < 3; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 3; t++) {
                        weights[t] /= sum_exp;
                    }

                    // Backprop
                    beta[prev_idx + 0] += grad_down * weights[0];  // -> align[i-1,j]
                    beta[prev_idx + 1] += grad_down * weights[1];  // -> right[i-1,j] (FLEXIBLE)
                    beta[prev_idx + 2] += grad_down * weights[2];  // -> down[i-1,j]
                }
            }
        }
    }

    // Cleanup only if manually allocated
    if (allocated_beta) {
        delete[] beta;
    }
}

/**
 * Direct Affine Flexible Backward Pass
 *
 * Computes ∂partition/∂scores via backpropagation through the Direct affine flexible formulation.
 * Uses 1-indexed alpha [L1+1 * L2+1 * 3] with 3 states: M (Match), I (Insertion), D (Deletion).
 * Flexible model allows I->D transitions.
 *
 * Algorithm:
 * 1. Compute ∂S/∂alpha for all positions and states (including empty alignment)
 * 2. Backprop through DP for each state:
 *    - M: [m_from_m, m_from_i, m_from_d, sky]
 *    - I: [i_from_m, i_from_i]
 *    - D: [d_from_m, d_from_i, d_from_d] (FLEXIBLE!)
 */
template <>
void smith_waterman_direct_affine_flexible_backward<ScalarBackend>(
    const float* alpha, const float* scores, int L1, int L2, const SWConfig& config,
    float partition, float* posteriors, pfalign::memory::GrowableArena* temp_arena) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling
    float upstream = 1.0f;

    // Initialize posteriors to 0
    std::memset(posteriors, 0, L1 * L2 * sizeof(float));

    // Allocate workspace for backward DP values (grad_alpha) - 3 states
    int size_per_state = (L1 + 1) * (L2 + 1);
    float* grad_alpha;
    bool allocated_grad_alpha = false;
    if (temp_arena) {
        grad_alpha = temp_arena->allocate<float>(size_per_state * 3);
    } else {
        grad_alpha = new float[size_per_state * 3];
        allocated_grad_alpha = true;
    }
    std::memset(grad_alpha, 0, size_per_state * 3 * sizeof(float));

    const float* M = alpha;
    const float* I = alpha + size_per_state;
    const float* D = alpha + 2 * size_per_state;

    float* grad_M = grad_alpha;
    float* grad_I = grad_alpha + size_per_state;
    float* grad_D = grad_alpha + 2 * size_per_state;

    // ========================================================================
    // STEP 1: Compute ∂S/∂alpha for all positions
    // ========================================================================

    for (int i = 0; i <= L1; i++) {
        for (int j = 0; j <= L2; j++) {
            int idx = i * (L2 + 1) + j;

            // Empty alignment at (0,0)
            if (i == 0 && j == 0) {
                float exponent = -partition;
                grad_M[idx] = upstream * std::exp(exponent / T);  // Divide by T (Approach B)
                continue;
            }

            if (i < 1 || i > L1 || j < 1 || j > L2)
                continue;

            // Gradient from being an ending position (all states)
            for (int state = 0; state < 3; state++) {
                const float* state_alpha = (state == 0) ? M : (state == 1) ? I : D;
                float* state_grad = (state == 0) ? grad_M : (state == 1) ? grad_I : grad_D;

                float alpha_val = state_alpha[idx];
                float exponent = alpha_val - partition;
                state_grad[idx] = upstream * std::exp(exponent / T);  // Divide by T (Approach B)
            }
        }
    }

    // ========================================================================
    // STEP 2: Backprop through DP recursion
    // ========================================================================

    for (int i = L1; i >= 1; i--) {
        for (int j = L2; j >= 1; j--) {
            int idx = i * (L2 + 1) + j;
            int idx_diag = (i - 1) * (L2 + 1) + (j - 1);
            int idx_up = (i - 1) * (L2 + 1) + j;
            int idx_left = i * (L2 + 1) + (j - 1);

            int score_idx = (i - 1) * L2 + (j - 1);
            float score_ij = scores[score_idx];  // Raw score (Approach B)

            // ================================================================
            // M state: Match/mismatch
            // ================================================================
            float grad_m = grad_M[idx];
            if (grad_m != 0.0f) {
                float m_from_m = (i > 1 && j > 1) ? M[idx_diag] + score_ij : NINF;
                float m_from_i = (i > 1 && j > 1) ? I[idx_diag] + score_ij : NINF;
                float m_from_d = (i > 1 && j > 1) ? D[idx_diag] + score_ij : NINF;
                float sky = score_ij;

                float options[4] = {m_from_m, m_from_i, m_from_d, sky};
                float max_val = NINF;
                for (int t = 0; t < 4; t++)
                    max_val = std::max(max_val, options[t]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[4];
                    for (int t = 0; t < 4; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 4; t++)
                        weights[t] /= sum_exp;

                    // Backprop
                    if (i > 1 && j > 1) {
                        grad_M[idx_diag] += grad_m * weights[0];
                        grad_I[idx_diag] += grad_m * weights[1];
                        grad_D[idx_diag] += grad_m * weights[2];
                        posteriors[score_idx] += grad_m * (weights[0] + weights[1] + weights[2]);
                    }
                    posteriors[score_idx] += grad_m * weights[3];  // sky
                }
            }

            // ================================================================
            // I state: Insertion (gap in seq1, moving down)
            // ================================================================
            float grad_i = grad_I[idx];
            if (grad_i != 0.0f && i > 1) {
                float i_from_m = M[idx_up] + gap_open;     // Raw gap
                float i_from_i = I[idx_up] + gap_extend;   // Raw gap

                float options[2] = {i_from_m, i_from_i};
                float max_val = std::max(options[0], options[1]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[2];
                    for (int t = 0; t < 2; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 2; t++)
                        weights[t] /= sum_exp;

                    grad_M[idx_up] += grad_i * weights[0];
                    grad_I[idx_up] += grad_i * weights[1];
                }
            }

            // ================================================================
            // D state: Deletion (gap in seq2, moving right) - FLEXIBLE!
            // ================================================================
            float grad_d = grad_D[idx];
            if (grad_d != 0.0f && j > 1) {
                float d_from_m = M[idx_left] + gap_open;    // Raw gap
                float d_from_i = I[idx_left] + gap_open;    // FLEXIBLE: I->D, Raw gap
                float d_from_d = D[idx_left] + gap_extend;  // Raw gap

                float options[3] = {d_from_m, d_from_i, d_from_d};
                float max_val = NINF;
                for (int t = 0; t < 3; t++)
                    max_val = std::max(max_val, options[t]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[3];
                    for (int t = 0; t < 3; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 3; t++)
                        weights[t] /= sum_exp;

                    grad_M[idx_left] += grad_d * weights[0];
                    grad_I[idx_left] += grad_d * weights[1];  // FLEXIBLE
                    grad_D[idx_left] += grad_d * weights[2];
                }
            }
        }
    }

    // Cleanup only if manually allocated
    if (allocated_grad_alpha) {
        delete[] grad_alpha;
    }
}

/**
 * JAX Affine Backward Pass (STANDARD model)
 *
 * Similar to JAX affine flexible but with STANDARD affine gaps:
 * - Right state: 2 transitions (align, right) - NO Down->Right
 * - Down state: 2 transitions (align, down) - NO Right->Down
 */
template <>
void smith_waterman_jax_affine_backward<ScalarBackend>(const float* hij, const float* scores,
                                                       int L1, int L2, const SWConfig& config,
                                                       float partition, float* posteriors,
                                                       pfalign::memory::GrowableArena* temp_arena) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling
    float upstream = 1.0f;

    // Initialize posteriors to 0
    std::memset(posteriors, 0, L1 * L2 * sizeof(float));

    // Allocate workspace for backward DP values (beta) - 3 states
    float* beta;
    bool allocated_beta = false;
    if (temp_arena) {
        beta = temp_arena->allocate<float>(L1 * L2 * 3);
    } else {
        beta = new float[L1 * L2 * 3];
        allocated_beta = true;
    }
    for (int i = 0; i < L1 * L2 * 3; i++) {
        beta[i] = NINF;
    }

    // ========================================================================
    // STEP 1: Backprop through final soft_max
    // ========================================================================

    for (int i = 0; i < L1 - 1; i++) {
        for (int j = 0; j < L2 - 1; j++) {
            float score_shifted = scores[(i + 1) * L2 + (j + 1)];  // Raw score, no pre-scaling

            for (int state = 0; state < 3; state++) {
                int idx = i * L2 * 3 + j * 3 + state;
                float z_ij = hij[idx] + score_shifted;
                float weight = std::exp((z_ij - partition) / T);  // Divide by T
                beta[idx] = upstream * weight;
                posteriors[(i + 1) * L2 + (j + 1)] += upstream * weight;
            }
        }
    }

    // ========================================================================
    // STEP 2: Backward DP through forward computation
    // ========================================================================

    for (int i = L1 - 2; i >= 0; i--) {
        for (int j = L2 - 2; j >= 0; j--) {
            float score_ij = scores[i * L2 + j];  // Raw score, no pre-scaling

            // Align state
            int align_idx = i * L2 * 3 + j * 3 + 0;
            float grad_align = beta[align_idx];

            if (!std::isinf(grad_align) && grad_align != 0.0f) {
                float align_from_align = NINF;
                float align_from_right = NINF;
                float align_from_down = NINF;
                float sky = score_ij;

                if (i > 0 && j > 0) {
                    int prev_idx = (i - 1) * L2 * 3 + (j - 1) * 3;
                    align_from_align = hij[prev_idx + 0] + score_ij;
                    align_from_right = hij[prev_idx + 1] + score_ij;
                    align_from_down = hij[prev_idx + 2] + score_ij;
                }

                float options[4] = {align_from_align, align_from_right, align_from_down, sky};
                float max_val = NINF;
                for (int t = 0; t < 4; t++)
                    max_val = std::max(max_val, options[t]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[4];
                    for (int t = 0; t < 4; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 4; t++)
                        weights[t] /= sum_exp;

                    if (i > 0 && j > 0) {
                        int prev_idx = (i - 1) * L2 * 3 + (j - 1) * 3;
                        beta[prev_idx + 0] += grad_align * weights[0];
                        beta[prev_idx + 1] += grad_align * weights[1];
                        beta[prev_idx + 2] += grad_align * weights[2];
                        posteriors[i * L2 + j] +=
                            grad_align * (weights[0] + weights[1] + weights[2]);
                    }
                    posteriors[i * L2 + j] += grad_align * weights[3];
                }
            }

            // Right state - STANDARD (2 transitions, NO Down->Right)
            int right_idx = i * L2 * 3 + j * 3 + 1;
            float grad_right = beta[right_idx];

            if (!std::isinf(grad_right) && grad_right != 0.0f && j > 0) {
                int prev_idx = i * L2 * 3 + (j - 1) * 3;
                float right_from_align = hij[prev_idx + 0] + gap_open;     // Raw gap
                float right_from_right = hij[prev_idx + 1] + gap_extend;   // Raw gap

                float options[2] = {right_from_align, right_from_right};
                float max_val = std::max(options[0], options[1]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[2];
                    for (int t = 0; t < 2; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 2; t++)
                        weights[t] /= sum_exp;

                    beta[prev_idx + 0] += grad_right * weights[0];
                    beta[prev_idx + 1] += grad_right * weights[1];
                }
            }

            // Down state - STANDARD (2 transitions, NO Right->Down)
            int down_idx = i * L2 * 3 + j * 3 + 2;
            float grad_down = beta[down_idx];

            if (!std::isinf(grad_down) && grad_down != 0.0f && i > 0) {
                int prev_idx = (i - 1) * L2 * 3 + j * 3;
                float down_from_align = hij[prev_idx + 0] + gap_open;    // Raw gap
                float down_from_down = hij[prev_idx + 2] + gap_extend;   // Raw gap

                float options[2] = {down_from_align, down_from_down};
                float max_val = std::max(options[0], options[1]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[2];
                    for (int t = 0; t < 2; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 2; t++)
                        weights[t] /= sum_exp;

                    beta[prev_idx + 0] += grad_down * weights[0];
                    beta[prev_idx + 2] += grad_down * weights[1];
                }
            }
        }
    }

    // Cleanup only if manually allocated
    if (allocated_beta) {
        delete[] beta;
    }
}

/**
 * Direct Affine Backward Pass (STANDARD model)
 *
 * Similar to Direct affine flexible but with STANDARD affine gaps:
 * - I state: 2 transitions (M, I)
 * - D state: 2 transitions (M, D) - NO I->D
 */
template <>
void smith_waterman_direct_affine_backward<ScalarBackend>(
    const float* alpha, const float* scores, int L1, int L2, const SWConfig& config,
    float partition, float* posteriors, pfalign::memory::GrowableArena* temp_arena) {
    float T = config.temperature;
    float gap_open = config.gap_open;     // Raw gap open, no pre-scaling
    float gap_extend = config.gap_extend; // Raw gap extend, no pre-scaling
    float upstream = 1.0f;

    // Initialize posteriors to 0
    std::memset(posteriors, 0, L1 * L2 * sizeof(float));

    // Allocate workspace for backward DP values (grad_alpha) - 3 states
    int size_per_state = (L1 + 1) * (L2 + 1);
    float* grad_alpha;
    bool allocated_grad_alpha = false;
    if (temp_arena) {
        grad_alpha = temp_arena->allocate<float>(size_per_state * 3);
    } else {
        grad_alpha = new float[size_per_state * 3];
        allocated_grad_alpha = true;
    }
    std::memset(grad_alpha, 0, size_per_state * 3 * sizeof(float));

    const float* M = alpha;
    const float* I = alpha + size_per_state;
    const float* D = alpha + 2 * size_per_state;

    float* grad_M = grad_alpha;
    float* grad_I = grad_alpha + size_per_state;
    float* grad_D = grad_alpha + 2 * size_per_state;

    // ========================================================================
    // STEP 1: Compute ∂S/∂alpha for all positions
    // ========================================================================

    for (int i = 0; i <= L1; i++) {
        for (int j = 0; j <= L2; j++) {
            int idx = i * (L2 + 1) + j;

            if (i == 0 && j == 0) {
                float exponent = -partition;
                grad_M[idx] = upstream * std::exp(exponent / T);  // Divide by T (Approach B)
                continue;
            }

            if (i < 1 || i > L1 || j < 1 || j > L2)
                continue;

            for (int state = 0; state < 3; state++) {
                const float* state_alpha = (state == 0) ? M : (state == 1) ? I : D;
                float* state_grad = (state == 0) ? grad_M : (state == 1) ? grad_I : grad_D;
                float alpha_val = state_alpha[idx];
                float exponent = alpha_val - partition;
                state_grad[idx] = upstream * std::exp(exponent / T);  // Divide by T (Approach B)
            }
        }
    }

    // ========================================================================
    // STEP 2: Backprop through DP recursion
    // ========================================================================

    for (int i = L1; i >= 1; i--) {
        for (int j = L2; j >= 1; j--) {
            int idx = i * (L2 + 1) + j;
            int idx_diag = (i - 1) * (L2 + 1) + (j - 1);
            int idx_up = (i - 1) * (L2 + 1) + j;
            int idx_left = i * (L2 + 1) + (j - 1);

            int score_idx = (i - 1) * L2 + (j - 1);
            float score_ij = scores[score_idx];  // Raw score (Approach B)

            // M state
            float grad_m = grad_M[idx];
            if (grad_m != 0.0f) {
                float m_from_m = (i > 1 && j > 1) ? M[idx_diag] + score_ij : NINF;
                float m_from_i = (i > 1 && j > 1) ? I[idx_diag] + score_ij : NINF;
                float m_from_d = (i > 1 && j > 1) ? D[idx_diag] + score_ij : NINF;
                float sky = score_ij;

                float options[4] = {m_from_m, m_from_i, m_from_d, sky};
                float max_val = NINF;
                for (int t = 0; t < 4; t++)
                    max_val = std::max(max_val, options[t]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[4];
                    for (int t = 0; t < 4; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 4; t++)
                        weights[t] /= sum_exp;

                    if (i > 1 && j > 1) {
                        grad_M[idx_diag] += grad_m * weights[0];
                        grad_I[idx_diag] += grad_m * weights[1];
                        grad_D[idx_diag] += grad_m * weights[2];
                        posteriors[score_idx] += grad_m * (weights[0] + weights[1] + weights[2]);
                    }
                    posteriors[score_idx] += grad_m * weights[3];
                }
            }

            // I state - STANDARD
            float grad_i = grad_I[idx];
            if (grad_i != 0.0f && i > 1) {
                float i_from_m = M[idx_up] + gap_open;     // Raw gap
                float i_from_i = I[idx_up] + gap_extend;   // Raw gap

                float options[2] = {i_from_m, i_from_i};
                float max_val = std::max(options[0], options[1]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[2];
                    for (int t = 0; t < 2; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 2; t++)
                        weights[t] /= sum_exp;

                    grad_M[idx_up] += grad_i * weights[0];
                    grad_I[idx_up] += grad_i * weights[1];
                }
            }

            // D state - STANDARD (NO I->D transition)
            float grad_d = grad_D[idx];
            if (grad_d != 0.0f && j > 1) {
                float d_from_m = M[idx_left] + gap_open;    // Raw gap
                float d_from_d = D[idx_left] + gap_extend;  // Raw gap

                float options[2] = {d_from_m, d_from_d};
                float max_val = std::max(options[0], options[1]);

                if (!(std::isinf(max_val) && max_val < 0)) {
                    float sum_exp = 0.0f;
                    float weights[2];
                    for (int t = 0; t < 2; t++) {
                        weights[t] = std::isinf(options[t]) ? 0.0f : std::exp((options[t] - max_val) / T);  // Divide by T (Approach B)
                        sum_exp += weights[t];
                    }
                    for (int t = 0; t < 2; t++)
                        weights[t] /= sum_exp;

                    grad_M[idx_left] += grad_d * weights[0];
                    grad_D[idx_left] += grad_d * weights[1];
                }
            }
        }
    }

    // Cleanup only if manually allocated
    if (allocated_grad_alpha) {
        delete[] grad_alpha;
    }
}

}  // namespace smith_waterman
}  // namespace pfalign
