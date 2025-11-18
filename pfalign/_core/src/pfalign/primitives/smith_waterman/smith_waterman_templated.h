/**
 * Templated Smith-Waterman implementations for testing different precisions.
 *
 * This file contains templated versions of the Direct Regular algorithm
 * to test whether temperature invariance is affected by floating point precision.
 *
 * Supports: float16 (half), float32 (float), float64 (double)
 */

#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

namespace pfalign {
namespace smith_waterman {

/**
 * Templated configuration for different precisions.
 */
template <typename T>
struct SWConfigT {
    T gap;          // Gap penalty (negative)
    T gap_open;     // Gap opening penalty (negative)
    T gap_extend;   // Gap extension penalty (negative)
    T temperature;  // Temperature for soft max
    bool affine;    // Use affine gap penalties

    SWConfigT() : gap(static_cast<T>(-0.1)),
                  gap_open(static_cast<T>(-1.0)),
                  gap_extend(static_cast<T>(-0.1)),
                  temperature(static_cast<T>(1.0)),
                  affine(false) {}

    SWConfigT(T gap_, T temp_) : gap(gap_),
                                 gap_open(static_cast<T>(-1.0)),
                                 gap_extend(static_cast<T>(-0.1)),
                                 temperature(temp_),
                                 affine(false) {}
};

/**
 * Get -inf value for a given type.
 */
template <typename T>
constexpr T get_ninf() {
    return -static_cast<T>(1e30);
}

// Specialization for __fp16 (needs smaller value to avoid overflow)
#ifdef __ARM_FP16_FORMAT_IEEE
template <>
constexpr __fp16 get_ninf<__fp16>() {
    return -static_cast<__fp16>(65000);  // Max __fp16 is ~65504
}
#endif

/**
 * Templated soft_max (logsumexp) for numerical stability.
 *
 * Formula: soft_max(x, T) = max(x) + T * log(sum(exp((x_i - max(x)) / T)))
 *
 * This is the Approach B (Boltzmann) formulation:
 * - Takes raw values (not pre-scaled by temperature)
 * - Returns max + T*log(sum) (includes T factor in partition)
 */
template <typename T>
inline T soft_max_templated(const T* values, int n, T temperature) {
    // Find maximum for numerical stability
    T max_val = values[0];
    for (int i = 1; i < n; i++) {
        max_val = std::max(max_val, values[i]);
    }

    // Handle -inf case
    constexpr T ninf = get_ninf<T>();
    if (std::isinf(max_val) && max_val < static_cast<T>(0)) {
        return ninf;
    }

    // Compute sum of exp
    T sum_exp = static_cast<T>(0);
    for (int i = 0; i < n; i++) {
        sum_exp += std::exp((values[i] - max_val) / temperature);
    }

    return temperature * (std::log(sum_exp) + max_val / temperature);
}

/**
 * Templated Direct Regular Smith-Waterman (forward pass).
 *
 * Approach B (Boltzmann formulation):
 * - Scores and gaps are raw values (NOT divided by T)
 * - soft_max uses actual temperature T
 * - Partition = max + T*log(sum)
 *
 * @tparam T        Floating point type (float16/float32/float64)
 * @param scores    Similarity matrix [L1 * L2]
 * @param L1        Length of sequence 1
 * @param L2        Length of sequence 2
 * @param config    Configuration (gap penalty, temperature)
 * @param alpha     Output DP matrix [(L1+1) * (L2+1)]
 * @param partition Output partition function
 */
template <typename T>
void smith_waterman_direct_regular_templated(
    const T* scores,
    int L1,
    int L2,
    const SWConfigT<T>& config,
    T* alpha,
    T* partition
) {
    constexpr T ninf = get_ninf<T>();
    const T temperature = config.temperature;
    const T gap_penalty = config.gap;  // Raw gap, no pre-scaling

    // Initialize boundaries to -inf (LOCAL alignment)
    const int size = (L1 + 1) * (L2 + 1);
    for (int i = 0; i < size; i++) {
        alpha[i] = ninf;
    }

    // Fill DP matrix (1-indexed)
    for (int i = 1; i <= L1; i++) {
        for (int j = 1; j <= L2; j++) {
            const T score_ij = scores[(i - 1) * L2 + (j - 1)];  // Raw score

            // Four transitions
            const T align = alpha[(i - 1) * (L2 + 1) + (j - 1)] + score_ij;
            const T gap_up = alpha[(i - 1) * (L2 + 1) + j] + gap_penalty;
            const T gap_left = alpha[i * (L2 + 1) + (j - 1)] + gap_penalty;
            const T sky = score_ij;  // LOCAL: start fresh

            T values[4] = {align, gap_up, gap_left, sky};
            alpha[i * (L2 + 1) + j] = soft_max_templated(values, 4, temperature);
        }
    }

    // Compute partition function: max + T*log(sum)
    T max_alpha = ninf;
    for (int i = 0; i < size; i++) {
        max_alpha = std::max(max_alpha, alpha[i]);
    }

    T sum_exp = static_cast<T>(0);
    for (int i = 0; i < size; i++) {
        if (alpha[i] > ninf) {
            sum_exp += std::exp((alpha[i] - max_alpha) / temperature);
        }
    }

    *partition = max_alpha + temperature * std::log(sum_exp);  // Boltzmann: include T factor
}

/**
 * Templated Direct Regular Smith-Waterman (backward pass).
 *
 * Computes posterior probabilities P(i aligns to j).
 *
 * Approach B backward formula:
 * - weight = exp((z - partition) / T)  (divide by T)
 *
 * @tparam T         Floating point type
 * @param alpha      DP values from forward pass [(L1+1) * (L2+1)]
 * @param scores     Similarity matrix [L1 * L2]
 * @param L1         Length of sequence 1
 * @param L2         Length of sequence 2
 * @param config     Configuration (gap penalty, temperature)
 * @param partition  Partition function from forward pass
 * @param posteriors Output posterior probabilities [L1 * L2]
 */
template <typename T>
void smith_waterman_direct_regular_backward_templated(
    const T* alpha,
    const T* scores,
    int L1,
    int L2,
    const SWConfigT<T>& config,
    T partition,
    T* posteriors
) {
    constexpr T ninf = get_ninf<T>();
    const T temperature = config.temperature;
    const T gap_penalty = config.gap;  // Raw gap

    // Allocate beta matrix (backward values)
    const int size = (L1 + 1) * (L2 + 1);
    T* beta = new T[size];

    // Initialize all beta to 0 (in log space: 1.0)
    for (int i = 0; i < size; i++) {
        beta[i] = static_cast<T>(0);
    }

    // Backward pass (reverse order)
    for (int i = L1; i >= 1; i--) {
        for (int j = L2; j >= 1; j--) {
            const T score_ij = scores[(i - 1) * L2 + (j - 1)];  // Raw score

            // Current beta
            T beta_curr = beta[i * (L2 + 1) + j];

            // Four incoming transitions from (i,j)

            // 1. Align: came from (i-1, j-1)
            if (i > 1 && j > 1) {
                const T score_next = scores[i * L2 + j];
                const T z_align = alpha[(i - 1) * (L2 + 1) + (j - 1)] + score_next;
                const T alpha_curr = alpha[i * (L2 + 1) + j];
                const T weight = std::exp((z_align - alpha_curr) / temperature);  // Divide by T
                beta[(i - 1) * (L2 + 1) + (j - 1)] += weight * beta_curr;
            }

            // 2. Gap up: came from (i-1, j)
            if (i > 1) {
                const T z_gap = alpha[(i - 1) * (L2 + 1) + j] + gap_penalty;
                const T alpha_curr = alpha[i * (L2 + 1) + j];
                const T weight = std::exp((z_gap - alpha_curr) / temperature);  // Divide by T
                beta[(i - 1) * (L2 + 1) + j] += weight * beta_curr;
            }

            // 3. Gap left: came from (i, j-1)
            if (j > 1) {
                const T z_gap = alpha[i * (L2 + 1) + (j - 1)] + gap_penalty;
                const T alpha_curr = alpha[i * (L2 + 1) + j];
                const T weight = std::exp((z_gap - alpha_curr) / temperature);  // Divide by T
                beta[i * (L2 + 1) + (j - 1)] += weight * beta_curr;
            }

            // 4. Sky: started at this position
            // No propagation needed (sky doesn't come from anywhere)
        }
    }

    // Compute posteriors
    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L2; j++) {
            // Posterior = exp(alpha + beta - partition)
            const T alpha_val = alpha[(i + 1) * (L2 + 1) + (j + 1)];
            const T beta_val = beta[(i + 1) * (L2 + 1) + (j + 1)];

            // Approach B: divide by T
            posteriors[i * L2 + j] = std::exp((alpha_val + beta_val - partition) / temperature);
        }
    }

    delete[] beta;
}

}  // namespace smith_waterman
}  // namespace pfalign
