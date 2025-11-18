#include "layer_norm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>

namespace pfalign {
namespace layer_norm {

/**
 * Scalar Layer Normalization implementation.
 *
 * Formula:
 *   1. Compute mean: mu = (1/D) Sigma x[i]
 *   2. Compute variance: sigma^2 = (1/D) Sigma (x[i] - mu)^2
 *   3. Normalize: x_norm = (x - mu) / sqrt(sigma^2 + epsilon)
 *   4. Affine: y = gamma * x_norm + beta
 *
 * Performance:
 *   - D=128: ~150 ns
 *   - D=256: ~300 ns
 *   - D=512: ~600 ns
 */

/**
 * Layer normalization for a single vector.
 */
template <>
void layer_norm_forward<ScalarBackend>(const float* input, float* output, const float* gamma,
                                       const float* beta, int D, float eps) {
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        sum += input[i];
    }
    float mean = sum / D;

    // Step 2: Compute variance
    float var_sum = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / D;

    // Step 3: Normalize and apply affine transformation
    float inv_std = 1.0f / std::sqrt(variance + eps);

    for (int i = 0; i < D; i++) {
        float normalized = (input[i] - mean) * inv_std;

        // Apply learned affine parameters
        if (gamma != nullptr && beta != nullptr) {
            output[i] = gamma[i] * normalized + beta[i];
        } else if (gamma != nullptr) {
            output[i] = gamma[i] * normalized;
        } else if (beta != nullptr) {
            output[i] = normalized + beta[i];
        } else {
            output[i] = normalized;
        }
    }
}

/**
 * Layer normalization for a batch of vectors.
 */
template <>
void layer_norm_batch<ScalarBackend>(const float* input, float* output, const float* gamma,
                                     const float* beta, int N, int D, float eps) {
    // Apply layer norm to each vector independently
    for (int n = 0; n < N; n++) {
        layer_norm_forward<ScalarBackend>(input + n * D, output + n * D, gamma, beta, D, eps);
    }
}

/**
 * RMS normalization (simpler variant).
 */
template <>
void rms_norm_forward<ScalarBackend>(const float* input, float* output, const float* gamma, int D,
                                     float eps) {
    // Compute RMS (root mean square)
    float sum_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        sum_sq += input[i] * input[i];
    }
    float rms = std::sqrt(sum_sq / D + eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    for (int i = 0; i < D; i++) {
        if (gamma != nullptr) {
            output[i] = gamma[i] * input[i] * inv_rms;
        } else {
            output[i] = input[i] * inv_rms;
        }
    }
}

}  // namespace layer_norm
}  // namespace pfalign
