#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace layer_norm {

/**
 * Layer Normalization for MPNN encoder.
 *
 * Normalizes activations across the feature dimension:
 *   LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta
 *
 * where:
 *   mu = mean(x) over features
 *   sigma^2 = variance(x) over features
 *   gamma = scale parameter (learned)
 *   beta = bias parameter (learned)
 *   epsilon = small constant for numerical stability (default: 1e-5)
 *
 * Used in MPNN after message passing to stabilize training.
 *
 * Reference:
 *   Ba et al. "Layer Normalization" (2016)
 *   https://arxiv.org/abs/1607.06450
 */

/**
 * Layer normalization for a single vector.
 *
 * Normalizes a D-dimensional vector to have mean=0, variance=1,
 * then applies learned affine transformation.
 *
 * @param input         Input vector [D]
 * @param output        Output vector [D]
 * @param gamma         Scale parameters [D] (can be nullptr for no scaling)
 * @param beta          Bias parameters [D] (can be nullptr for no bias)
 * @param D             Feature dimension
 * @param eps           Epsilon for numerical stability (default: 1e-5)
 *
 * Example:
 *   float input[128];
 *   float output[128];
 *   float gamma[128];  // Learned scales
 *   float beta[128];   // Learned biases
 *   layer_norm_forward<ScalarBackend>(input, output, gamma, beta, 128);
 */
template <typename Backend>
void layer_norm_forward(const float* input, float* output, const float* gamma, const float* beta,
                        int D, float eps = 1e-5f);

/**
 * Layer normalization for a batch of vectors.
 *
 * Applies layer norm independently to each vector in the batch.
 *
 * @param input         Input vectors [N * D] (row-major)
 * @param output        Output vectors [N * D] (row-major)
 * @param gamma         Scale parameters [D]
 * @param beta          Bias parameters [D]
 * @param N             Batch size (number of vectors)
 * @param D             Feature dimension
 * @param eps           Epsilon for numerical stability
 *
 * Example:
 *   float input[100 * 128];   // 100 embeddings, 128 dim
 *   float output[100 * 128];
 *   float gamma[128];
 *   float beta[128];
 *   layer_norm_batch<ScalarBackend>(input, output, gamma, beta, 100, 128);
 */
template <typename Backend>
void layer_norm_batch(const float* input, float* output, const float* gamma, const float* beta,
                      int N, int D, float eps = 1e-5f);

/**
 * RMS (Root Mean Square) normalization variant.
 *
 * Simpler than full layer norm - only normalizes by RMS (no mean centering).
 * Used in some modern architectures (e.g., LLaMA).
 *
 *   RMSNorm(x) = gamma * x / sqrt(mean(x^2) + epsilon)
 *
 * @param input         Input vector [D]
 * @param output        Output vector [D]
 * @param gamma         Scale parameters [D]
 * @param D             Feature dimension
 * @param eps           Epsilon for numerical stability
 */
template <typename Backend>
void rms_norm_forward(const float* input, float* output, const float* gamma, int D,
                      float eps = 1e-5f);

}  // namespace layer_norm
}  // namespace pfalign
