/**
 * Scalar implementation of distance matrix computation.
 */

#include "distance_matrix.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>

namespace pfalign {
namespace structural_metrics {

/**
 * Compute pairwise distance matrix (Scalar backend).
 *
 * Implementation strategy:
 * 1. Compute upper triangle only (j > i) to avoid redundant computation
 * 2. Copy to lower triangle using symmetry
 * 3. Set diagonal to zero
 *
 * This saves ~50% computation compared to computing all pairs.
 */
template <>
void compute_distance_matrix<ScalarBackend>(const float* ca_coords, int L, float* dist_mx) {
    // Step 1: Initialize diagonal to zero
    for (int i = 0; i < L; i++) {
        dist_mx[i * L + i] = 0.0f;
    }

    // Step 2: Compute upper triangle (j > i) and copy to lower triangle
    for (int i = 0; i < L; i++) {
        const float* ca_i = &ca_coords[i * 3];

        for (int j = i + 1; j < L; j++) {
            const float* ca_j = &ca_coords[j * 3];

            // Compute Euclidean distance
            float dx = ca_i[0] - ca_j[0];
            float dy = ca_i[1] - ca_j[1];
            float dz = ca_i[2] - ca_j[2];

            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            // Store in both upper and lower triangle
            dist_mx[i * L + j] = dist;
            dist_mx[j * L + i] = dist;  // Symmetry
        }
    }
}

// ============================================================================
}  // namespace structural_metrics
}  // namespace pfalign
