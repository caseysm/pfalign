/**
 * Scalar implementation of DALI scoring.
 */

#include "dali_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace pfalign {
namespace structural_metrics {

/**
 * Helper: Compute DALI weight based on mean distance.
 *
 * w = exp(-(mean_distance / horizon)^2)
 */
static inline float dali_weight(float mean_distance, float horizon) {
    float ratio = mean_distance / horizon;
    return std::exp(-ratio * ratio);
}

/**
 * Helper: Compute DALI distance pair score.
 *
 * score = weight * (diagwt - |d1 - d2| / mean)
 */
static inline float dali_pair_score(float d1, float d2, float diagwt, float horizon) {
    float diff = std::abs(d1 - d2);
    float mean = (d1 + d2) / 2.0f;

    // Avoid division by zero and limit contribution from very distant pairs
    if (mean > 100.0f) {
        return 0.0f;
    }

    float weight = dali_weight(mean, horizon);

    if (mean == 0.0f) {
        return weight * diagwt;
    } else {
        float ratio = diff / mean;
        return weight * (diagwt - ratio);
    }
}

/**
 * Helper: Check if distance pair should be included (if R0 is specified).
 */
static inline bool should_include_dali_pair(float d1, float d2, float R0, const char* symmetry) {
    if (R0 < 0.0f) {
        return true;  // No R0 filtering
    }

    if (std::strcmp(symmetry, "first") == 0) {
        return d1 <= R0;
    } else if (std::strcmp(symmetry, "both") == 0) {
        return (d1 <= R0) && (d2 <= R0);
    } else if (std::strcmp(symmetry, "either") == 0) {
        return (d1 <= R0) || (d2 <= R0);
    }
    return d1 <= R0;
}

/**
 * DALI score implementation (Scalar backend).
 */
template <>
DALIResult dali_score<ScalarBackend>(const float* dist_mx1, const float* dist_mx2,
                                     const int* alignment, int aligned_length, int L1, int L2,
                                     const DALIParams& params) {
    if (aligned_length == 0) {
        return {0.0f, 0.0f};
    }

    float score = 0.0f;

    // For each aligned pair (coli, colj)
    for (int coli = 0; coli < aligned_length; coli++) {
        int pos1i = alignment[coli * 2 + 0];
        int pos2i = alignment[coli * 2 + 1];

        if (pos1i < 0 || pos2i < 0)
            continue;  // Skip gaps

        for (int colj = 0; colj < aligned_length; colj++) {
            if (colj == coli)
                continue;

            int pos1j = alignment[colj * 2 + 0];
            int pos2j = alignment[colj * 2 + 1];

            if (pos1j < 0 || pos2j < 0)
                continue;  // Skip gaps

            // Get distances from distance matrices
            float d1 = dist_mx1[pos1i * L1 + pos1j];
            float d2 = dist_mx2[pos2i * L2 + pos2j];

            // Apply R0 filter if specified
            if (!should_include_dali_pair(d1, d2, params.R0, params.symmetry)) {
                continue;
            }

            // Compute DALI pair score
            float pair_score = dali_pair_score(d1, d2, params.diagwt, params.horizon);
            score += pair_score;
        }
    }

    // Add diagonal contribution: N * diagwt
    score += aligned_length * params.diagwt;

    // Compute Z-score
    float Z = dali_Z_from_score_and_lengths(score, L1, L2);

    return {score, Z};
}

// ============================================================================
}  // namespace structural_metrics
}  // namespace pfalign
