/**
 * DALI (Distance Alignment) Scoring
 *
 * Computes structural alignment quality using distance matrix comparison.
 * Returns both raw DALI score and Z-score normalized for alignment length.
 *
 * Reference:
 *   Holm & Sander (1993). "Protein structure comparison by alignment
 *   of distance matrices." Journal of Molecular Biology 233.1: 123-138.
 *
 * DALI score measures structural similarity by comparing distance patterns
 * between aligned structures, with exponential weighting favoring shorter
 * distances. The Z-score normalization accounts for alignment length.
 */

#pragma once

#include "pfalign/dispatch/backend_traits.h"
#include <cmath>
#include <algorithm>

namespace pfalign {
namespace structural_metrics {

/**
 * DALI result containing raw score and Z-score.
 */
struct DALIResult {
    float score;  // Raw DALI score
    float Z;      // Z-score (length-normalized)
};

/**
 * DALI scoring parameters.
 */
struct DALIParams {
    float horizon = 20.0f;           // Horizon parameter D (Å) - distance weighting decay
    float diagwt = 0.2f;             // Diagonal weight d0
    float R0 = -1.0f;                // Optional inclusion radius (-1 = no limit)
    const char* symmetry = "first";  // Symmetry mode (for R0 filtering)
};

/**
 * Compute DALI score and Z-score for pairwise structural comparison.
 *
 * DALI scoring algorithm:
 * 1. For each aligned pair (i,j) and (i',j'):
 *    - Compute mean distance: mean = (d_ij + d_i'j') / 2
 *    - Weight: w = exp(-(mean/horizon)^2)
 *    - Distance difference ratio: ratio = |d_ij - d_i'j'| / mean
 *    - Score contribution: w * (diagwt - ratio)
 * 2. Total score = sum of contributions + N * diagwt
 * 3. Z-score = (score - expected_mean(N)) / expected_sigma(N)
 *
 * The Z-score normalization uses empirical formulas calibrated on
 * protein structure databases to account for random similarity based
 * on alignment length.
 *
 * @tparam Backend Computation backend
 * @param dist_mx1 Distance matrix for structure 1 [L1 * L1]
 * @param dist_mx2 Distance matrix for structure 2 [L2 * L2]
 * @param alignment Aligned position pairs [aligned_length * 2]
 * @param aligned_length Number of aligned residue pairs
 * @param L1 Length of structure 1 (for Z-score normalization)
 * @param L2 Length of structure 2 (for Z-score normalization)
 * @param params DALI parameters
 * @return DALIResult containing raw score and Z-score
 *
 * Example:
 * ```cpp
 *   // Compute distance matrices
 *   std::vector<float> dist1(L1 * L1), dist2(L2 * L2);
 *   compute_distance_matrix<ScalarBackend>(ca1, L1, dist1.data());
 *   compute_distance_matrix<ScalarBackend>(ca2, L2, dist2.data());
 *
 *   // Define alignment
 *   std::vector<int> alignment(N * 2);
 *   for (int i = 0; i < N; i++) {
 *       alignment[i*2 + 0] = i;
 *       alignment[i*2 + 1] = i;
 *   }
 *
 *   // Compute DALI
 *   DALIParams params;
 *   DALIResult result = dali_score<ScalarBackend>(
 *       dist1.data(), dist2.data(), alignment.data(), N, L1, L2, params
 *   );
 *
 *   std::cout << "DALI score: " << result.score << "\n";
 *   std::cout << "DALI Z-score: " << result.Z << "\n";
 * ```
 *
 * Interpretation:
 * - Z > 2: Likely structural similarity
 * - Z > 5: High confidence homology
 * - Z > 20: Highly significant similarity
 *
 * Performance:
 * - Scalar: O(N^2) ~10ms for N=200
 */
template <typename Backend>
DALIResult dali_score(const float* dist_mx1,  // [L1 * L1] structure 1 distance matrix
                      const float* dist_mx2,  // [L2 * L2] structure 2 distance matrix
                      const int* alignment,   // [aligned_length * 2] position pairs
                      int aligned_length,     // number of aligned pairs
                      int L1,                 // length of structure 1
                      int L2,                 // length of structure 2
                      const DALIParams& params = DALIParams());

/**
 * Compute DALI Z-score from raw score and alignment lengths.
 *
 * Uses empirical formula calibrated on protein structure databases:
 *   n12 = sqrt(L1 * L2)
 *   x = min(n12, 400)
 *   mean = 7.9494 + 0.70852*x + 2.5895e-4*x^2 - 1.9156e-6*x^3
 *   if n12 > 400: mean += (n12 - 400)
 *   sigma = 0.5 * mean
 *   Z = (score - mean) / max(1.0, sigma)
 *
 * This normalization allows fair comparison of DALI scores across
 * different alignment lengths.
 *
 * @param score Raw DALI score
 * @param L1 Length of structure 1
 * @param L2 Length of structure 2
 * @return Z-score
 *
 * Example:
 * ```cpp
 *   float raw_score = 150.0f;
 *   float z = dali_Z_from_score_and_lengths(raw_score, 200, 200);
 *   // z ~= 10.5 (highly significant)
 * ```
 */
inline float dali_Z_from_score_and_lengths(float score, int L1, int L2) {
    float n12 = std::sqrt(static_cast<float>(L1) * static_cast<float>(L2));
    float x = std::min(n12, 400.0f);

    // Polynomial fit for expected mean
    float mean = 7.9494f + 0.70852f * x + 2.5895e-4f * x * x - 1.9156e-6f * x * x * x;

    // Linear extrapolation for very long alignments
    if (n12 > 400.0f) {
        mean += (n12 - 400.0f);
    }

    // Expected standard deviation
    float sigma = 0.5f * mean;

    // Z-score
    float Z = (score - mean) / std::max(1.0f, sigma);
    return Z;
}

}  // namespace structural_metrics
}  // namespace pfalign
