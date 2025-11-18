#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace rbf {

/**
 * Radial Basis Function (RBF) kernel for MPNN atom-pair features.
 *
 * Computes Gaussian RBF features from distances:
 *   RBF[i] = exp(-((distance - center[i]) / sigma)^2)
 *
 * Used in MPNN encoder to featurize pairwise atom distances into
 * a fixed-dimensional representation.
 *
 * Default parameters (matching JAX reference):
 * - num_bins: 16 Gaussian centers
 * - distance_range: 2.0 to 22.0 Å
 * - sigma: 1.25 Å (automatically computed)
 *
 * Reference:
 *   align/_jax_reference/mpnn.py::_rbf()
 */

/**
 * Compute RBF features for a single distance.
 *
 * Evaluates num_bins Gaussian basis functions centered at evenly-spaced
 * points along the distance range.
 *
 * @param distance      Input distance in Angstroms
 * @param centers       Array of RBF centers [num_bins] (precomputed)
 * @param inv_sigma_sq  Precomputed 1/(sigma^2) for efficiency
 * @param features      Output RBF features [num_bins]
 * @param num_bins      Number of Gaussian centers (default: 16)
 *
 * Example:
 *   float centers[16];
 *   rbf_initialize_centers(centers, 16, 2.0f, 22.0f);
 *   float inv_sigma_sq = rbf_compute_inv_sigma_sq(2.0f, 22.0f, 16);
 *
 *   float distance = 10.5f;
 *   float features[16];
 *   rbf_single<ScalarBackend>(distance, centers, inv_sigma_sq, features, 16);
 */
template <typename Backend>
void rbf_single(float distance, const float* centers, float inv_sigma_sq, float* features,
                int num_bins);

/**
 * Compute RBF features for multiple distances (batched).
 *
 * @param distances     Input distances [N]
 * @param centers       RBF centers [num_bins]
 * @param inv_sigma_sq  Precomputed 1/(sigma^2)
 * @param features      Output RBF features [N * num_bins] (row-major)
 * @param N             Number of distances
 * @param num_bins      Number of Gaussian centers
 */
template <typename Backend>
void rbf_batch(const float* distances, const float* centers, float inv_sigma_sq, float* features,
               int N, int num_bins);

/**
 * Initialize RBF centers (evenly spaced along distance range).
 *
 * centers[i] = min_dist + i * step
 * where step = (max_dist - min_dist) / (num_bins - 1)
 *
 * @param centers       Output centers [num_bins]
 * @param num_bins      Number of centers
 * @param min_dist      Minimum distance (default: 2.0 Å)
 * @param max_dist      Maximum distance (default: 22.0 Å)
 */
inline void rbf_initialize_centers(float* centers, int num_bins, float min_dist = 2.0f,
                                   float max_dist = 22.0f) {
    float step = (max_dist - min_dist) / (num_bins - 1);
    for (int i = 0; i < num_bins; i++) {
        centers[i] = min_dist + i * step;
    }
}

/**
 * Compute inverse sigma squared for efficiency.
 *
 * sigma = (max_dist - min_dist) / num_bins
 * inv_sigma_sq = 1 / sigma^2
 *
 * @param min_dist      Minimum distance
 * @param max_dist      Maximum distance
 * @param num_bins      Number of bins
 * @return              1/(sigma^2)
 */
inline float rbf_compute_inv_sigma_sq(float min_dist, float max_dist, int num_bins) {
    float sigma = (max_dist - min_dist) / num_bins;
    return 1.0f / (sigma * sigma);
}

}  // namespace rbf
}  // namespace pfalign
