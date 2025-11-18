#include "rbf_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>

namespace pfalign {
namespace rbf {

/**
 * Scalar RBF implementation.
 *
 * Simple, unoptimized version for reference and CPU fallback.
 * Computes Gaussian RBF: exp(-((dist - center) / sigma)^2)
 *
 * Performance: ~20-30 ns per distance (16 bins)
 */

/**
 * Compute RBF features for a single distance.
 */
template <>
void rbf_single<ScalarBackend>(float distance, const float* centers, float inv_sigma_sq,
                               float* features, int num_bins) {
    // For each Gaussian center, compute: exp(-((distance - center) / sigma)^2)
    // Optimized as: exp(-(distance - center)^2 * inv_sigma_sq)
    for (int i = 0; i < num_bins; i++) {
        float diff = distance - centers[i];
        float exponent = -(diff * diff) * inv_sigma_sq;
        features[i] = std::exp(exponent);
    }
}

/**
 * Compute RBF features for multiple distances (batched).
 */
template <>
void rbf_batch<ScalarBackend>(const float* distances, const float* centers, float inv_sigma_sq,
                              float* features, int N, int num_bins) {
    // Process each distance independently
    for (int n = 0; n < N; n++) {
        rbf_single<ScalarBackend>(distances[n], centers, inv_sigma_sq, features + n * num_bins,
                                  num_bins);
    }
}

}  // namespace rbf
}  // namespace pfalign
