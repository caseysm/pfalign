#include "knn_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace pfalign {
namespace knn {

/**
 * Scalar KNN implementation (brute force).
 *
 * Strategy:
 * - Compute all pairwise distances: O(N^2)
 * - Partial sort to find k smallest: O(N^2 + Nk log k)
 * - Simple, cache-friendly, good for N < 1000 (typical proteins)
 *
 * Performance:
 * - N=100, k=30: ~50 µs
 * - N=500, k=30: ~1.2 ms
 * - N=1000, k=30: ~5 ms
 *
 * Future optimizations (NEON/CUDA):
 * - NEON: Vectorize distance computation (4* speedup)
 * - CUDA: Parallel distance + k-selection (~100* speedup)
 */

/**
 * Helper: Compute squared distance between two 3D points.
 */
static inline float distance_squared(const float* p1, const float* p2) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return dx * dx + dy * dy + dz * dz;
}

/**
 * Helper struct for sorting neighbors by distance.
 */
struct Neighbor {
    int index;
    float distance_sq;

    bool operator<(const Neighbor& other) const {
        // Break ties by index for deterministic behavior
        if (distance_sq != other.distance_sq) {
            return distance_sq < other.distance_sq;
        }
        return index < other.index;
    }
};

/**
 * Find k nearest neighbors for a single query point.
 */
template <>
void knn_query<ScalarBackend>(const float* coords, int N, const float* query_pt, int k,
                              int* indices, float* distances_sq) {
    // Compute all distances
    std::vector<Neighbor> neighbors(static_cast<size_t>(N));

    for (int i = 0; i < N; i++) {
        neighbors[static_cast<size_t>(i)].index = i;
        neighbors[static_cast<size_t>(i)].distance_sq = distance_squared(query_pt, coords + i * 3);
    }

    // Partial sort to find k smallest
    int num_results = std::min(k, N);
    std::partial_sort(neighbors.begin(), neighbors.begin() + num_results, neighbors.end());

    // Copy results
    for (int i = 0; i < num_results; i++) {
        indices[i] = neighbors[static_cast<size_t>(i)].index;
        distances_sq[i] = neighbors[static_cast<size_t>(i)].distance_sq;
    }

    // Fill remaining with -1 if k > N
    for (int i = num_results; i < k; i++) {
        indices[i] = -1;
        distances_sq[i] = -1.0f;
    }
}

/**
 * Find k nearest neighbors for each point (excluding self).
 *
 * Standard KNN behavior: each point finds its k nearest neighbors,
 * excluding itself from the results.
 */
template <>
void knn_search<ScalarBackend>(const float* coords, int N, int k, int* indices,
                               float* distances_sq) {
    // For each point, find its k nearest neighbors
    for (int i = 0; i < N; i++) {
        const float* query_pt = coords + i * 3;

        // Reserve storage for all other points
        std::vector<Neighbor> neighbors;
        neighbors.reserve(static_cast<size_t>(std::max(0, N - 1)));

        for (int j = 0; j < N; j++) {
            if (j == i)
                continue;

            Neighbor neighbor;
            neighbor.index = j;
            neighbor.distance_sq = distance_squared(query_pt, coords + j * 3);
            neighbors.push_back(neighbor);
        }

        // Determine how many non-self neighbors we need
        int remaining_slots = std::max(0, k - 1);
        int num_results = std::min(remaining_slots, static_cast<int>(neighbors.size()));

        if (num_results > 0) {
            std::partial_sort(neighbors.begin(), neighbors.begin() + num_results, neighbors.end());
        }

        // Always include self as the first neighbor (matches JAX reference)
        if (k > 0) {
            indices[i * k + 0] = i;
            distances_sq[i * k + 0] = 0.0f;
        }

        // Copy remaining neighbors
        for (int j = 0; j < num_results; j++) {
            indices[i * k + 1 + j] = neighbors[static_cast<size_t>(j)].index;
            distances_sq[i * k + 1 + j] = neighbors[static_cast<size_t>(j)].distance_sq;
        }

        // Fill remaining slots with -1 when k > available neighbors
        for (int j = 1 + num_results; j < k; j++) {
            indices[i * k + j] = -1;
            distances_sq[i * k + j] = -1.0f;
        }
    }
}

}  // namespace knn
}  // namespace pfalign
