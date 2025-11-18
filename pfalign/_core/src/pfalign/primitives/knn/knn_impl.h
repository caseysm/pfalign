#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace knn {

/**
 * K-Nearest Neighbors (KNN) search for MPNN encoder.
 *
 * Finds k nearest neighbors in 3D space for each query point.
 * Used in MPNN to identify neighboring residues for message passing.
 *
 * Implementation:
 * - Scalar: KD-tree via nanoflann (O(N log N) build, O(k log N) query)
 * - NEON: Same KD-tree (vectorized distance computation)
 * - CUDA: Brute force for small N (<10k), or GPU KD-tree for large N
 *
 * Default k: 30 neighbors (typical for protein MPNN)
 *
 * Reference:
 *   align/_jax_reference/mpnn.py::_dist()
 */

/**
 * Find k nearest neighbors for each point in a point cloud.
 *
 * For each point i, finds the k nearest points INCLUDING self.
 * This matches JAX approx_min_k behavior used in MPNN.
 * Returns both indices and squared distances.
 *
 * @param coords        Input coordinates [N * 3] (x, y, z) in row-major order
 * @param N             Number of points
 * @param k             Number of neighbors to find
 * @param indices       Output neighbor indices [N * k] (row-major)
 * @param distances_sq  Output squared distances [N * k] (row-major)
 *
 * Example:
 *   float coords[100 * 3];  // 100 Ca atoms
 *   int indices[100 * 30];
 *   float distances_sq[100 * 30];
 *   knn_search<ScalarBackend>(coords, 100, 30, indices, distances_sq);
 *   // indices[0] == 0 (self), indices[1..29] are true neighbors
 *
 * Notes:
 * - Self is INCLUDED as first neighbor (matches JAX approx_min_k)
 * - indices[i*k + 0] == i with distance 0 (self-edge for MPNN)
 * - If N < k, only N neighbors returned (rest filled with -1)
 * - Distances are SQUARED (sqrt not computed for efficiency)
 */
template <typename Backend>
void knn_search(const float* coords, int N, int k, int* indices, float* distances_sq);

/**
 * Find k nearest neighbors for a single query point.
 *
 * @param coords        Point cloud coordinates [N * 3]
 * @param N             Number of points in cloud
 * @param query_pt      Query point [3] (x, y, z)
 * @param k             Number of neighbors
 * @param indices       Output neighbor indices [k]
 * @param distances_sq  Output squared distances [k]
 *
 * Example:
 *   float query[3] = {10.0f, 5.0f, 8.0f};
 *   int indices[30];
 *   float distances_sq[30];
 *   knn_query<ScalarBackend>(coords, 100, query, 30, indices, distances_sq);
 */
template <typename Backend>
void knn_query(const float* coords, int N, const float* query_pt, int k, int* indices,
               float* distances_sq);

}  // namespace knn
}  // namespace pfalign
