/**
 * UPGMA guide tree construction.
 *
 * UPGMA (Unweighted Pair Group Method with Arithmetic Mean) is a simple
 * agglomerative hierarchical clustering algorithm that assumes a molecular
 * clock (ultrametric tree).
 *
 * Algorithm:
 * 1. Start with N singleton clusters (one per sequence)
 * 2. Repeat N-1 times:
 *    a. Find pair (i,j) with minimum distance (O(N) with nearest-neighbor tracking)
 *    b. Create new cluster k by merging i and j (binary node)
 *    c. Compute distances: d(k,m) = (d(i,m) + d(j,m)) / 2
 *    d. Set branch lengths: b(i) = b(j) = d(i,j) / 2
 *    e. Update nearest neighbors for affected clusters
 * 3. Result: binary tree with N leaves, N-1 internal nodes
 *
 * Optimization: Maintain nearest[i] = closest cluster to i
 * After merging, only update nearest[m] if it was i or j, or if new cluster k is closer.
 * This reduces complexity from O(N^3) to O(N^2).
 *
 * Complexity: O(N^2) time, O(N^2) space
 */

#include "builders.h"
#include "pfalign/types/guide_tree_types.h"
#include "pfalign/common/arena_allocator.h"
#include <cstring>
#include <cfloat>
#include <cstdio>

namespace pfalign {
namespace tree_builders {

using pfalign::types::GuideTree;
using pfalign::types::GuideTreeNode;

/**
 * Build UPGMA guide tree from distance matrix.
 *
 * @param distances Distance matrix [N * N] (will be modified in-place)
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return          Binary guide tree
 */
GuideTree build_upgma_tree(const float* distances, int N, pfalign::memory::GrowableArena* arena) {
    if (N == 0) {
        return GuideTree(nullptr, 0);
    }

    if (N == 1) {
        // Single sequence: just a leaf node
        GuideTreeNode* nodes = arena->allocate<GuideTreeNode>(1);
        new (nodes) GuideTreeNode();
        nodes[0] = GuideTreeNode::create_leaf(0, 0);
        return GuideTree(nodes, N);
    }

    // Allocate tree nodes: N leaves + (N-1) internal nodes = 2N-1 total
    GuideTreeNode* nodes = arena->allocate<GuideTreeNode>(2 * N - 1);

    // Construct all nodes
    for (int i = 0; i < 2 * N - 1; i++) {
        new (&nodes[i]) GuideTreeNode();
    }

    // Initialize leaf nodes (0 to N-1)
    for (int i = 0; i < N; i++) {
        nodes[i] = GuideTreeNode::create_leaf(i, i);
    }

    // Working distance matrix (copy so we can modify)
    float* dist = arena->allocate<float>(N * N);
    std::memcpy(dist, distances, N * N * sizeof(float));

    // Cluster state
    bool* active = arena->allocate<bool>(N);            // Is cluster still active?
    int* cluster_to_node = arena->allocate<int>(N);     // Cluster ID -> node ID
    float* cluster_height = arena->allocate<float>(N);  // Height of each cluster
    int* cluster_size = arena->allocate<int>(N);        // Number of sequences in cluster

    // O(N^2) optimization: track nearest neighbor for each cluster
    int* nearest = arena->allocate<int>(N);           // nearest[i] = closest cluster to i
    float* nearest_dist = arena->allocate<float>(N);  // distance to nearest neighbor

    // Initialize cluster state
    for (int i = 0; i < N; i++) {
        active[i] = true;
        cluster_to_node[i] = i;  // Initially, cluster i = leaf node i
        cluster_height[i] = 0.0f;
        cluster_size[i] = 1;
    }

    // Initialize nearest neighbor for each cluster
    for (int i = 0; i < N; i++) {
        nearest[i] = -1;
        nearest_dist[i] = FLT_MAX;
        for (int j = 0; j < N; j++) {
            if (i == j)
                continue;
            float d = dist[i * N + j];
            if (d < nearest_dist[i]) {
                nearest_dist[i] = d;
                nearest[i] = j;
            }
        }
    }

    int next_node_id = N;  // Next available internal node ID

    // Perform N-1 merges
    for (int merge = 0; merge < N - 1; merge++) {
        // Find pair (i, j) with minimum distance using nearest neighbors
        // This is O(N) instead of O(N^2)
        int min_i = -1, min_j = -1;
        float min_dist = FLT_MAX;

        for (int i = 0; i < N; i++) {
            if (!active[i])
                continue;

            // Check if nearest neighbor is still valid
            if (nearest[i] != -1 && active[nearest[i]]) {
                if (nearest_dist[i] < min_dist) {
                    min_dist = nearest_dist[i];
                    min_i = i;
                    min_j = nearest[i];
                }
            }
        }

        if (min_i == -1 || min_j == -1) {
            // Should never happen with valid distance matrix
            std::fprintf(stderr, "UPGMA: No valid pair found at merge %d\n", merge);
            break;
        }

        // Create new internal node k
        int k = next_node_id++;
        int node_i = cluster_to_node[min_i];
        int node_j = cluster_to_node[min_j];

        // UPGMA: new cluster height is half the distance
        float new_height = min_dist / 2.0f;

        // Branch lengths: distance from cluster to new node
        float branch_i = new_height - cluster_height[min_i];
        float branch_j = new_height - cluster_height[min_j];

        // Create internal node
        nodes[k] = GuideTreeNode::create_internal(
            k, node_i, node_j,
            0.0f,  // Distance to parent (will be set when this becomes a child)
            cluster_size[min_i] + cluster_size[min_j], new_height);

        // Update branch lengths of children
        nodes[node_i].distance = branch_i;
        nodes[node_j].distance = branch_j;

        // Update cluster state for new cluster (reuse min_i slot)
        cluster_to_node[min_i] = k;
        cluster_height[min_i] = new_height;
        cluster_size[min_i] = cluster_size[min_i] + cluster_size[min_j];

        // Deactivate cluster j (merged into i)
        active[min_j] = false;

        // Update distances: d(k, m) = (d(i, m) + d(j, m)) / 2
        for (int m = 0; m < N; m++) {
            if (!active[m] || m == min_i)
                continue;

            float d_im = dist[min_i * N + m];
            float d_jm = dist[min_j * N + m];
            float d_km = (d_im + d_jm) / 2.0f;

            // Store in min_i row/column (symmetric)
            dist[min_i * N + m] = d_km;
            dist[m * N + min_i] = d_km;
        }

        // Update nearest neighbor information (O(N) per merge)
        // 1. Invalidate min_j's nearest neighbor (it's no longer active)
        nearest[min_j] = -1;
        nearest_dist[min_j] = FLT_MAX;

        // 2. Update nearest neighbor for new cluster k (reusing min_i slot)
        nearest[min_i] = -1;
        nearest_dist[min_i] = FLT_MAX;
        for (int m = 0; m < N; m++) {
            if (!active[m] || m == min_i)
                continue;
            float d = dist[min_i * N + m];
            if (d < nearest_dist[min_i]) {
                nearest_dist[min_i] = d;
                nearest[min_i] = m;
            }
        }

        // 3. Update nearest neighbor for any cluster m where nearest[m] was i or j
        //    or where the new cluster k is closer
        for (int m = 0; m < N; m++) {
            if (!active[m] || m == min_i)
                continue;

            // If m's nearest was i or j, need to recompute
            if (nearest[m] == min_i || nearest[m] == min_j) {
                nearest[m] = -1;
                nearest_dist[m] = FLT_MAX;
                for (int p = 0; p < N; p++) {
                    if (!active[p] || p == m)
                        continue;
                    float d = dist[m * N + p];
                    if (d < nearest_dist[m]) {
                        nearest_dist[m] = d;
                        nearest[m] = p;
                    }
                }
            } else {
                // Check if new cluster k is closer than current nearest
                float d_km = dist[m * N + min_i];
                if (d_km < nearest_dist[m]) {
                    nearest_dist[m] = d_km;
                    nearest[m] = min_i;
                }
            }
        }
    }

    return GuideTree(nodes, N);
}

}  // namespace tree_builders
}  // namespace pfalign
