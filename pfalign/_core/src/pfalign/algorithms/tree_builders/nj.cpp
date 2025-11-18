/**
 * Neighbor Joining guide tree construction.
 *
 * NJ (Neighbor Joining) is a bottom-up clustering method that does NOT
 * assume a molecular clock. It uses a Q-criterion to find the best pair
 * to join at each step.
 *
 * Algorithm:
 * 1. Start with N sequences (star tree)
 * 2. Repeat N-2 times:
 *    a. Compute Q-matrix: Q(i,j) = d(i,j) - (r_i + r_j)
 *       where r_i = Sigma_k d(i,k) / (n-2)
 *    b. Find pair (i,j) with minimum Q(i,j)
 *    c. Create new node k by joining i and j (binary node)
 *    d. Compute branch lengths using NJ formula
 *    e. Update distances: d(k,m) = (d(i,m) + d(j,m) - d(i,j)) / 2
 * 3. Connect last two nodes
 *
 * Optimization: Maintain row sums S[i] = Sigma_k d(i,k) incrementally.
 * This makes r_i computation O(1) instead of O(N), reducing overall
 * complexity from O(N^3) to O(N^2).
 *
 * Result: Binary tree with N leaves, N-1 internal nodes
 *
 * Complexity: O(N^2) time, O(N^2) space
 */

#include "builders.h"
#include "pfalign/types/guide_tree_types.h"
#include "pfalign/common/arena_allocator.h"
#include <cstring>
#include <cfloat>
#include <cstdio>
#include <cmath>

namespace pfalign {
namespace tree_builders {

using pfalign::types::GuideTree;
using pfalign::types::GuideTreeNode;

/**
 * Build Neighbor Joining guide tree from distance matrix.
 *
 * @param distances Distance matrix [N * N] (will be modified in-place)
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return          Binary guide tree
 */
GuideTree build_nj_tree(const float* distances, int N, pfalign::memory::GrowableArena* arena) {
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

    if (N == 2) {
        // Two sequences: one internal node connecting them
        GuideTreeNode* nodes = arena->allocate<GuideTreeNode>(3);
        // Construct all nodes
        for (int i = 0; i < 3; i++) {
            new (&nodes[i]) GuideTreeNode();
        }
        nodes[0] = GuideTreeNode::create_leaf(0, 0);
        nodes[1] = GuideTreeNode::create_leaf(1, 1);

        float dist = distances[0 * N + 1];
        nodes[2] = GuideTreeNode::create_internal(2, 0, 1, 0.0f, 2, dist / 2.0f);
        nodes[0].distance = dist / 2.0f;
        nodes[1].distance = dist / 2.0f;

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
    bool* active = arena->allocate<bool>(N);         // Is cluster still active?
    int* cluster_to_node = arena->allocate<int>(N);  // Cluster ID -> node ID
    int* cluster_size = arena->allocate<int>(N);     // Number of sequences in cluster

    // Initialize cluster state
    for (int i = 0; i < N; i++) {
        active[i] = true;
        cluster_to_node[i] = i;  // Initially, cluster i = leaf node i
        cluster_size[i] = 1;
    }

    int n_active = N;      // Number of active clusters
    int next_node_id = N;  // Next available internal node ID

    // O(N^2) optimization: maintain row sums S[i] = Sigma_k d(i,k)
    // This allows O(1) computation of r_i = S[i] / (n-2)
    float* row_sum = arena->allocate<float>(N);
    for (int i = 0; i < N; i++) {
        row_sum[i] = 0.0f;
        for (int k = 0; k < N; k++) {
            row_sum[i] += dist[i * N + k];
        }
    }

    // Perform N-2 joins (leaving 2 clusters at the end)
    while (n_active > 2) {
        // Compute r_i = row_sum[i] / (n-2) for each active cluster (O(1) per cluster)
        float* r = arena->allocate<float>(N);
        for (int i = 0; i < N; i++) {
            if (!active[i]) {
                r[i] = 0.0f;
            } else {
                r[i] = row_sum[i] / (n_active - 2);
            }
        }

        // Find pair (i, j) with minimum Q(i,j) = d(i,j) - (r_i + r_j)
        int min_i = -1, min_j = -1;
        float min_Q = FLT_MAX;

        for (int i = 0; i < N; i++) {
            if (!active[i])
                continue;
            for (int j = i + 1; j < N; j++) {
                if (!active[j])
                    continue;
                float Q = dist[i * N + j] - (r[i] + r[j]);
                if (Q < min_Q) {
                    min_Q = Q;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        if (min_i == -1 || min_j == -1) {
            std::fprintf(stderr, "NJ: No valid pair found\n");
            break;
        }

        // Create new internal node k
        int k = next_node_id++;
        int node_i = cluster_to_node[min_i];
        int node_j = cluster_to_node[min_j];

        float d_ij = dist[min_i * N + min_j];

        // NJ branch lengths:
        // b_i = d(i,j)/2 + (r_i - r_j)/2
        // b_j = d(i,j)/2 + (r_j - r_i)/2 = d(i,j) - b_i
        float branch_i = d_ij / 2.0f + (r[min_i] - r[min_j]) / 2.0f;
        float branch_j = d_ij - branch_i;

        // Clamp to non-negative (NJ can produce negative branches)
        if (branch_i < 0.0f)
            branch_i = 0.0f;
        if (branch_j < 0.0f)
            branch_j = 0.0f;

        // Create internal node
        float height = std::max(nodes[node_i].height + branch_i, nodes[node_j].height + branch_j);

        nodes[k] = GuideTreeNode::create_internal(
            k, node_i, node_j,
            0.0f,  // Distance to parent (will be set when this becomes a child)
            cluster_size[min_i] + cluster_size[min_j], height);

        // Update branch lengths of children
        nodes[node_i].distance = branch_i;
        nodes[node_j].distance = branch_j;

        // Update cluster state for new cluster (reuse min_i slot)
        cluster_to_node[min_i] = k;
        cluster_size[min_i] = cluster_size[min_i] + cluster_size[min_j];

        // Deactivate cluster j (merged into i)
        active[min_j] = false;

        // Update distances: d(k, m) = (d(i, m) + d(j, m) - d(i,j)) / 2
        // Also update row sums incrementally (O(N) per merge)
        float new_row_sum = 0.0f;
        for (int m = 0; m < N; m++) {
            if (!active[m] || m == min_i)
                continue;

            float d_im = dist[min_i * N + m];
            float d_jm = dist[min_j * N + m];
            float d_km = (d_im + d_jm - d_ij) / 2.0f;

            // Ensure non-negative
            if (d_km < 0.0f)
                d_km = 0.0f;

            // Store in min_i row/column (symmetric)
            dist[min_i * N + m] = d_km;
            dist[m * N + min_i] = d_km;

            // Update row sum for cluster m (subtract old distances, add new)
            row_sum[m] -= d_im + d_jm;
            row_sum[m] += d_km;

            // Accumulate row sum for new cluster k
            new_row_sum += d_km;
        }

        // Update row sum for new cluster k (reusing min_i slot)
        row_sum[min_i] = new_row_sum;
        row_sum[min_j] = 0.0f;  // min_j is no longer active

        n_active--;
    }

    // Connect last two active clusters as root
    int last_i = -1, last_j = -1;
    for (int i = 0; i < N; i++) {
        if (active[i]) {
            if (last_i == -1) {
                last_i = i;
            } else {
                last_j = i;
                break;
            }
        }
    }

    if (last_i == -1 || last_j == -1) {
        std::fprintf(stderr, "NJ: Failed to find last two clusters\n");
        return GuideTree(nodes, N);
    }

    // Create root node connecting last two clusters
    int root = next_node_id;
    int node_i = cluster_to_node[last_i];
    int node_j = cluster_to_node[last_j];

    float d_ij = dist[last_i * N + last_j];
    float branch_i = d_ij / 2.0f;
    float branch_j = d_ij / 2.0f;

    float height = std::max(nodes[node_i].height + branch_i, nodes[node_j].height + branch_j);

    nodes[root] =
        GuideTreeNode::create_internal(root, node_i, node_j,
                                       0.0f,  // Root has no parent
                                       cluster_size[last_i] + cluster_size[last_j], height);

    nodes[node_i].distance = branch_i;
    nodes[node_j].distance = branch_j;

    return GuideTree(nodes, N);
}

}  // namespace tree_builders
}  // namespace pfalign
