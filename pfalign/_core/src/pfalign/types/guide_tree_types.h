/**
 * Guide tree data structures for progressive MSA.
 *
 * A guide tree determines the order in which sequences/profiles are merged
 * during progressive multiple sequence alignment. The tree is binary, where:
 * - Leaves represent input sequences
 * - Internal nodes represent merged profiles
 * - Tree topology determines merge order
 *
 * This file contains pure data structures with no algorithm dependencies.
 * Tree construction algorithms are in algorithms/tree_builders/.
 */

#pragma once

#include "pfalign/common/growable_arena.h"
#include <vector>
#include <string>


namespace pfalign::types {

/**
 * Node in a binary guide tree.
 *
 * The tree is represented as a flat array of nodes, indexed by node_id.
 * Leaves have is_leaf=true and store seq_id (index into SequenceCache).
 * Internal nodes have is_leaf=false and store left_child/right_child indices.
 */
struct GuideTreeNode {
    int node_id;   ///< Index of this node in tree array [0, 2N-2]
    bool is_leaf;  ///< True if leaf (input sequence), false if internal (merged profile)

    // Leaf data
    int seq_id;  ///< Sequence index (if is_leaf=true)

    // Internal node data
    int left_child;   ///< Index of left child node (if is_leaf=false)
    int right_child;  ///< Index of right child node (if is_leaf=false)
    float distance;   ///< Branch length to parent (0.0 for root)

    // Cluster metadata
    int num_sequences;  ///< Number of sequences in this subtree
    float height;       ///< Height of this node (distance from furthest leaf)

    /**
     * Default constructor - creates invalid node.
     */
    GuideTreeNode()
        : node_id(-1),
          is_leaf(false),
          seq_id(-1),
          left_child(-1),
          right_child(-1),
          distance(0.0f),
          num_sequences(0),
          height(0.0f) {
    }

    /**
     * Create a leaf node.
     */
    static GuideTreeNode create_leaf(int node_id, int seq_id) {
        GuideTreeNode node;
        node.node_id = node_id;
        node.is_leaf = true;
        node.seq_id = seq_id;
        node.num_sequences = 1;
        node.height = 0.0f;
        return node;
    }

    /**
     * Create an internal node.
     */
    static GuideTreeNode create_internal(int node_id, int left_child, int right_child,
                                         float distance, int num_sequences, float height) {
        GuideTreeNode node;
        node.node_id = node_id;
        node.is_leaf = false;
        node.left_child = left_child;
        node.right_child = right_child;
        node.distance = distance;
        node.num_sequences = num_sequences;
        node.height = height;
        return node;
    }
};

/**
 * Binary guide tree for progressive MSA.
 *
 * Tree structure:
 * - N leaf nodes (input sequences, node IDs: 0 to N-1)
 * - N-1 internal nodes (merged profiles, node IDs: N to 2N-2)
 * - Total: 2N-1 nodes
 * - Root is always at index 2N-2 (last node)
 *
 * Memory management:
 * - Tree nodes are stored in arena-allocated array
 * - Tree does not own the arena
 * - Tree lifetime must not exceed arena lifetime
 *
 * Post-order traversal:
 * - Processes children before parent
 * - Ensures dependencies are resolved before merge
 * - Used during progressive alignment
 *
 * Construction:
 * - Use factory functions in algorithms/tree_builders/ to build trees
 * - This class only contains the data structure and traversal methods
 */
class GuideTree {
public:
    /**
     * Create empty tree (invalid state).
     */
    GuideTree() : nodes_(nullptr), N_(0), root_idx_(-1) {
    }

    /**
     * Create tree from pre-allocated node array.
     *
     * @param nodes     Array of 2N-1 nodes (arena-allocated)
     * @param N         Number of leaf nodes (input sequences)
     */
    GuideTree(GuideTreeNode* nodes, int N) : nodes_(nodes), N_(N), root_idx_(2 * N - 2) {
    }

    /**
     * Get number of input sequences (leaf nodes).
     */
    int num_sequences() const {
        return N_;
    }

    /**
     * Get total number of nodes (2N-1).
     */
    int num_nodes() const {
        return 2 * N_ - 1;
    }

    /**
     * Get root node index (always 2N-2).
     */
    int root_index() const {
        return root_idx_;
    }

    /**
     * Access node by index.
     *
     * @param idx Node index [0, 2N-2]
     * @return Reference to node
     */
    const GuideTreeNode& node(int idx) const {
        return nodes_[idx];
    }

    /**
     * Access mutable node by index (for construction).
     */
    GuideTreeNode& node_mut(int idx) {
        return nodes_[idx];
    }

    /**
     * Get direct pointer to nodes array.
     */
    const GuideTreeNode* nodes() const {
        return nodes_;
    }

    /**
     * Compute post-order traversal.
     *
     * Post-order visits children before parent, ensuring that when we process
     * an internal node, both children have already been processed.
     *
     * This is critical for progressive alignment: we must build child profiles
     * before we can merge them into the parent profile.
     *
     * @param arena Arena for allocating traversal array
     * @param out_order Output array [2N-1] of node indices in post-order
     *
     * Example:
     *       4
     *      / \
     *     2   3
     *    / \
     *   0   1
     *
     * Post-order: [0, 1, 2, 3, 4]
     * (Process leaves 0,1 -> merge to 2 -> process leaf 3 -> merge 2,3 to 4)
     */
    void compute_post_order(pfalign::memory::GrowableArena* arena, int* out_order) const;

    /**
     * Compute reverse level-order traversal (bottom-up, batched by depth).
     *
     * Groups nodes by tree depth/level, processing from leaves to root.
     * Nodes at the same level can be processed in parallel, enabling batching
     * of independent alignment operations.
     *
     * **Advantages over post-order:**
     * - Enables batch processing at each level (GPU/SIMD-friendly)
     * - All operations at level k can run in parallel
     * - Better cache locality within levels
     * - Natural for distributed MSA (map-reduce pattern)
     *
     * **Output format:**
     * - out_order: Flat array [2N-1] of node indices, grouped by level
     * - out_level_offsets: Start index of each level in out_order [depth+1]
     *
     * @param arena Arena for allocating traversal arrays
     * @param out_order Output array [2N-1] of node indices in reverse level-order
     * @param out_level_offsets Start indices for each level [depth+1]
     * @param out_depth Output: maximum tree depth
     *
     * Example:
     *           6
     *          / \
     *         4   5
     *        / \ / \
     *       0  1 2  3
     *
     * Reverse level-order: [0, 1, 2, 3, 4, 5, 6]
     * Levels: [[0,1,2,3], [4,5], [6]]
     * Level offsets: [0, 4, 6, 7]
     *
     * **Batched progressive alignment:**
     * ```cpp
     * for (int level = 0; level < depth; level++) {
     *     int start = level_offsets[level];
     *     int end = level_offsets[level + 1];
     *     int num_nodes = end - start;
     *
     *     // Batch process all nodes at this level in parallel
     *     #pragma omp parallel for
     *     for (int i = start; i < end; i++) {
     *         int node_idx = order[i];
     *         if (!tree.node(node_idx).is_leaf) {
     *             // Align left and right child profiles
     *             profiles[node_idx] = align_and_merge(
     *                 profiles[left_child],
     *                 profiles[right_child]
     *             );
     *         }
     *     }
     * }
     * ```
     *
     * **Performance implications:**
     * - Level k has <= 2^k nodes (binary tree)
     * - Total work: O(N^2) (same as post-order)
     * - Parallelism: Up to 2^k-way at level k
     * - Memory: Requires storing all profiles at current level (higher than sequential)
     *
     * **Trade-offs:**
     * - ✅ Better parallelism (batch GPU kernels)
     * - ✅ More predictable performance (uniform level sizes)
     * - ❌ Higher peak memory (all level k profiles in memory)
     * - ❌ Slightly more complex indexing
     */
    void compute_reverse_level_order(pfalign::memory::GrowableArena* arena, int* out_order,
                                     int* out_level_offsets, int* out_depth) const;

    /**
     * Export tree to Newick format (optional, for visualization).
     *
     * Newick format: ((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);
     *
     * @param seq_names Names of input sequences [N]
     * @param arena Arena for string allocation
     * @return Newick string (arena-allocated)
     */
    const char* to_newick(const char** seq_names, pfalign::memory::GrowableArena* arena) const;

private:
    GuideTreeNode* nodes_;  ///< Array of 2N-1 nodes (arena-allocated)
    int N_;                 ///< Number of leaf nodes (input sequences)
    int root_idx_;          ///< Index of root node (always 2N-2)
};

} // namespace pfalign::types

