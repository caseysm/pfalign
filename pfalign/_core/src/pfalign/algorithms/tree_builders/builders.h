/**
 * Guide tree construction algorithms.
 *
 * Public API for building guide trees from distance matrices using various
 * hierarchical clustering methods.
 *
 * All methods produce binary trees with N leaves and N-1 internal nodes,
 * suitable for progressive multiple sequence alignment.
 */

#pragma once

#include "pfalign/types/guide_tree_types.h"
#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace tree_builders {

/**
 * Build guide tree from distance matrix using UPGMA.
 *
 * UPGMA (Unweighted Pair Group Method with Arithmetic Mean):
 * - Agglomerative hierarchical clustering
 * - Assumes ultrametric tree (molecular clock)
 * - Simplest and fastest method
 * - O(N^2) time complexity
 * - Produces binary tree (merges exactly 2 clusters at each step)
 *
 * @param distances Distance matrix [N * N] (symmetric, d[i][i]=0)
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return Binary guide tree with N leaves, N-1 internal nodes
 */
types::GuideTree build_upgma_tree(const float* distances, int N,
                                  pfalign::memory::GrowableArena* arena);

/**
 * Build guide tree from distance matrix using Neighbor Joining (NJ).
 *
 * NJ (Neighbor Joining):
 * - Does NOT assume molecular clock (non-ultrametric)
 * - More accurate than UPGMA for sequences with varying rates
 * - Uses Q-criterion to find best join
 * - O(N^2) time complexity (with optimizations)
 * - Produces binary tree
 *
 * @param distances Distance matrix [N * N]
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return Binary guide tree
 */
types::GuideTree build_nj_tree(const float* distances, int N,
                               pfalign::memory::GrowableArena* arena);

/**
 * Build guide tree from distance matrix using BIONJ.
 *
 * BIONJ (Variance-weighted Neighbor Joining):
 * - Improvement over NJ using variance weighting
 * - More accurate for noisy distance matrices
 * - Weights distance updates by inverse variance
 * - O(N^2) time complexity
 *
 * @param distances Distance matrix [N * N]
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return Binary guide tree
 */
types::GuideTree build_bionj_tree(const float* distances, int N,
                                  pfalign::memory::GrowableArena* arena);

/**
 * Build guide tree from distance matrix using Minimum Spanning Tree (MST).
 *
 * MST via Kruskal's algorithm + binary tree conversion:
 * - Builds minimum spanning tree from complete distance graph
 * - Converts to rooted binary tree via midpoint rooting + DFS
 * - Fast and simple: O(N^2 log N) time complexity
 * - Good for quick initial trees
 *
 * @param distances Distance matrix [N * N]
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return Rooted binary guide tree
 */
types::GuideTree build_mst_tree(const float* distances, int N,
                                pfalign::memory::GrowableArena* arena);

}  // namespace tree_builders
}  // namespace pfalign
