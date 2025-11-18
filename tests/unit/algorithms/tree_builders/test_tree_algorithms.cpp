/**
 * Unit tests for guide tree construction algorithms.
 *
 * Tests UPGMA, NJ, BIONJ, and MST algorithms with various distance matrices.
 */

#include "pfalign/types/guide_tree_types.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace pfalign::types;
using namespace pfalign::tree_builders;
using namespace pfalign::memory;

/**
 * Verify basic tree properties.
 */
void verify_tree_structure(const GuideTree& tree, int N, const char* algorithm) {
    printf("  Verifying %s tree structure (N=%d)...\n", algorithm, N);

    // Check node count
    assert(tree.num_sequences() == N);
    assert(tree.num_nodes() == 2 * N - 1);
    assert(tree.root_index() == 2 * N - 2);

    // Verify all leaf nodes
    for (int i = 0; i < N; i++) {
        const auto& node = tree.node(i);
        assert(node.is_leaf);
        assert(node.seq_id == i);
        assert(node.num_sequences == 1);
    }

    // Verify all internal nodes are binary
    for (int i = N; i < 2 * N - 1; i++) {
        const auto& node = tree.node(i);
        assert(!node.is_leaf);
        assert(node.left_child >= 0 && node.left_child < i);
        assert(node.right_child >= 0 && node.right_child < i);
        assert(node.left_child != node.right_child);

        // Verify num_sequences = sum of children
        const auto& left = tree.node(node.left_child);
        const auto& right = tree.node(node.right_child);
        assert(node.num_sequences == left.num_sequences + right.num_sequences);
    }

    printf("  ✓ %s tree structure is valid\n", algorithm);
}

/**
 * Test with simple 3-sequence distance matrix.
 */
void test_small_distance_matrix() {
    printf("Testing small distance matrix (N=3)...\n");

    GrowableArena arena(1);
    int N = 3;

    // Distance matrix:
    //     0    1    2
    // 0 [ 0.0  0.3  0.5 ]
    // 1 [ 0.3  0.0  0.4 ]
    // 2 [ 0.5  0.4  0.0 ]
    //
    // Expected: 0 and 1 are closest (d=0.3), merge first
    float distances[9] = {
        0.0f, 0.3f, 0.5f,
        0.3f, 0.0f, 0.4f,
        0.5f, 0.4f, 0.0f
    };

    // Test UPGMA
    {
        float dist_copy[9];
        std::memcpy(dist_copy, distances, 9 * sizeof(float));
        GuideTree tree = build_upgma_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "UPGMA");
    }

    // Test NJ
    {
        float dist_copy[9];
        std::memcpy(dist_copy, distances, 9 * sizeof(float));
        GuideTree tree = build_nj_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "NJ");
    }

    // Test BIONJ
    {
        float dist_copy[9];
        std::memcpy(dist_copy, distances, 9 * sizeof(float));
        GuideTree tree = build_bionj_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "BIONJ");
    }

    // Test MST
    {
        float dist_copy[9];
        std::memcpy(dist_copy, distances, 9 * sizeof(float));
        GuideTree tree = build_mst_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "MST");
    }

    printf("✓ All algorithms handle N=3 correctly\n\n");
}

/**
 * Test with larger distance matrix (N=5).
 */
void test_medium_distance_matrix() {
    printf("Testing medium distance matrix (N=5)...\n");

    GrowableArena arena(1);
    int N = 5;

    // Simple distance matrix (symmetric)
    float distances[25] = {
        0.0f, 0.2f, 0.4f, 0.6f, 0.8f,
        0.2f, 0.0f, 0.3f, 0.5f, 0.7f,
        0.4f, 0.3f, 0.0f, 0.4f, 0.6f,
        0.6f, 0.5f, 0.4f, 0.0f, 0.5f,
        0.8f, 0.7f, 0.6f, 0.5f, 0.0f
    };

    // Test UPGMA
    {
        float dist_copy[25];
        std::memcpy(dist_copy, distances, 25 * sizeof(float));
        GuideTree tree = build_upgma_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "UPGMA");
    }

    // Test NJ
    {
        float dist_copy[25];
        std::memcpy(dist_copy, distances, 25 * sizeof(float));
        GuideTree tree = build_nj_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "NJ");
    }

    // Test BIONJ
    {
        float dist_copy[25];
        std::memcpy(dist_copy, distances, 25 * sizeof(float));
        GuideTree tree = build_bionj_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "BIONJ");
    }

    // Test MST
    {
        float dist_copy[25];
        std::memcpy(dist_copy, distances, 25 * sizeof(float));
        GuideTree tree = build_mst_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "MST");
    }

    printf("✓ All algorithms handle N=5 correctly\n\n");
}

/**
 * Test edge case: single sequence (N=1).
 */
void test_single_sequence() {
    printf("Testing single sequence (N=1)...\n");

    GrowableArena arena(1);
    int N = 1;
    float distances[1] = {0.0f};

    // Test UPGMA
    {
        GuideTree tree = build_upgma_tree(distances, N, &arena);
        assert(tree.num_sequences() == 1);
        assert(tree.num_nodes() == 1);
        assert(tree.root_index() == 0);
        assert(tree.node(0).is_leaf);
        printf("  ✓ UPGMA handles N=1\n");
    }

    // Test NJ
    {
        GuideTree tree = build_nj_tree(distances, N, &arena);
        assert(tree.num_sequences() == 1);
        assert(tree.num_nodes() == 1);
        printf("  ✓ NJ handles N=1\n");
    }

    // Test BIONJ
    {
        GuideTree tree = build_bionj_tree(distances, N, &arena);
        assert(tree.num_sequences() == 1);
        assert(tree.num_nodes() == 1);
        printf("  ✓ BIONJ handles N=1\n");
    }

    // Test MST
    {
        GuideTree tree = build_mst_tree(distances, N, &arena);
        assert(tree.num_sequences() == 1);
        assert(tree.num_nodes() == 1);
        printf("  ✓ MST handles N=1\n");
    }

    printf("✓ All algorithms handle N=1 correctly\n\n");
}

/**
 * Test edge case: two sequences (N=2).
 */
void test_two_sequences() {
    printf("Testing two sequences (N=2)...\n");

    GrowableArena arena(1);
    int N = 2;
    float distances[4] = {
        0.0f, 0.5f,
        0.5f, 0.0f
    };

    // Test UPGMA
    {
        float dist_copy[4];
        std::memcpy(dist_copy, distances, 4 * sizeof(float));
        GuideTree tree = build_upgma_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "UPGMA");
    }

    // Test NJ
    {
        float dist_copy[4];
        std::memcpy(dist_copy, distances, 4 * sizeof(float));
        GuideTree tree = build_nj_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "NJ");
    }

    // Test BIONJ
    {
        float dist_copy[4];
        std::memcpy(dist_copy, distances, 4 * sizeof(float));
        GuideTree tree = build_bionj_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "BIONJ");
    }

    // Test MST
    {
        float dist_copy[4];
        std::memcpy(dist_copy, distances, 4 * sizeof(float));
        GuideTree tree = build_mst_tree(dist_copy, N, &arena);
        verify_tree_structure(tree, N, "MST");
    }

    printf("✓ All algorithms handle N=2 correctly\n\n");
}

/**
 * Test post-order traversal for progressive alignment.
 */
void test_post_order_traversal() {
    printf("Testing post-order traversal...\n");

    GrowableArena arena(1);
    int N = 4;

    float distances[16] = {
        0.0f, 0.2f, 0.4f, 0.6f,
        0.2f, 0.0f, 0.3f, 0.5f,
        0.4f, 0.3f, 0.0f, 0.4f,
        0.6f, 0.5f, 0.4f, 0.0f
    };

    // Build tree with UPGMA
    GuideTree tree = build_upgma_tree(distances, N, &arena);

    // Compute post-order
    int* order = arena.allocate<int>(tree.num_nodes());
    tree.compute_post_order(&arena, order);

    // Verify post-order property: children come before parent
    bool* visited = arena.allocate<bool>(tree.num_nodes());
    for (int i = 0; i < tree.num_nodes(); i++) visited[i] = false;

    for (int i = 0; i < tree.num_nodes(); i++) {
        int node_id = order[i];
        const auto& node = tree.node(node_id);

        if (!node.is_leaf) {
            // Internal node: children must have been visited
            assert(visited[node.left_child]);
            assert(visited[node.right_child]);
        }

        visited[node_id] = true;
    }

    // Root should be last
    assert(order[tree.num_nodes() - 1] == tree.root_index());

    printf("  ✓ Post-order traversal is correct\n");
    printf("✓ Post-order traversal test passed\n\n");
}

/**
 * Test reverse level-order traversal for batched progressive alignment.
 */
void test_reverse_level_order_traversal() {
    printf("Testing reverse level-order traversal...\n");

    GrowableArena arena(1);
    int N = 4;

    float distances[16] = {
        0.0f, 0.2f, 0.4f, 0.6f,
        0.2f, 0.0f, 0.3f, 0.5f,
        0.4f, 0.3f, 0.0f, 0.4f,
        0.6f, 0.5f, 0.4f, 0.0f
    };

    // Build tree with UPGMA
    GuideTree tree = build_upgma_tree(distances, N, &arena);

    // Compute reverse level-order
    int* order = arena.allocate<int>(tree.num_nodes());
    int* level_offsets = arena.allocate<int>(tree.num_nodes() + 1);
    int depth;
    tree.compute_reverse_level_order(&arena, order, level_offsets, &depth);

    printf("  Tree depth: %d levels\n", depth);

    // Verify level-order property: nodes at same level are grouped
    for (int level = 0; level < depth; level++) {
        int start = level_offsets[level];
        int end = level_offsets[level + 1];
        int num_nodes = end - start;

        printf("  Level %d: %d nodes [", level, num_nodes);
        for (int i = start; i < end; i++) {
            printf("%d", order[i]);
            if (i < end - 1) printf(", ");
        }
        printf("]\n");

        // All nodes at this level should be processable
        // (i.e., their children have already been processed)
        for (int i = start; i < end; i++) {
            int node_id = order[i];
            const auto& node = tree.node(node_id);

            if (!node.is_leaf) {
                // Internal node: children must appear earlier in order
                bool left_found = false;
                bool right_found = false;

                for (int j = 0; j < i; j++) {
                    if (order[j] == node.left_child) left_found = true;
                    if (order[j] == node.right_child) right_found = true;
                }

                assert(left_found && "Left child must appear before parent");
                assert(right_found && "Right child must appear before parent");
            }
        }
    }

    // Root should be in last level
    int last_level_start = level_offsets[depth - 1];
    int last_level_end = level_offsets[depth];
    bool root_in_last_level = false;
    for (int i = last_level_start; i < last_level_end; i++) {
        if (order[i] == tree.root_index()) {
            root_in_last_level = true;
            break;
        }
    }
    assert(root_in_last_level && "Root must be in last level");

    // All leaves should be in first level
    int first_level_end = level_offsets[1];
    for (int i = 0; i < first_level_end; i++) {
        const auto& node = tree.node(order[i]);
        assert(node.is_leaf && "First level should contain only leaves");
    }

    printf("  ✓ Reverse level-order traversal is correct\n");
    printf("  ✓ All nodes properly batched by level\n");
    printf("  ✓ Children processed before parents\n");
    printf("✓ Reverse level-order traversal test passed\n\n");
}

/**
 * Test Newick export.
 */
void test_newick_export() {
    printf("Testing Newick export...\n");

    GrowableArena arena(1);
    int N = 3;

    float distances[9] = {
        0.0f, 0.3f, 0.5f,
        0.3f, 0.0f, 0.4f,
        0.5f, 0.4f, 0.0f
    };

    const char* seq_names[3] = {"Seq0", "Seq1", "Seq2"};

    // Build tree with UPGMA
    GuideTree tree = build_upgma_tree(distances, N, &arena);

    // Export to Newick
    const char* newick = tree.to_newick(seq_names, &arena);

    printf("  Newick: %s\n", newick);

    // Verify format
    assert(newick[0] == '(');  // Starts with (
    assert(newick[std::strlen(newick) - 1] == ';');  // Ends with ;

    // Check all sequence names appear
    assert(std::strstr(newick, "Seq0") != nullptr);
    assert(std::strstr(newick, "Seq1") != nullptr);
    assert(std::strstr(newick, "Seq2") != nullptr);

    printf("  ✓ Newick export is correct\n");
    printf("✓ Newick export test passed\n\n");
}

int main() {
    printf("=== Guide Tree Algorithm Tests ===\n\n");

    test_single_sequence();
    test_two_sequences();
    test_small_distance_matrix();
    test_medium_distance_matrix();
    test_post_order_traversal();
    test_reverse_level_order_traversal();
    test_newick_export();

    printf("=== All tests passed! ===\n");
    return 0;
}
