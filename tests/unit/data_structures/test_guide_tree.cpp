/**
 * Unit tests for GuideTree data structure.
 */

#include "pfalign/types/guide_tree_types.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <cassert>
#include <cstdio>
#include <cstring>

using namespace pfalign::types;
using namespace pfalign::memory;

void test_guide_tree_construction() {
    printf("Testing guide tree construction...\n");

    GrowableArena arena(1);
    int N = 3;

    // Manually construct a simple tree:
    //       4
    //      / \.
    //     2   3
    //    / \.
    //   0   1
    //
    // Leaf nodes: 0, 1, 2 (sequence IDs)
    // Internal nodes: 3, 4

    GuideTreeNode* nodes = arena.allocate<GuideTreeNode>(2 * N - 1);

    // Construct all nodes (2*3-1 = 5 nodes)
    for (int i = 0; i < 2 * N - 1; i++) {
        new (&nodes[i]) GuideTreeNode();
    }

    // Leaf 0
    nodes[0] = GuideTreeNode::create_leaf(0, 0);

    // Leaf 1
    nodes[1] = GuideTreeNode::create_leaf(1, 1);

    // Leaf 2
    nodes[2] = GuideTreeNode::create_leaf(2, 2);

    // Internal node 3: merge 0 and 1
    nodes[3] = GuideTreeNode::create_internal(3, 0, 1, 0.1f, 2, 0.1f);

    // Root node 4: merge 3 and 2
    nodes[4] = GuideTreeNode::create_internal(4, 3, 2, 0.2f, 3, 0.2f);

    GuideTree tree(nodes, N);

    // Verify structure
    assert(tree.num_sequences() == 3);
    assert(tree.num_nodes() == 5);
    assert(tree.root_index() == 4);

    // Verify nodes
    assert(tree.node(0).is_leaf);
    assert(tree.node(0).seq_id == 0);

    assert(tree.node(1).is_leaf);
    assert(tree.node(1).seq_id == 1);

    assert(tree.node(2).is_leaf);
    assert(tree.node(2).seq_id == 2);

    assert(!tree.node(3).is_leaf);
    assert(tree.node(3).left_child == 0);
    assert(tree.node(3).right_child == 1);

    assert(!tree.node(4).is_leaf);
    assert(tree.node(4).left_child == 3);
    assert(tree.node(4).right_child == 2);

    printf("  ✓ Tree structure is correct\n");
}

void test_post_order_traversal() {
    printf("Testing post-order traversal...\n");

    GrowableArena arena(1);
    int N = 4;

    // Construct tree:
    //         6
    //        / \.
    //       4   5
    //      / \ / \.
    //     0  1 2  3
    //
    // Expected post-order: [0, 1, 4, 2, 3, 5, 6]

    GuideTreeNode* nodes = arena.allocate<GuideTreeNode>(2 * N - 1);

    // Construct all nodes (2*4-1 = 7 nodes)
    for (int i = 0; i < 2 * N - 1; i++) {
        new (&nodes[i]) GuideTreeNode();
    }

    // Leaves
    nodes[0] = GuideTreeNode::create_leaf(0, 0);
    nodes[1] = GuideTreeNode::create_leaf(1, 1);
    nodes[2] = GuideTreeNode::create_leaf(2, 2);
    nodes[3] = GuideTreeNode::create_leaf(3, 3);

    // Internal nodes
    nodes[4] = GuideTreeNode::create_internal(4, 0, 1, 0.1f, 2, 0.1f);
    nodes[5] = GuideTreeNode::create_internal(5, 2, 3, 0.1f, 2, 0.1f);
    nodes[6] = GuideTreeNode::create_internal(6, 4, 5, 0.2f, 4, 0.2f);

    GuideTree tree(nodes, N);

    // Compute post-order
    int* order = arena.allocate<int>(tree.num_nodes());
    tree.compute_post_order(&arena, order);

    // Verify post-order
    // Children must come before parents
    assert(order[0] == 0);
    assert(order[1] == 1);
    assert(order[2] == 4);  // Merge of 0,1
    assert(order[3] == 2);
    assert(order[4] == 3);
    assert(order[5] == 5);  // Merge of 2,3
    assert(order[6] == 6);  // Root (merge of 4,5)

    printf("  ✓ Post-order traversal is correct\n");
}

void test_newick_export() {
    printf("Testing Newick export...\n");

    GrowableArena arena(1);
    int N = 3;

    // Construct simple tree:
    //       4
    //      / \.
    //     2   3
    //    / \.
    //   0   1

    GuideTreeNode* nodes = arena.allocate<GuideTreeNode>(2 * N - 1);

    // Construct all nodes (2*3-1 = 5 nodes)
    for (int i = 0; i < 2 * N - 1; i++) {
        new (&nodes[i]) GuideTreeNode();
    }

    nodes[0] = GuideTreeNode::create_leaf(0, 0);
    nodes[1] = GuideTreeNode::create_leaf(1, 1);
    nodes[2] = GuideTreeNode::create_leaf(2, 2);
    nodes[3] = GuideTreeNode::create_internal(3, 0, 1, 0.1f, 2, 0.1f);
    nodes[4] = GuideTreeNode::create_internal(4, 3, 2, 0.2f, 3, 0.2f);

    GuideTree tree(nodes, N);

    // Sequence names
    const char* seq_names[3] = {"SeqA", "SeqB", "SeqC"};

    // Export to Newick
    const char* newick = tree.to_newick(seq_names, &arena);

    // Verify format (approximate - exact format depends on distance precision)
    printf("  Newick: %s\n", newick);

    // Check that all sequence names appear
    assert(std::strstr(newick, "SeqA") != nullptr);
    assert(std::strstr(newick, "SeqB") != nullptr);
    assert(std::strstr(newick, "SeqC") != nullptr);

    // Check delimiters
    assert(newick[0] == '(');  // Starts with (
    assert(newick[std::strlen(newick) - 1] == ';');  // Ends with ;

    printf("  ✓ Newick export is correct\n");
}

void test_single_sequence_tree() {
    printf("Testing single-sequence tree (edge case)...\n");

    GrowableArena arena(1);
    int N = 1;

    // Tree with single leaf (node 0, also root)
    GuideTreeNode* nodes = arena.allocate<GuideTreeNode>(1);
    new (nodes) GuideTreeNode();
    nodes[0] = GuideTreeNode::create_leaf(0, 0);

    // For N=1, root_idx should be 2*1-2 = 0
    GuideTree tree(nodes, N);

    assert(tree.num_sequences() == 1);
    assert(tree.num_nodes() == 1);
    assert(tree.root_index() == 0);

    // Post-order should just be [0]
    int order[1];
    tree.compute_post_order(&arena, order);
    assert(order[0] == 0);

    printf("  ✓ Single-sequence tree works\n");
}

void test_two_sequence_tree() {
    printf("Testing two-sequence tree...\n");

    GrowableArena arena(1);
    int N = 2;

    // Tree: 2
    //      / \.
    //     0   1

    GuideTreeNode* nodes = arena.allocate<GuideTreeNode>(3);
    // Construct all nodes
    for (int i = 0; i < 3; i++) {
        new (&nodes[i]) GuideTreeNode();
    }
    nodes[0] = GuideTreeNode::create_leaf(0, 0);
    nodes[1] = GuideTreeNode::create_leaf(1, 1);
    nodes[2] = GuideTreeNode::create_internal(2, 0, 1, 0.1f, 2, 0.1f);

    GuideTree tree(nodes, N);

    assert(tree.num_sequences() == 2);
    assert(tree.num_nodes() == 3);
    assert(tree.root_index() == 2);

    // Post-order: [0, 1, 2]
    int order[3];
    tree.compute_post_order(&arena, order);
    assert(order[0] == 0);
    assert(order[1] == 1);
    assert(order[2] == 2);

    printf("  ✓ Two-sequence tree works\n");
}

int main() {
    printf("=== GuideTree Tests ===\n\n");

    test_guide_tree_construction();
    test_post_order_traversal();
    test_newick_export();
    test_single_sequence_tree();
    test_two_sequence_tree();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
