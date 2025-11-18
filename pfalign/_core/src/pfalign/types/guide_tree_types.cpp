/**
 * Guide tree implementation.
 */

#include "guide_tree_types.h"
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace pfalign {
namespace types {

void GuideTree::compute_post_order(pfalign::memory::GrowableArena* arena, int* out_order) const {
    if (N_ == 0)
        return;

    // Use arena for temporary stack
    int* stack = arena->allocate<int>(static_cast<size_t>(num_nodes()));
    int stack_size = 0;

    // Push root
    stack[stack_size++] = root_idx_;

    int order_idx = 0;

    // Depth-first traversal with explicit stack
    // We use a "visited" flag by marking processed nodes with negative indices
    int* visited = arena->allocate<int>(static_cast<size_t>(num_nodes()));
    for (int i = 0; i < num_nodes(); i++) {
        visited[i] = 0;
    }

    while (stack_size > 0) {
        int node_idx = stack[--stack_size];
        const GuideTreeNode& node = nodes_[node_idx];

        if (visited[node_idx]) {
            // Already processed this node, add to output
            out_order[order_idx++] = node_idx;
            continue;
        }

        if (node.is_leaf) {
            // Leaf: add directly to output
            out_order[order_idx++] = node_idx;
            visited[node_idx] = 1;
        } else {
            // Internal node: push self, then right child, then left child
            // (Stack is LIFO, so we want to process left before right)
            stack[stack_size++] = node_idx;
            visited[node_idx] = 1;

            // Push children (right first, then left, so left is processed first)
            stack[stack_size++] = node.right_child;
            stack[stack_size++] = node.left_child;
        }
    }
}

void GuideTree::compute_reverse_level_order(pfalign::memory::GrowableArena* arena, int* out_order,
                                            int* out_level_offsets, int* out_depth) const {
    if (N_ == 0) {
        *out_depth = 0;
        return;
    }

    // Step 1: Compute depth of each node via BFS from root
    int* depths = arena->allocate<int>(static_cast<size_t>(num_nodes()));
    for (int i = 0; i < num_nodes(); i++) {
        depths[i] = -1;
    }

    // BFS queue (using arena-allocated array)
    int* queue = arena->allocate<int>(static_cast<size_t>(num_nodes()));
    int queue_front = 0;
    int queue_back = 0;

    // Start from root
    queue[queue_back++] = root_idx_;
    depths[root_idx_] = 0;
    int max_depth = 0;

    while (queue_front < queue_back) {
        int node_idx = queue[queue_front++];
        const GuideTreeNode& node = nodes_[node_idx];
        int depth = depths[node_idx];

        if (depth > max_depth) {
            max_depth = depth;
        }

        if (!node.is_leaf) {
            // Add children to queue
            depths[node.left_child] = depth + 1;
            depths[node.right_child] = depth + 1;
            queue[queue_back++] = node.left_child;
            queue[queue_back++] = node.right_child;
        }
    }

    *out_depth = max_depth + 1;  // Number of levels (0-indexed depths)

    // Step 2: Count nodes at each level
    int* level_counts = arena->allocate<int>(static_cast<size_t>(max_depth + 1));
    for (int i = 0; i <= max_depth; i++) {
        level_counts[i] = 0;
    }

    for (int i = 0; i < num_nodes(); i++) {
        if (depths[i] < 0)
            continue;
        level_counts[depths[i]]++;
    }

    // Step 3: Compute level offsets in reverse order (leaves first)
    // Level 0 in output = max_depth nodes (leaves)
    // Level k in output = (max_depth - k) nodes
    out_level_offsets[0] = 0;
    for (int rev_level = 0; rev_level <= max_depth; rev_level++) {
        int tree_depth = max_depth - rev_level;  // Reverse mapping
        out_level_offsets[rev_level + 1] = out_level_offsets[rev_level] + level_counts[tree_depth];
    }

    // Step 4: Fill out_order by placing each node at its reversed level
    // Reverse level = max_depth - depth (so leaves at depth max_depth -> level 0)
    int* write_positions = arena->allocate<int>(static_cast<size_t>(max_depth + 1));
    for (int rev_level = 0; rev_level <= max_depth; rev_level++) {
        write_positions[rev_level] = out_level_offsets[rev_level];
    }

    // Traverse all nodes and place them in reverse level order
    for (int i = 0; i < num_nodes(); i++) {
        if (depths[i] < 0)
            continue;
        int tree_depth = depths[i];
        int rev_level = max_depth - tree_depth;  // Reverse: leaves (depth=max) -> level 0
        int pos = write_positions[rev_level]++;
        out_order[pos] = i;
    }

    // Note: Nodes are now grouped by level, with leaves (level 0) first,
    // root (level max_depth) last. This is reverse level-order (bottom-up).

    // Fill any unused slots with sentinel value
    int total_nodes = out_level_offsets[max_depth + 1];
    for (int i = total_nodes; i < num_nodes(); ++i) {
        out_order[i] = -1;
    }
}

const char* GuideTree::to_newick(const char** seq_names,
                                 pfalign::memory::GrowableArena* arena) const {
    // Allocate buffer for Newick string
    // Worst case: ~30 chars per node (name + distance + delimiters)
    const int buffer_size = num_nodes() * 30;
    char* buffer = arena->allocate<char>(static_cast<size_t>(buffer_size));
    int offset = 0;

    // Recursive helper function
    struct NewickHelper {
        static void write_node(const GuideTree& tree, int node_idx, const char** seq_names,
                               char* buffer, int buffer_size, int& offset) {
            const GuideTreeNode& node = tree.node(node_idx);

            if (node.is_leaf) {
                // Leaf: write sequence name
                const char* name = seq_names[node.seq_id];
                int len = static_cast<int>(std::strlen(name));
                std::memcpy(buffer + offset, name, static_cast<size_t>(len));
                offset += len;
            } else {
                // Internal node: (left,right)
                buffer[offset++] = '(';
                write_node(tree, node.left_child, seq_names, buffer, buffer_size, offset);
                buffer[offset++] = ',';
                write_node(tree, node.right_child, seq_names, buffer, buffer_size, offset);
                buffer[offset++] = ')';
            }

            // Write branch length if not root
            if (node_idx != tree.root_index()) {
                int remaining = buffer_size - offset;
                if (remaining > 0) {
                    int written = std::snprintf(buffer + offset, static_cast<size_t>(remaining),
                                                ":%.6f", static_cast<double>(node.distance));
                    offset += std::max(0, written);
                }
            }
        }
    };

    // Write tree starting from root
    NewickHelper::write_node(*this, root_idx_, seq_names, buffer, buffer_size, offset);

    // Terminate with semicolon and null
    buffer[offset++] = ';';
    buffer[offset] = '\0';

    return buffer;
}

}  // namespace types
}  // namespace pfalign
