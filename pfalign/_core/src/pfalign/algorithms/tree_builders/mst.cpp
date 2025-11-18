/**
 * MST guide tree construction with midpoint rooting.
 *
 * Builds a minimum spanning tree from the distance matrix using Kruskal's
 * algorithm, then converts it to a rooted binary tree via midpoint rooting.
 *
 * Algorithm:
 * 1. Create edge list from all N(N-1)/2 pairwise distances
 * 2. Sort edges by distance (ascending)
 * 3. Use Union-Find to build MST (Kruskal's algorithm)
 * 4. Root at midpoint of longest edge:
 *    - Find longest edge (i,j) in MST
 *    - Split edge, create root R at midpoint
 *    - R becomes parent of i and j with distances d/2 each
 * 5. Build binary tree via DFS from root:
 *    - If vertex has >2 children, create intermediate binary nodes
 *    - Preserve all MST edges with correct distances
 * 6. Assign heights via traversal
 *
 * Result: Binary tree with N leaves, N-1 internal nodes
 * MST may have high-degree vertices; conversion creates extra internal nodes
 * to enforce binary property while preserving topology.
 *
 * Complexity: O(N^2 log N) time, O(N^2) space
 */

#include "builders.h"
#include "pfalign/types/guide_tree_types.h"
#include "pfalign/common/arena_allocator.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

namespace pfalign {
namespace tree_builders {

using pfalign::types::GuideTree;
using pfalign::types::GuideTreeNode;

/**
 * Edge in the distance graph.
 */
struct Edge {
    int i, j;        ///< Node indices
    float distance;  ///< Edge weight

    bool operator<(const Edge& other) const {
        return distance < other.distance;
    }
};

/**
 * Union-Find (Disjoint Set Union) data structure.
 */
class UnionFind {
public:
    UnionFind(int n, pfalign::memory::GrowableArena* arena) : n_(n) {
        parent_ = arena->allocate<int>(n);
        rank_ = arena->allocate<int>(n);
        for (int i = 0; i < n; i++) {
            parent_[i] = i;
            rank_[i] = 0;
        }
    }

    int find(int x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);  // Path compression
        }
        return parent_[x];
    }

    bool unite(int x, int y) {
        int rx = find(x);
        int ry = find(y);

        if (rx == ry)
            return false;  // Already in same set

        // Union by rank
        if (rank_[rx] < rank_[ry]) {
            parent_[rx] = ry;
        } else if (rank_[rx] > rank_[ry]) {
            parent_[ry] = rx;
        } else {
            parent_[ry] = rx;
            rank_[rx]++;
        }
        return true;
    }

private:
    [[maybe_unused]] int n_;
    int* parent_;
    int* rank_;
};

/**
 * Adjacency list with growable neighbor storage.
 */
struct AdjacencyList {
    struct Neighbor {
        int node;
        float distance;
    };

    Neighbor** neighbors;  ///< neighbors[i] = array of neighbors of node i
    int* num_neighbors;    ///< num_neighbors[i] = count
    int* capacity;         ///< capacity[i] = allocated size

    AdjacencyList(int n, pfalign::memory::GrowableArena* arena) : arena_(arena) {
        neighbors = arena->allocate<Neighbor*>(n);
        num_neighbors = arena->allocate<int>(n);
        capacity = arena->allocate<int>(n);
        for (int i = 0; i < n; i++) {
            neighbors[i] = nullptr;
            num_neighbors[i] = 0;
            capacity[i] = 0;
        }
    }

    void add_edge(int u, int v, float dist) {
        add_neighbor(u, v, dist);
        add_neighbor(v, u, dist);  // Undirected
    }

private:
    pfalign::memory::GrowableArena* arena_;

    void add_neighbor(int u, int v, float dist) {
        // Grow if needed
        if (num_neighbors[u] >= capacity[u]) {
            int new_cap = (capacity[u] == 0) ? 4 : capacity[u] * 2;
            Neighbor* new_neighbors = arena_->allocate<Neighbor>(new_cap);

            // Copy existing
            if (neighbors[u] != nullptr) {
                for (int i = 0; i < num_neighbors[u]; i++) {
                    new_neighbors[i] = neighbors[u][i];
                }
            }

            neighbors[u] = new_neighbors;
            capacity[u] = new_cap;
        }

        neighbors[u][num_neighbors[u]++] = {v, dist};
    }
};

/**
 * Build MST guide tree from distance matrix.
 *
 * @param distances Distance matrix [N * N]
 * @param N         Number of sequences
 * @param arena     Arena for allocating tree nodes
 * @return          Binary guide tree with midpoint rooting
 */
GuideTree build_mst_tree(const float* distances, int N, pfalign::memory::GrowableArena* arena) {
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

    // Step 1: Create edge list from distance matrix
    int num_edges = N * (N - 1) / 2;
    Edge* edges = arena->allocate<Edge>(num_edges);
    int edge_idx = 0;

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            edges[edge_idx++] = {i, j, distances[i * N + j]};
        }
    }

    // Step 2: Sort edges by distance (ascending)
    std::sort(edges, edges + num_edges);

    // Step 3: Build MST using Kruskal's algorithm (Union-Find)
    UnionFind uf(N, arena);
    Edge* mst_edges = arena->allocate<Edge>(N - 1);
    int mst_count = 0;

    for (int e = 0; e < num_edges && mst_count < N - 1; e++) {
        if (uf.unite(edges[e].i, edges[e].j)) {
            mst_edges[mst_count++] = edges[e];
        }
    }

    if (mst_count != N - 1) {
        std::fprintf(stderr, "MST: Failed to build spanning tree (got %d edges, expected %d)\n",
                     mst_count, N - 1);
    }

    // Step 4: Find longest edge for midpoint rooting
    int longest_idx = 0;
    for (int e = 1; e < mst_count; e++) {
        if (mst_edges[e].distance > mst_edges[longest_idx].distance) {
            longest_idx = e;
        }
    }

    int root_left = mst_edges[longest_idx].i;
    int root_right = mst_edges[longest_idx].j;
    float root_dist = mst_edges[longest_idx].distance;

    // Build adjacency list (excluding longest edge, which we'll root on)
    AdjacencyList adj(N, arena);
    for (int e = 0; e < mst_count; e++) {
        if (e == longest_idx)
            continue;  // Skip longest edge
        adj.add_edge(mst_edges[e].i, mst_edges[e].j, mst_edges[e].distance);
    }

    // Step 5: Build binary tree structure via DFS from root
    // Allocate tree nodes: N leaves + up to (N-1) internal nodes for MST edges
    // + extra nodes for binarizing high-degree vertices
    // Worst case: star topology requires N-2 extra nodes for binarization, plus root
    // Safe upper bound: 3*N (generous for any MST topology)
    GuideTreeNode* nodes = arena->allocate<GuideTreeNode>(3 * N);

    // Construct all nodes (0 to 2N-2, which is the full range needed)
    // MST may not use all nodes sequentially, so we initialize the full range
    for (int i = 0; i < 2 * N - 1; i++) {
        new (&nodes[i]) GuideTreeNode();
    }

    // Initialize leaf nodes (0 to N-1)
    for (int i = 0; i < N; i++) {
        nodes[i] = GuideTreeNode::create_leaf(i, i);
    }

    int next_node_id = N;

    // Track which nodes have been visited in DFS
    bool* visited = arena->allocate<bool>(N);
    for (int i = 0; i < N; i++)
        visited[i] = false;

    // Recursive DFS to build binary tree structure
    // Returns: (node_id of subtree root, height of subtree, total sequences)
    struct SubtreeInfo {
        int node_id;
        float height;
        int num_sequences;
    };

    // Lambda for recursive DFS (using std::function to allow recursion)
    std::function<SubtreeInfo(int node, int parent, const AdjacencyList& adj, GuideTreeNode* nodes,
                              int& next_node_id, bool* visited)>
        build_subtree;

    // Define recursive function
    build_subtree = [&build_subtree, N](int node, int parent, const AdjacencyList& adj,
                                        GuideTreeNode* nodes, int& next_node_id,
                                        bool* visited) -> SubtreeInfo {
        visited[node] = true;

        // Collect unvisited children (excluding parent)
        struct ChildInfo {
            int node_id;
            float distance;
            float height;
            int num_sequences;
        };

        std::vector<ChildInfo> children;
        children.reserve(adj.num_neighbors[node]);

        for (int n = 0; n < adj.num_neighbors[node]; n++) {
            int neighbor = adj.neighbors[node][n].node;
            float dist = adj.neighbors[node][n].distance;

            if (neighbor == parent || visited[neighbor])
                continue;

            SubtreeInfo child_info =
                build_subtree(neighbor, node, adj, nodes, next_node_id, visited);

            children.push_back(
                {child_info.node_id, dist, child_info.height, child_info.num_sequences});
        }

        if (children.empty()) {
            SubtreeInfo result;
            result.node_id = node;
            result.height = 0.0f;
            result.num_sequences = 1;
            return result;
        }

        // Include the current vertex itself as a leaf (zero-length branch)
        children.insert(children.begin(), ChildInfo{node, 0.0f, 0.0f, 1});

        auto attach_children = [&](const ChildInfo& left, const ChildInfo& right) -> ChildInfo {
            nodes[left.node_id].distance = left.distance;
            nodes[right.node_id].distance = right.distance;

            float left_height = left.height + left.distance;
            float right_height = right.height + right.distance;
            float new_height = std::max(left_height, right_height);
            int new_num_seq = left.num_sequences + right.num_sequences;

            int new_node = next_node_id++;
            if (new_node >= 3 * N) {
                printf("ERROR: MST node overflow! new_node=%d, max=%d (N=%d)\n", new_node,
                       3 * N - 1, N);
                fflush(stdout);
                abort();
            }

            nodes[new_node] = GuideTreeNode::create_internal(new_node, left.node_id, right.node_id,
                                                             0.0f, new_num_seq, new_height);

            return ChildInfo{new_node, 0.0f, new_height, new_num_seq};
        };

        // Merge children into a binary structure
        ChildInfo combined = attach_children(children[0], children[1]);
        for (size_t c = 2; c < children.size(); ++c) {
            ChildInfo rhs = children[c];
            ChildInfo lhs{combined.node_id, 0.0f, combined.height, combined.num_sequences};
            combined = attach_children(lhs, rhs);
        }

        SubtreeInfo result;
        result.node_id = combined.node_id;
        result.height = combined.height;
        result.num_sequences = combined.num_sequences;
        return result;
    };

    // Build left and right subtrees
    SubtreeInfo left_info = build_subtree(root_left, -1, adj, nodes, next_node_id, visited);
    SubtreeInfo right_info = build_subtree(root_right, -1, adj, nodes, next_node_id, visited);

    // Create root node - MUST be at index 2*N-2 for GuideTree constructor
    int root = 2 * N - 2;  // GuideTree expects root at this index
    float branch_left = root_dist / 2.0f;
    float branch_right = root_dist / 2.0f;

    // Safety check: Ensure we don't exceed allocated space
    if (root >= 3 * N) {
        printf("ERROR: MST root overflow! root=%d, max=%d (N=%d)\n", root, 3 * N - 1, N);
        fflush(stdout);
        abort();
    }

    nodes[left_info.node_id].distance = branch_left;
    nodes[right_info.node_id].distance = branch_right;

    float height = std::max(left_info.height + branch_left, right_info.height + branch_right);

    int total_sequences = left_info.num_sequences + right_info.num_sequences;
    assert(total_sequences == N && "Guide tree must include all sequences");

    nodes[root] = GuideTreeNode::create_internal(root, left_info.node_id, right_info.node_id,
                                                 0.0f,  // Root has no parent
                                                 total_sequences, height);

    return GuideTree(nodes, N);
}

}  // namespace tree_builders
}  // namespace pfalign
