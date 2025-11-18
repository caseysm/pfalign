/**
 * Unit tests for progressive MSA.
 *
 * Tests the complete MSA pipeline:
 * 1. Sequence caching
 * 2. Guide tree construction
 * 3. Progressive alignment
 * 4. Profile merging
 * 5. ECS computation
 */

#include "pfalign/algorithms/progressive_msa/progressive_msa.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/common/perf_timer.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/algorithms/distance_metrics/distance_matrix.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace pfalign;
using namespace pfalign::tree_builders;
using namespace pfalign::msa;
using namespace pfalign::tree_builders;
using namespace pfalign::memory;
using namespace pfalign::tree_builders;
using pfalign::ScalarBackend;

/**
 * Create synthetic protein coordinates for testing.
 *
 * Generates a simple extended structure (all phi/psi = -120deg/+120deg).
 *
 * @param L         Sequence length
 * @param coords    Output coordinates [L * 4 * 3]
 */
void create_synthetic_coords(int L, float* coords) {
    // Simple extended structure
    for (int i = 0; i < L; i++) {
        // N atom
        coords[i * 12 + 0] = i * 3.8f;
        coords[i * 12 + 1] = 0.0f;
        coords[i * 12 + 2] = 0.0f;

        // CA atom
        coords[i * 12 + 3] = i * 3.8f + 1.5f;
        coords[i * 12 + 4] = 0.0f;
        coords[i * 12 + 5] = 0.0f;

        // C atom
        coords[i * 12 + 6] = i * 3.8f + 2.5f;
        coords[i * 12 + 7] = 0.0f;
        coords[i * 12 + 8] = 0.0f;

        // O atom
        coords[i * 12 + 9] = i * 3.8f + 3.5f;
        coords[i * 12 + 10] = 0.0f;
        coords[i * 12 + 11] = 0.0f;
    }
}

/**
 * Create synthetic embeddings directly (bypass MPNN encoding for testing).
 */
SequenceEmbeddings* create_synthetic_embeddings(
    int seq_id,
    int length,
    int hidden_dim,
    const char* identifier,
    GrowableArena* arena
) {
    SequenceEmbeddings* seq = arena->allocate<SequenceEmbeddings>(1);
    // CRITICAL: Use placement-new to construct std::string member
    new (seq) SequenceEmbeddings();

    seq->seq_id = seq_id;
    seq->length = length;
    seq->hidden_dim = hidden_dim;
    seq->identifier = identifier;

    // Allocate coords and embeddings
    seq->coords = arena->allocate<float>(length * 12);
    seq->embeddings = arena->allocate<float>(length * hidden_dim);

    // Create simple coords
    create_synthetic_coords(length, seq->coords);

    // Create random embeddings (normalized)
    for (int i = 0; i < length; i++) {
        float* emb = seq->embeddings + i * hidden_dim;
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            emb[d] = (float)(rand() % 1000) / 1000.0f - 0.5f;
            norm_sq += emb[d] * emb[d];
        }
        // Normalize
        float norm = std::sqrt(norm_sq);
        if (norm > 0.0f) {
            for (int d = 0; d < hidden_dim; d++) {
                emb[d] /= norm;
            }
        }
    }

    return seq;
}

/**
 * Test MSA with 2 sequences (minimal case).
 */
void test_two_sequence_msa() {
    printf("Testing MSA with 2 sequences...\n");

    GrowableArena arena(100);  // 100 MB

    // Create synthetic sequences with embeddings
    int L1 = 10;
    int L2 = 12;
    int hidden_dim = 64;

    SequenceEmbeddings* seq1 = create_synthetic_embeddings(0, L1, hidden_dim, "Seq1", &arena);
    SequenceEmbeddings* seq2 = create_synthetic_embeddings(1, L2, hidden_dim, "Seq2", &arena);

    // Create sequence cache and add pre-computed embeddings
    SequenceCache cache(&arena);
    cache.add_sequence_from_embeddings(seq1);
    cache.add_sequence_from_embeddings(seq2);

    assert(cache.size() == 2);
    printf("  ✓ Added 2 sequences to cache\n");

    // Build guide tree (simple 2-sequence tree)
    float distances[4] = {
        0.0f, 0.5f,
        0.5f, 0.0f
    };

    GuideTree tree = build_upgma_tree(distances, 2, &arena);
    assert(tree.num_sequences() == 2);
    assert(tree.num_nodes() == 3);  // 2 leaves + 1 root
    printf("  ✓ Built guide tree\n");

    // Run progressive MSA
    MSAConfig msa_config;
    msa_config.use_affine_gaps = true;
    msa_config.gap_open = -1.0f;
    msa_config.gap_extend = -0.1f;

    MSAResult result = progressive_msa<ScalarBackend>(cache, tree, msa_config, &arena);

    // Verify result
    assert(result.alignment != nullptr);
    assert(result.num_sequences == 2);
    assert(result.aligned_length >= L1);  // At least as long as shorter sequence
    assert(result.aligned_length <= L1 + L2);  // At most sum of both
    printf("  ✓ MSA completed: %d sequences, %d columns\n",
           result.num_sequences, result.aligned_length);

    // Check ECS is in valid range
    assert(result.ecs >= -1.0f && result.ecs <= 1.0f);
    printf("  ✓ ECS = %.4f (valid range)\n", result.ecs);

    printf("✓ Two-sequence MSA test passed\n\n");

    Profile::destroy(result.alignment);
}

/**
 * Test MSA with 4 sequences (small tree with multiple levels).
 */
void test_four_sequence_msa() {
    printf("Testing MSA with 4 sequences...\n");

    GrowableArena arena(100);

    // Create sequences of varying lengths
    int lengths[4] = {8, 10, 9, 11};
    int hidden_dim = 64;

    // Create sequence cache
    SequenceCache cache(&arena);
    for (int i = 0; i < 4; i++) {
        char name[32];
        std::snprintf(name, sizeof(name), "Seq%d", i);
        SequenceEmbeddings* seq = create_synthetic_embeddings(i, lengths[i], hidden_dim, name, &arena);
        cache.add_sequence_from_embeddings(seq);
    }

    assert(cache.size() == 4);
    printf("  ✓ Added 4 sequences to cache (lengths: %d, %d, %d, %d)\n",
           lengths[0], lengths[1], lengths[2], lengths[3]);

    // Build distance matrix (simple symmetric matrix)
    float distances[16] = {
        0.0f, 0.2f, 0.4f, 0.6f,
        0.2f, 0.0f, 0.3f, 0.5f,
        0.4f, 0.3f, 0.0f, 0.4f,
        0.6f, 0.5f, 0.4f, 0.0f
    };

    GuideTree tree = build_upgma_tree(distances, 4, &arena);
    assert(tree.num_sequences() == 4);
    assert(tree.num_nodes() == 7);  // 4 leaves + 3 internal
    printf("  ✓ Built guide tree (7 nodes)\n");

    // Compute reverse level-order to verify batching
    int* order = arena.allocate<int>(tree.num_nodes());
    int* level_offsets = arena.allocate<int>(tree.num_nodes() + 1);
    int depth;
    tree.compute_reverse_level_order(&arena, order, level_offsets, &depth);

    printf("  ✓ Tree depth: %d levels\n", depth);
    for (int level = 0; level < depth; level++) {
        int num_nodes = level_offsets[level + 1] - level_offsets[level];
        printf("    Level %d: %d nodes\n", level, num_nodes);
    }

    // Run progressive MSA
    MSAConfig msa_config;
    MSAResult result = progressive_msa<ScalarBackend>(cache, tree, msa_config, &arena);

    // Verify result
    assert(result.alignment != nullptr);
    assert(result.num_sequences == 4);
    printf("  ✓ MSA completed: %d sequences, %d columns\n",
           result.num_sequences, result.aligned_length);
    printf("  ✓ ECS = %.4f\n", result.ecs);

    // Check that alignment includes all sequences
    for (int col = 0; col < result.aligned_length; col++) {
        assert(result.alignment->columns[col].positions.size() == 4);
    }
    printf("  ✓ All columns have 4 positions (correct structure)\n");

    printf("✓ Four-sequence MSA test passed\n\n");

    Profile::destroy(result.alignment);
}

/**
 * Test MSA with single sequence (edge case).
 */
void test_single_sequence_msa() {
    printf("Testing MSA with single sequence (edge case)...\n");

    GrowableArena arena(100);

    int L = 15;
    int hidden_dim = 64;

    SequenceEmbeddings* seq = create_synthetic_embeddings(0, L, hidden_dim, "SingleSeq", &arena);

    SequenceCache cache(&arena);
    cache.add_sequence_from_embeddings(seq);

    // Build trivial tree (single node)
    float distances[1] = {0.0f};
    GuideTree tree = build_upgma_tree(distances, 1, &arena);

    MSAConfig msa_config;
    MSAResult result = progressive_msa<ScalarBackend>(cache, tree, msa_config, &arena);

    // Verify result
    assert(result.alignment != nullptr);
    assert(result.num_sequences == 1);
    assert(result.aligned_length == L);
    assert(std::abs(result.ecs - 1.0f) < 0.001f);  // Perfect coherence
    printf("  ✓ Single sequence MSA: L=%d, ECS=%.4f\n", L, result.ecs);

    printf("✓ Single-sequence MSA test passed\n\n");

    Profile::destroy(result.alignment);
}

/**
 * Test posteriors_to_alignment function.
 */
void test_posteriors_to_alignment() {
    printf("Testing posteriors_to_alignment...\n");

    GrowableArena arena(10);

    // Create simple 3*3 posterior matrix with clear diagonal
    int L1 = 3;
    int L2 = 3;
    float posteriors[9] = {
        0.8f, 0.1f, 0.05f,
        0.1f, 0.7f, 0.1f,
        0.05f, 0.1f, 0.75f
    };

    // Create dummy profiles
    int hidden_dim = 64;
    float* emb1 = arena.allocate<float>(L1 * hidden_dim);
    float* emb2 = arena.allocate<float>(L2 * hidden_dim);
    std::memset(emb1, 0, L1 * hidden_dim * sizeof(float));
    std::memset(emb2, 0, L2 * hidden_dim * sizeof(float));

    Profile* p1 = Profile::from_single_sequence(emb1, L1, hidden_dim, 0, &arena);
    Profile* p2 = Profile::from_single_sequence(emb2, L2, hidden_dim, 1, &arena);

    // Convert to alignment
    AlignmentColumn* columns = arena.allocate<AlignmentColumn>(L1 + L2);
    // Construct each column
    for (int i = 0; i < L1 + L2; i++) {
        new (&columns[i]) AlignmentColumn();
    }
    int aligned_length;

    posteriors_to_alignment(
        posteriors, L1, L2, *p1, *p2,
        columns, &aligned_length,
        0.5f  // High threshold to select only diagonal
    );

    // Should align 3 positions (diagonal)
    assert(aligned_length == 3);
    printf("  ✓ Aligned length: %d (expected 3)\n", aligned_length);

    // Each column should have 2 sequences (p1 and p2)
    for (int i = 0; i < aligned_length; i++) {
        assert(columns[i].positions.size() == 2);
    }
    printf("  ✓ All columns have 2 positions\n");

    printf("✓ posteriors_to_alignment test passed\n\n");

    for (int i = 0; i < L1 + L2; i++) {
        columns[i].~AlignmentColumn();
    }
    Profile::destroy(p1);
    Profile::destroy(p2);
}

/**
 * Force the MSA workspace to grow multiple times via ensure_capacity.
 * Verifies that column vectors are reconstructed correctly after each reallocation.
 */
void test_msa_workspace_reallocation() {
    printf("Testing MSAWorkspace reallocation stress...\n");

    GrowableArena arena(10);
    // Start tiny so we trigger growth on every iteration.
    MSAWorkspace* workspace = MSAWorkspace::create(
        4, 4, 4,
        true,  // affine mode → dp has 3 states
        &arena
    );

    struct Requirements {
        int L1;
        int L2;
        int aligned;
    };

    // Each step substantially increases the requested capacity so we force
    // multiple rounds of reallocation and column reconstruction.
    Requirements growth_plan[] = {
        {6, 5, 12},
        {12, 10, 24},
        {18, 16, 48},
        {32, 28, 96},
        {56, 40, 160},
    };

    for (size_t step = 0; step < sizeof(growth_plan) / sizeof(growth_plan[0]); ++step) {
        Requirements req = growth_plan[step];

        float* prev_similarity = workspace->similarity_matrix;
        float* prev_dp = workspace->dp_matrix;
        float* prev_post = workspace->posteriors;
        AlignmentColumn* prev_columns = workspace->alignment_columns;

        int prev_L1 = workspace->max_L1;
        int prev_L2 = workspace->max_L2;
        int prev_aligned = workspace->max_aligned_length;

        workspace->ensure_capacity(req.L1, req.L2, req.aligned);

        // All buffers must accommodate the request.
        assert(workspace->max_L1 >= req.L1);
        assert(workspace->max_L2 >= req.L2);
        assert(workspace->max_aligned_length >= req.aligned);
        assert(workspace->similarity_matrix != nullptr);
        assert(workspace->dp_matrix != nullptr);
        assert(workspace->posteriors != nullptr);
        assert(workspace->alignment_columns != nullptr);

        // When the request exceeds the previous maxima, a fresh allocation
        // should occur for every buffer (monotonic arena ⇒ pointer changes).
        assert(prev_similarity != workspace->similarity_matrix);
        assert(prev_dp != workspace->dp_matrix);
        assert(prev_post != workspace->posteriors);
        assert(prev_columns != workspace->alignment_columns);

        // Newly constructed columns must behave like default-constructed
        // vectors: start empty, accept pushes, and own their storage.
        for (int col = 0; col < 3; ++col) {
            assert(workspace->alignment_columns[col].positions.empty());
            workspace->alignment_columns[col].positions.push_back({static_cast<int>(step), col});
            workspace->alignment_columns[col].positions.push_back({static_cast<int>(step) + 1, col + 1});
            assert(workspace->alignment_columns[col].positions.size() == 2);
            workspace->alignment_columns[col].positions.clear();
            assert(workspace->alignment_columns[col].positions.empty());
        }

        printf("  ✓ Growth %zu -> L1=%d L2=%d aligned=%d (prev L1=%d L2=%d aligned=%d)\n",
               step + 1, workspace->max_L1, workspace->max_L2, workspace->max_aligned_length,
               prev_L1, prev_L2, prev_aligned);
    }

    MSAWorkspace::destroy(workspace);

    printf("✓ MSAWorkspace reallocation stress test passed\n\n");
}

/**
 * Test MST star topology (high-degree vertex binarization).
 */
void test_mst_star_topology() {
    printf("Testing MST with star topology...\n");

    GrowableArena arena(100);
    printf("  [DEBUG] Arena created: capacity=%zu MB\n", arena.capacity() / (1024*1024));
    int hidden_dim = 64;

    // Create 6 sequences (1 hub + 5 spokes)
    int N = 6;
    int lengths[6] = {10, 10, 10, 10, 10, 10};

    printf("  [DEBUG] Creating SequenceCache...\n");
    fflush(stdout);
    SequenceCache cache(&arena);
    printf("  [DEBUG] SequenceCache created\n");

    for (int i = 0; i < N; i++) {
        printf("  [DEBUG] Creating sequence %d/%d (length=%d)...\n", i+1, N, lengths[i]);
        fflush(stdout);
        char name[32];
        std::snprintf(name, sizeof(name), "Seq%d", i);
        SequenceEmbeddings* seq = create_synthetic_embeddings(i, lengths[i], hidden_dim, name, &arena);
        printf("  [DEBUG] Adding sequence %d to cache...\n", i+1);
        fflush(stdout);
        cache.add_sequence_from_embeddings(seq);
        printf("  [DEBUG] Sequence %d added (cache size=%d)\n", i+1, cache.size());
        fflush(stdout);
    }

    printf("  [DEBUG] All sequences added successfully\n");
    fflush(stdout);

    // Create star topology distance matrix:
    // Seq0 is hub, all others connect to it
    // Distance from 0 to any other: 1.0
    // Distance between non-hub sequences: 2.0
    printf("  [DEBUG] Allocating distance matrix (%d x %d)...\n", N, N);
    fflush(stdout);
    float* distances = arena.allocate<float>(N * N);
    printf("  [DEBUG] Distance matrix allocated, filling...\n");
    fflush(stdout);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                distances[i * N + j] = 0.0f;
            } else if (i == 0 || j == 0) {
                distances[i * N + j] = 1.0f;  // Hub connections
            } else {
                distances[i * N + j] = 2.0f;  // Non-hub to non-hub
            }
        }
    }
    printf("  [DEBUG] Distance matrix filled\n");
    fflush(stdout);

    // Build MST (should create star topology)
    printf("  [DEBUG] Building MST guide tree...\n");
    fflush(stdout);
    GuideTree tree = build_mst_tree(distances, N, &arena);
    printf("  [DEBUG] MST guide tree built: %d nodes\n", tree.num_nodes());
    fflush(stdout);
    printf("  ✓ Built MST guide tree (%d nodes)\n", tree.num_nodes());

    // Verify all sequences are represented in the tree
    int leaf_count = 0;
    bool seq_found[6] = {false};
    for (int i = 0; i < tree.num_nodes(); i++) {
        if (tree.node(i).is_leaf) {
            leaf_count++;
            int seq_id = tree.node(i).seq_id;
            assert(seq_id >= 0 && seq_id < N);
            assert(!seq_found[seq_id]);  // No duplicates
            seq_found[seq_id] = true;
        }
    }

    assert(leaf_count == N);
    for (int i = 0; i < N; i++) {
        assert(seq_found[i]);  // All sequences present
    }
    printf("  ✓ All %d sequences preserved in tree\n", N);

    // Debug: Print tree structure before MSA
    printf("  [DEBUG] Tree structure (num_nodes=%d, root=%d):\n", tree.num_nodes(), tree.root_index());
    for (int i = 0; i < tree.num_nodes(); i++) {
        const GuideTreeNode& node = tree.node(i);
        printf("    Node %d: is_leaf=%d, seq_id=%d, left=%d, right=%d, num_seq=%d\n",
               i, node.is_leaf, node.seq_id, node.left_child, node.right_child, node.num_sequences);
    }
    fflush(stdout);

    // Run progressive MSA
    printf("  → Starting progressive_msa with %d sequences...\n", N);
    fflush(stdout);
    MSAConfig msa_config;
    MSAResult result = progressive_msa<ScalarBackend>(cache, tree, msa_config, &arena);
    printf("  ← progressive_msa completed\n");
    fflush(stdout);

    assert(result.alignment != nullptr);
    assert(result.num_sequences == N);
    printf("  ✓ MSA completed: %d sequences, %d columns, ECS=%.4f\n",
           result.num_sequences, result.aligned_length, result.ecs);

    // Verify all columns have N positions
    for (int col = 0; col < result.aligned_length; col++) {
        assert(result.alignment->columns[col].positions.size() == (size_t)N);
    }
    printf("  ✓ All columns have %d positions\n", N);

    printf("✓ MST star topology test passed\n\n");

    Profile::destroy(result.alignment);
}

/**
 * Test progressive MSA with different guide tree algorithms.
 */
void test_all_guide_tree_types() {
    printf("Testing progressive MSA with all guide tree types...\n");

    GrowableArena arena(100);
    int hidden_dim = 64;

    // Create 4 sequences with varying lengths
    int N = 4;
    int lengths[4] = {8, 10, 9, 11};

    SequenceCache cache(&arena);
    for (int i = 0; i < N; i++) {
        char name[32];
        std::snprintf(name, sizeof(name), "Seq%d", i);
        SequenceEmbeddings* seq = create_synthetic_embeddings(i, lengths[i], hidden_dim, name, &arena);
        cache.add_sequence_from_embeddings(seq);
    }

    // Simple symmetric distance matrix
    float distances[16] = {
        0.0f, 0.2f, 0.4f, 0.6f,
        0.2f, 0.0f, 0.3f, 0.5f,
        0.4f, 0.3f, 0.0f, 0.4f,
        0.6f, 0.5f, 0.4f, 0.0f
    };

    MSAConfig msa_config;

    // Test each guide tree algorithm
    const char* methods[] = {"UPGMA", "NJ", "BIONJ", "MST"};

    for (int method_idx = 0; method_idx < 4; method_idx++) {
        const char* method = methods[method_idx];

        // Build guide tree
        GuideTree tree;
        switch (method_idx) {
            case 0: tree = build_upgma_tree(distances, N, &arena); break;
            case 1: tree = build_nj_tree(distances, N, &arena); break;
            case 2: tree = build_bionj_tree(distances, N, &arena); break;
            case 3: tree = build_mst_tree(distances, N, &arena); break;
        }

        assert(tree.num_sequences() == N);
        printf("  Testing %s guide tree...\n", method);

        // Run progressive MSA
        MSAResult result = progressive_msa<ScalarBackend>(cache, tree, msa_config, &arena);

        // Verify result
        assert(result.alignment != nullptr);
        assert(result.num_sequences == N);
        assert(result.aligned_length > 0);
        assert(result.ecs >= -1.0f && result.ecs <= 1.0f);

        // Verify all columns have N positions
        for (int col = 0; col < result.aligned_length; col++) {
            assert(result.alignment->columns[col].positions.size() == (size_t)N);
        }

        printf("    ✓ %s: %d sequences, %d columns, ECS=%.4f\n",
               method, result.num_sequences, result.aligned_length, result.ecs);

        Profile::destroy(result.alignment);
    }

    printf("✓ All guide tree types test passed\n\n");
}

/**
 * Regression test for N=45 crash (GitHub issue: std::bad_alloc in progressive MSA).
 *
 * This test reproduces the allocation pattern that caused crashes with N=45:
 * - Large merged profiles (L1 ~ 1890, L2 ~ 1474)
 * - Total allocation: ~53 MB per workspace
 * - Arena capacity: 200 MB
 *
 * The test verifies:
 * 1. Large allocations don't crash with sufficient arena
 * 2. Arena exhaustion is properly detected
 * 3. Workspace can handle real-world late-stage progressive MSA sizes
 */
void test_large_allocation_regression() {
    printf("Testing large allocation regression (N=45 crash)...\n");

    // Test 1: Verify large allocation succeeds with sufficient arena
    {
        printf("  Test 1: Large allocation with 100 MB arena...\n");
        GrowableArena arena(100);  // 100 MB - sufficient for one large workspace

        MSAWorkspace* workspace = MSAWorkspace::create(
            100, 100, 100,
            true,  // affine mode (3 states)
            &arena
        );

        // Simulate N=45 late-stage merge parameters
        // From debug log: L1=1890, L2=1474, aligned_length=3364
        // Total: ~53 MB (10 MB + 31 MB + 10 MB + 0.08 MB)
        workspace->ensure_capacity(1890, 1474, 3364);

        // Verify allocation succeeded
        assert(workspace->max_L1 >= 1890);
        assert(workspace->max_L2 >= 1474);
        assert(workspace->max_aligned_length >= 3364);
        assert(workspace->similarity_matrix != nullptr);
        assert(workspace->dp_matrix != nullptr);
        assert(workspace->posteriors != nullptr);
        assert(workspace->alignment_columns != nullptr);

        printf("    ✓ Large allocation (L1=%d, L2=%d, aligned=%d) succeeded\n",
               workspace->max_L1, workspace->max_L2, workspace->max_aligned_length);
        printf("    ✓ Arena used: %.1f MB / %.1f MB\n",
               arena.used() / 1024.0 / 1024.0, arena.capacity() / 1024.0 / 1024.0);
    }

    // Test 2: Verify arena exhaustion is properly detected
    {
        printf("  Test 2: Arena exhaustion detection...\n");
        GrowableArena arena(10);  // Only 10 MB - too small for large allocation

        MSAWorkspace* workspace = MSAWorkspace::create(
            100, 100, 100,
            true,
            &arena
        );

        bool threw_bad_alloc = false;
        try {
            // This should exhaust the 10 MB arena
            workspace->ensure_capacity(2000, 2000, 4000);
        } catch (const std::bad_alloc& e) {
            threw_bad_alloc = true;
        }

        assert(threw_bad_alloc);
        printf("    ✓ Arena exhaustion properly throws std::bad_alloc\n");
    }

    // Test 3: Progressive growth leading to large allocation
    {
        printf("  Test 3: Progressive growth pattern...\n");
        GrowableArena arena(80);  // 80 MB arena

        MSAWorkspace* workspace = MSAWorkspace::create(
            200, 200, 200,
            true,
            &arena
        );

        // Simulate progressive MSA growth pattern (increasing aligned lengths)
        struct GrowthStep {
            int L1, L2, aligned;
        };

        GrowthStep steps[] = {
            {400, 400, 800},     // Early merge
            {600, 600, 1200},    // Mid merge
            {800, 800, 1600},    // Late merge
            {1200, 1000, 2200},  // Very late merge
            {1500, 1200, 2700},  // Near-final merge
        };

        for (size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); i++) {
            workspace->ensure_capacity(steps[i].L1, steps[i].L2, steps[i].aligned);

            assert(workspace->max_L1 >= steps[i].L1);
            assert(workspace->max_L2 >= steps[i].L2);
            assert(workspace->max_aligned_length >= steps[i].aligned);
        }

        printf("    ✓ Progressive growth pattern succeeded\n");
        printf("    ✓ Final dimensions: L1=%d, L2=%d, aligned=%d\n",
               workspace->max_L1, workspace->max_L2, workspace->max_aligned_length);
        printf("    ✓ Arena used: %.1f MB / %.1f MB\n",
               arena.used() / 1024.0 / 1024.0, arena.capacity() / 1024.0 / 1024.0);
    }

    printf("✓ Large allocation regression test passed\n\n");
}

int main() {
    pfalign::perf::PerfTimer perf_timer("test_progressive_msa");
    printf("=== Progressive MSA Tests ===\n\n");

    test_single_sequence_msa();
    test_two_sequence_msa();
    test_four_sequence_msa();
    test_mst_star_topology();
    test_all_guide_tree_types();
    test_posteriors_to_alignment();
    test_msa_workspace_reallocation();
    test_large_allocation_regression();

    printf("=== All tests passed! ===\n");
    return 0;
}
