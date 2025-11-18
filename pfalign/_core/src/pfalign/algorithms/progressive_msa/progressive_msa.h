/**
 * Progressive Multiple Sequence Alignment (MSA) using guide trees.
 *
 * This module implements batched progressive MSA via reverse level-order
 * traversal of guide trees, enabling parallel batch processing at each level.
 *
 * Algorithm:
 * 1. Compute guide tree from pairwise distance matrix (UPGMA/NJ/BIONJ/MST)
 * 2. Compute reverse level-order traversal (leaves->root, batched by depth)
 * 3. Initialize leaf profiles from cached sequence embeddings
 * 4. For each level (bottom-up):
 *    a. Batch all profile-profile alignments at this level
 *    b. Merge aligned profiles into new profiles
 * 5. Extract final MSA from root profile
 *
 * Key features:
 * - Batched processing: All operations at same tree level run in parallel
 * - Profile-aware alignment: Uses ECS (Embedding Coherence Score) similarity
 * - Memory efficient: Arena-based allocation with bulk deallocation
 * - GPU-ready: Natural batching for SIMD/CUDA acceleration
 *
 * Example:
 *   // 1. Setup
 *   Arena arena(100 * 1024 * 1024);
 *   SequenceCache cache(&arena);
 *
 *   // Add sequences
 *   for (const auto& protein : proteins) {
 *       cache.add_sequence(coords, length, name, weights, config);
 *   }
 *
 *   // 2. Build guide tree
 *   GuideTree tree = build_upgma_tree(
 *       distances, cache.size(), &arena
 *   );
 *
 *   // 3. Run progressive MSA
 *   MSAConfig msa_config;
 *   Profile* msa = progressive_msa<ScalarBackend>(
 *       cache, tree, msa_config, &arena
 *   );
 *
 *   // 4. Extract alignment
 *   printf("Final MSA: %d sequences, %d columns\n",
 *          msa->num_sequences, msa->length);
 */

#pragma once

#include "pfalign/common/growable_arena.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/types/guide_tree_types.h"
#include "pfalign/modules/msa/profile.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include <functional>
#include <vector>
#include <string>

namespace pfalign {
namespace msa {

using pfalign::types::GuideTree;
using pfalign::types::GuideTreeNode;

/**
 * Configuration for progressive MSA.
 */
struct MSAConfig {
    // Smith-Waterman alignment parameters
    float gap_penalty;     // Gap penalty for profile-profile alignment (default: -0.1)
    float gap_open;        // Gap opening penalty for affine mode (default: -1.0)
    float gap_extend;      // Gap extension penalty for affine mode (default: -0.1)
    float temperature;     // Temperature for soft alignment (default: 1.0)
    bool use_affine_gaps;  // Use affine gap penalties (default: true)

    // Posterior decoding parameters
    float decode_gap_penalty;  // Gap penalty for hard decoder (default: -1.0, MEA: 0.0)

    // ECS similarity parameters
    float ecs_temperature;  // Temperature for ECS-weighted similarity (default: 5.0)

    // Thread configuration
    int thread_count;

    // Progress callback (optional)
    std::function<void(int, int)> progress_callback;  // (current, total) callback

    MSAConfig()
        : gap_penalty(-2.0f),
          gap_open(-1.0f),
          gap_extend(-0.1f),
          temperature(1.0f),
          use_affine_gaps(false),
          decode_gap_penalty(-10.0f),
          ecs_temperature(20.0f),
          thread_count(0),
          progress_callback(nullptr) {
    }
};

/**
 * Workspace for progressive MSA.
 *
 * Manages temporary memory for alignment operations:
 * - Similarity matrices
 * - Smith-Waterman DP matrices
 * - Posterior alignment matrices
 * - Alignment column buffers
 *
 * Memory is allocated from arena and reused across alignment operations.
 */
struct MSAWorkspace {
    // Similarity computation
    float* similarity_matrix;  // [max_L1 * max_L2]
    int max_L1;
    int max_L2;

    // Smith-Waterman DP
    float* dp_matrix;  // [max_L1 * max_L2 * max_states]
    int max_states;    // 1 for regular, 3 for affine

    // Posteriors (alignment matrix from backward pass)
    float* posteriors;  // [max_L1 * max_L2]

    // Alignment columns buffer
    AlignmentColumn* alignment_columns;  // [max_aligned_length]
    int max_aligned_length;

    // Arena for workspace allocations
    pfalign::memory::GrowableArena* arena;

    /**
     * Create workspace with initial capacity.
     *
     * @param initial_L1            Initial capacity for L1 dimension
     * @param initial_L2            Initial capacity for L2 dimension
     * @param initial_aligned_len   Initial capacity for aligned length
     * @param use_affine            Allocate for affine gaps (3 states) or regular (1 state)
     * @param arena                 Arena for allocations
     */
    static MSAWorkspace* create(int initial_L1, int initial_L2, int initial_aligned_len,
                                bool use_affine, pfalign::memory::GrowableArena* arena);

    /**
     * Ensure workspace has sufficient capacity for given dimensions.
     *
     * Reallocates from arena if current capacity is insufficient.
     *
     * @param L1                Required L1 dimension
     * @param L2                Required L2 dimension
     * @param aligned_length    Required aligned length
     */
    void ensure_capacity(int L1, int L2, int aligned_length);

    /**
     * Destroy workspace and release any heap-backed buffers (std::vector in columns).
     * Does not free arena storage.
     */
    static void destroy(MSAWorkspace* workspace);
};

/**
 * Result of progressive MSA.
 *
 * Contains the final alignment profile and metadata.
 */
struct MSAResult {
    Profile* alignment;          // Final MSA profile (root of guide tree)
    int num_sequences;           // Number of sequences aligned
    int aligned_length;          // Number of columns in alignment
    float ecs;                   // Embedding Coherence Score (quality metric)
    const SequenceCache* cache;  // Reference to sequence cache (coords + metadata)

    MSAResult()
        : alignment(nullptr), num_sequences(0), aligned_length(0), ecs(0.0f), cache(nullptr) {
    }

    /**
     * Write aligned sequences to FASTA format (gapped).
     *
     * @param output_path Destination FASTA file path
     * @return true on success, false on error
     */
    bool write_fasta(const std::string& output_path) const;

    /**
     * Select a reference sequence (fewest gaps) for structural output.
     *
     * @return Sequence index in [0, num_sequences)
     */
    int select_reference_sequence() const;

    /**
     * Write superposed coordinates for all sequences to a multi-model PDB.
     *
     * @param output_path Destination PDB file path
     * @param reference_seq_idx Reference sequence index (-1 = auto-select)
     * @param arena Optional arena for temporary allocations
     * @return true on success, false on error
     */
    bool write_superposed_pdb(const std::string& output_path, int reference_seq_idx = -1,
                              pfalign::memory::GrowableArena* arena = nullptr) const;
};

// ============================================================================
// Progressive MSA Algorithm
// ============================================================================

/**
 * Perform progressive multiple sequence alignment using guide tree.
 *
 * Uses reverse level-order traversal to enable batching of operations
 * at each tree level. All alignments at the same level can be processed
 * in parallel.
 *
 * Parallelization (auto-dispatch):
 * - N >= 10: Parallel implementation using ThreadPool (level-wise parallelism)
 *   - Each level processes all nodes in parallel using per-thread workspaces
 *   - Leaf profiles initialized in parallel
 *   - Expected speedup: 3-5* on 8 cores for N=20-100
 *   - Constrained by Amdahl's Law (root merge is sequential)
 * - N < 10: Sequential implementation (avoids thread overhead)
 *
 * @tparam Backend      Execution backend (ScalarBackend, SIMDBackend, CUDABackend)
 * @param cache         Cached sequence embeddings [N sequences]
 * @param tree          Guide tree determining merge order
 * @param config        MSA configuration (gap penalties, temperature, etc.)
 * @param arena         Arena for allocating profiles and temporary data
 * @return              MSA result with final alignment profile
 *
 * Algorithm steps:
 * 1. Compute reverse level-order traversal of tree
 * 2. Initialize leaf profiles from cached sequences (parallel for N >= 10)
 * 3. For each level (bottom-up):
 *    a. Collect all profile pairs to align at this level
 *    b. Align each pair (profile-profile alignment) - parallel for N >= 10
 *    c. Merge aligned profiles into new profiles
 * 4. Return root profile as final MSA
 *
 * Memory management:
 * - All profiles allocated from arena (automatic cleanup)
 * - Sequential mode: Single workspace reused across alignments
 * - Parallel mode: Per-thread workspaces and arenas (~50 MB each)
 * - Peak memory: O(sqrtN) profiles at middle tree levels
 *
 * Performance:
 * - Time: O(N^2 L^2) total work (same as sequential)
 * - Parallelism: Up to 2^k-way at level k
 * - Parallel speedup: 3-5* on 8 cores (limited by tree depth)
 */
template <typename Backend>
MSAResult progressive_msa(const SequenceCache& cache, const GuideTree& tree,
                          const MSAConfig& config, pfalign::memory::GrowableArena* arena);

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Align two profiles and merge into new profile.
 *
 * Steps:
 * 1. Compute profile-aware similarity matrix (ECS-weighted)
 * 2. Run Smith-Waterman forward pass (get partition function)
 * 3. Run Smith-Waterman backward pass (get posteriors)
 * 4. Convert posteriors to alignment columns
 * 5. Merge profiles according to alignment
 *
 * @tparam Backend      Execution backend
 * @param profile1      First profile
 * @param profile2      Second profile
 * @param config        MSA configuration
 * @param workspace     Reusable workspace for alignment
 * @param arena         Arena for allocating merged profile
 * @return              Merged profile
 */
template <typename Backend>
Profile* align_and_merge_profiles(const Profile& profile1, const Profile& profile2,
                                  const MSAConfig& config, MSAWorkspace* workspace,
                                  pfalign::memory::GrowableArena* arena);

/**
 * Convert posterior alignment matrix to alignment columns.
 *
 * Extracts the most likely alignment path from the posterior probability
 * matrix by selecting high-probability position pairs.
 *
 * @param posteriors        Posterior matrix [L1 * L2] from backward pass
 * @param L1                Length of profile 1
 * @param L2                Length of profile 2
 * @param profile1          First profile (for column construction)
 * @param profile2          Second profile (for column construction)
 * @param out_columns       Output alignment columns
 * @param out_length        Output aligned length
 * @param threshold         Minimum posterior probability to include (default: 0.01)
 */
void posteriors_to_alignment(const float* posteriors, int L1, int L2, const Profile& profile1,
                             const Profile& profile2, AlignmentColumn* out_columns, int* out_length,
                             float threshold = 0.01f);

/**
 * Compute profile-aware similarity matrix.
 *
 * Uses ECS-weighted dot product similarity:
 * - Standard dot product between profile embeddings
 * - Weighted by column coherence (low coherence = downweight)
 * - Temperature scaling for soft alignment
 *
 * @tparam Backend      Execution backend
 * @param profile1      First profile
 * @param profile2      Second profile
 * @param ecs_temp      ECS temperature (higher = less weight on coherence)
 * @param out_matrix    Output similarity matrix [L1 * L2]
 */
template <typename Backend>
void compute_profile_similarity(const Profile& profile1, const Profile& profile2, float ecs_temp,
                                float* out_matrix);

}  // namespace msa
}  // namespace pfalign
