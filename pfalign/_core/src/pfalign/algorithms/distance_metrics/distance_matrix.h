/**
 * Distance matrix computation for guide tree construction.
 *
 * Computes pairwise alignment distances from MPNN embeddings.
 * Used as input to guide tree algorithms (UPGMA, NJ, BIONJ, MST).
 */

#pragma once

#include <functional>

#include "pfalign/common/growable_arena.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/modules/mpnn/sequence_cache.h"

namespace pfalign {
namespace msa {

/**
 * Compute pairwise distance matrix for guide tree construction.
 *
 * Uses alignment-based distance from posterior-weighted cosine similarity.
 *
 * Distance formula:
 *   d(i,j) = (1 - score_cosine) / 2
 *
 * where score_cosine ∈ [-1, +1] is the posterior-weighted cosine similarity
 * from pairwise alignment. This maps similarity [-1, +1] → distance [0, 1].
 *
 * IMPORTANT: This formula relies on the pairwise alignment score being bounded
 * to [-1, +1] range (posterior-weighted cosine similarity). If the scoring
 * function changes in the future to produce unbounded values, this distance
 * computation will need to be updated accordingly.
 *
 * Properties:
 * - d(i,i) = 0 (self-distance is zero)
 * - d(i,j) = d(j,i) (symmetric)
 * - d(i,j) ∈ [0, 1] (non-negative and bounded)
 * - Empirically, d ∈ [0.05, 0.5] for typical aligned proteins
 *
 * Parallelization (auto-dispatch):
 * - N >= 10: Parallel implementation using ThreadPool (row-level parallelism)
 *   - Each thread processes a range of rows independently
 *   - Expected speedup: 3.3-8× on 4-8 cores for N=20-100
 *   - No synchronization needed (disjoint row ownership)
 * - N < 10: Sequential implementation (avoids thread overhead)
 *
 * Memory management:
 * - cache: Holds pre-computed MPNN embeddings (persistent arena, never reset)
 * - scratch_arena: Temporary allocations for alignment (sequential mode only)
 * - workspace: Reused across all N² alignment pairs (per-thread in parallel mode)
 * - Parallel mode: ThreadPool provides per-thread arenas (~50 MB each)
 *
 * Complexity:
 * - Time: O(N² × L² × D) for N sequences of length L, embedding dim D
 * - Space: O(N × L × D) for embeddings + O(L²) for workspace (reused)
 *
 * @param cache         SequenceCache with precomputed MPNN embeddings
 * @param sw_config     Smith-Waterman configuration (gap penalties, temperature)
 * @param scratch_arena Arena for temporary allocations (separate from cache arena)
 * @param distances     Output distance matrix [N × N] (symmetric, caller-allocated)
 *
 * Example:
 * ```cpp
 * Arena cache_arena(50 * 1024 * 1024);   // 50 MB persistent
 * Arena scratch_arena(10 * 1024 * 1024); // 10 MB scratch
 *
 * SequenceCache cache(&cache_arena);
 * for (int i = 0; i < N; i++) {
 *     cache.add_sequence(coords[i], lengths[i], ids[i], weights, mpnn_config);
 * }
 *
 * float* distances = new float[N * N];
 *
 * SWConfig sw_config;
 * sw_config.affine = true;
 * sw_config.gap_open = -1.0f;
 * sw_config.gap_extend = -0.1f;
 * sw_config.temperature = 1.0f;
 *
 * compute_distance_matrix_alignment(
 *     cache, sw_config, &scratch_arena, distances
 * );
 *
 * // Build guide tree from distances
 * GuideTree tree = build_nj_tree(distances, N, &cache_arena);
 *
 * delete[] distances;
 * ```
 */
void compute_distance_matrix_alignment(
    const SequenceCache& cache,
    const pfalign::smith_waterman::SWConfig& sw_config,
    pfalign::memory::GrowableArena* scratch_arena,
    float* distances,  // [N × N] output
    size_t num_threads = 0,  // 0 = auto-detect
    std::function<void(int, int)> progress_callback = nullptr  // (current, total) callback
);

/**
 * Validate distance matrix properties.
 *
 * Checks that the distance matrix satisfies required properties:
 * 1. Self-distance is zero: d(i,i) = 0
 * 2. Symmetry: d(i,j) = d(j,i)
 * 3. Non-negativity: d(i,j) >= 0
 * 4. Bounded by 1: d(i,j) <= 1
 *
 * @param distances Distance matrix [N × N]
 * @param N         Number of sequences
 * @param eps       Tolerance for floating-point comparisons (default: 1e-6)
 * @return          True if all properties are satisfied
 */
bool validate_distance_matrix(const float* distances, int N, float eps = 1e-6f);

}  // namespace msa
}  // namespace pfalign
