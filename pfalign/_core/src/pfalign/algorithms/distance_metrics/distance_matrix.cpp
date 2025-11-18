/**
 * Distance matrix computation implementation.
 */

#include "distance_matrix.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/thread_pool.h"
#include <atomic>
#include <cmath>
#include <cstdio>

namespace pfalign {
namespace msa {

// Sequential implementation (used for small N or as fallback)
static void compute_distance_matrix_alignment_sequential(
    const SequenceCache& cache, const pfalign::smith_waterman::SWConfig& sw_config,
    pfalign::memory::GrowableArena* scratch_arena, float* distances,
    std::function<void(int, int)> progress_callback = nullptr) {
    int N = cache.size();
    const auto& sequences = cache.sequences();

    // Get maximum dimensions from cache (no hard-coded values)
    int L_max = cache.max_length();
    int hidden_dim = cache.hidden_dim();

    // Configure pairwise alignment
    pfalign::pairwise::PairwiseConfig config;
    config.sw_config = sw_config;
    config.sw_mode = pfalign::pairwise::PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;
    config.mpnn_config.hidden_dim = hidden_dim;  // Sync hidden_dim with cache

    // Allocate workspace ONCE for largest possible pair
    // This is reused across all N² alignments
    pfalign::pairwise::PairwiseWorkspace workspace(L_max, L_max, config);

    // Total pairs to compute
    int total_pairs = N * (N - 1) / 2;
    int completed = 0;

    // Compute all pairwise distances
    for (int i = 0; i < N; i++) {
        // Self-distance is always zero
        distances[i * N + i] = 0.0f;

        for (int j = i + 1; j < N; j++) {
            const SequenceEmbeddings* seq1 = sequences[i];
            const SequenceEmbeddings* seq2 = sequences[j];

            // Scoped arena: automatically resets scratch_arena when destroyed
            // This ensures we don't accumulate memory across pairs
            pfalign::memory::ScopedGrowableArena pair_scope(*scratch_arena);

            float partition, score_cosine;

            // Compute alignment (forward + backward pass)
            // score_cosine ∈ [-1, +1] (posterior-weighted cosine similarity)
            pfalign::pairwise::pairwise_align_from_embeddings_with_score<pfalign::ScalarBackend>(
                seq1->embeddings, seq1->length, seq2->embeddings, seq2->length, hidden_dim, config,
                &workspace, &partition, &score_cosine, scratch_arena);

            // Convert cosine similarity → distance
            // Map [-1, +1] → [0, 1]
            // score = +1 (identical) → distance = 0.0
            // score = 0 (orthogonal) → distance = 0.5
            // score = -1 (opposite) → distance = 1.0
            float distance = (1.0f - score_cosine) / 2.0f;

            // Clamp to valid range (defensive, should not be needed)
            if (distance < 0.0f)
                distance = 0.0f;
            if (distance > 1.0f)
                distance = 1.0f;

            // Store symmetric distance
            distances[i * N + j] = distance;
            distances[j * N + i] = distance;

            // Update progress
            completed++;
            if (progress_callback) {
                progress_callback(completed, total_pairs);
            }

            // pair_scope destructor resets scratch_arena for next iteration
        }
    }
}

/**
 * Convert linear pair index to (i, j) coordinates in upper triangle.
 *
 * Upper triangle enumeration (row-major):
 *   pair_idx=0 → (0,1), pair_idx=1 → (0,2), ..., pair_idx=N-2 → (0,N-1)
 *   pair_idx=N-1 → (1,2), pair_idx=N → (1,3), ..., pair_idx=2N-3 → (1,N-1)
 *   ...
 *
 * Uses inverse triangular number formula to compute row index i,
 * then derives column index j from the residual.
 *
 * @param pair_idx Linear index of pair in range [0, N*(N-1)/2)
 * @param N        Size of distance matrix
 * @param i        Output: row index
 * @param j        Output: column index (j > i)
 */
inline void pair_index_to_ij(size_t pair_idx, int N, size_t& i, size_t& j) {
    // Inverse triangular number formula:
    // For row-major enumeration, the pair index for (i, j) where j > i is:
    //   pair_idx = i * (2N - i - 1) / 2 + (j - i - 1)
    //
    // Solve for i using quadratic formula:
    //   i ≈ floor((2N - 1 - sqrt((2N-1)^2 - 8*pair_idx)) / 2)

    double N_d = static_cast<double>(N);
    double discriminant =
        (2.0 * N_d - 1.0) * (2.0 * N_d - 1.0) - 8.0 * static_cast<double>(pair_idx);
    double i_exact = (2.0 * N_d - 1.0 - std::sqrt(discriminant)) / 2.0;
    i = static_cast<size_t>(i_exact);

    // Compute j from residual
    size_t offset = i * (2 * static_cast<size_t>(N) - i - 1) / 2;
    j = pair_idx - offset + i + 1;
}

// Parallel implementation using ThreadPool (default for N >= 10)
static void
compute_distance_matrix_alignment_parallel(const SequenceCache& cache,
                                           const pfalign::smith_waterman::SWConfig& sw_config,
                                           float* distances,
                                           size_t num_threads = 0,  // 0 = auto-detect
                                           std::function<void(int, int)> progress_callback = nullptr
) {
    int N = cache.size();
    const auto& sequences = cache.sequences();

    // Get maximum dimensions from cache
    int L_max = cache.max_length();
    int hidden_dim = cache.hidden_dim();

    // Configure pairwise alignment
    pfalign::pairwise::PairwiseConfig config;
    config.sw_config = sw_config;
    config.sw_mode = pfalign::pairwise::PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;
    config.mpnn_config.hidden_dim = hidden_dim;

    // Create ThreadPool with per-thread arenas
    // Arena size: ~50 MB per thread for scratch allocations
    // num_threads=0 means auto-detect based on system resources
    pfalign::threading::ThreadPool pool(num_threads, 50);

    // Calculate total pairs in upper triangle
    size_t total_pairs = static_cast<size_t>(N) * (N - 1) / 2;

    // Atomic counter for thread-safe progress tracking
    std::atomic<int> completed_count{0};

    // Parallel computation: each thread processes a range of PAIRS (not rows!)
    // This ensures perfect load balancing (each thread gets ~equal work)
    pool.parallel_for(total_pairs, [&](int tid, size_t pair_begin, size_t pair_end,
                                       pfalign::memory::GrowableArena& thread_arena) {
        (void)tid;  // Unused

        // Per-thread workspace (reused across all pairs processed by this thread)
        pfalign::pairwise::PairwiseWorkspace workspace(L_max, L_max, config);

        for (size_t pair_idx = pair_begin; pair_idx < pair_end; pair_idx++) {
            // Convert linear pair index → (i, j) coordinates
            size_t i, j;
            pair_index_to_ij(pair_idx, N, i, j);

            const SequenceEmbeddings* seq1 = sequences[i];
            const SequenceEmbeddings* seq2 = sequences[j];

            // Scoped arena: automatically resets thread_arena when destroyed
            pfalign::memory::ScopedGrowableArena pair_scope(thread_arena);

            float partition, score_cosine;

            // Compute alignment (forward + backward pass)
            pfalign::pairwise::pairwise_align_from_embeddings_with_score<pfalign::ScalarBackend>(
                seq1->embeddings, seq1->length, seq2->embeddings, seq2->length, hidden_dim, config,
                &workspace, &partition, &score_cosine, &thread_arena);

            // Convert cosine similarity → distance
            float distance = (1.0f - score_cosine) / 2.0f;

            // Clamp to valid range
            if (distance < 0.0f)
                distance = 0.0f;
            if (distance > 1.0f)
                distance = 1.0f;

            // Store distance symmetrically in both triangles
            // This avoids a second parallel_for call (which spawns threads again)
            // Multiple threads may write to different locations, no race condition
            distances[i * N + j] = distance;
            distances[j * N + i] = distance;

            // Update progress (throttled to every 10 pairs to reduce overhead)
            if (progress_callback) {
                int local_count = completed_count.fetch_add(1) + 1;
                if (local_count % 10 == 0 || local_count == static_cast<int>(total_pairs)) {
                    progress_callback(local_count, static_cast<int>(total_pairs));
                }
            }

            // pair_scope destructor resets thread_arena for next iteration
        }
    });

    // Fill diagonal (only 100 elements, negligible)
    for (int i = 0; i < N; i++) {
        distances[i * N + i] = 0.0f;
    }
}

// Public API with auto-dispatch (sequential for small N, parallel for large N)
void compute_distance_matrix_alignment(
    const SequenceCache& cache,
    const pfalign::smith_waterman::SWConfig& sw_config,
    pfalign::memory::GrowableArena* scratch_arena,
    float* distances,
    std::function<void(int, int)> progress_callback) {
    int N = cache.size();

    // Use parallel implementation for N >= 10 (O(N²) grows quickly)
    // Use sequential for small N to avoid thread overhead
    if (N >= 10) {
        compute_distance_matrix_alignment_parallel(cache, sw_config, distances, 0, progress_callback);
    } else {
        compute_distance_matrix_alignment_sequential(cache, sw_config, scratch_arena, distances, progress_callback);
    }
}

bool validate_distance_matrix(const float* distances, int N, float eps) {
    for (int i = 0; i < N; i++) {
        // 1. Self-distance is zero
        if (std::abs(distances[i * N + i]) > eps) {
            std::fprintf(stderr, "Distance matrix validation failed: d(%d,%d) = %f (expected 0)\n",
                         i, i, distances[i * N + i]);
            return false;
        }

        for (int j = 0; j < N; j++) {
            // 2. Symmetry
            float d_ij = distances[i * N + j];
            float d_ji = distances[j * N + i];
            if (std::abs(d_ij - d_ji) > eps) {
                std::fprintf(stderr,
                             "Distance matrix validation failed: d(%d,%d) = %f != d(%d,%d) = %f "
                             "(asymmetric)\n",
                             i, j, d_ij, j, i, d_ji);
                return false;
            }

            // 3. Non-negativity
            if (d_ij < -eps) {
                std::fprintf(stderr,
                             "Distance matrix validation failed: d(%d,%d) = %f (negative)\n", i, j,
                             d_ij);
                return false;
            }

            // 4. Bounded by 1
            if (d_ij > 1.0f + eps) {
                std::fprintf(stderr,
                             "Distance matrix validation failed: d(%d,%d) = %f (exceeds 1.0)\n", i,
                             j, d_ij);
                return false;
            }
        }
    }

    return true;
}

}  // namespace msa
}  // namespace pfalign
