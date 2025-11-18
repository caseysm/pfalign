/**
 * Progressive MSA implementation.
 */

#include "progressive_msa.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/primitives/alignment_decode/alignment_decode.h"
#include "pfalign/common/thread_pool.h"
#include <atomic>

// ============================================================================
// PROFILING INSTRUMENTATION
// ============================================================================
// ENABLE_PROFILING is controlled by meson build option: -Dprofiling=true/false
// Defaults to false (disabled) for zero-overhead in production builds.

#if ENABLE_PROFILING
    #include "pfalign/common/profile_scope.h"
    #include "pfalign/common/cpu_utilization.h"
    #include "pfalign/common/memory_timeline.h"

    // Convenience macros
    #define MSA_PROFILE_SCOPE(name) pfalign::ProfileScope _profile_scope_##__LINE__(name)
    #define MSA_CPU_TRACKER(name, threads) \
        pfalign::CPUUtilizationTracker _cpu_tracker_##__LINE__(name, threads)
#else
    #define MSA_PROFILE_SCOPE(name) ((void)0)
    #define MSA_CPU_TRACKER(name, threads) ((void)0)
#endif
// ============================================================================

#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <mutex>
#include <chrono>
#include <thread>

namespace pfalign {
namespace msa {

using pfalign::memory::GrowableArena;
using pfalign::smith_waterman::smith_waterman_jax_affine_flexible;
using pfalign::smith_waterman::smith_waterman_jax_affine_flexible_backward;
using pfalign::smith_waterman::smith_waterman_jax_regular;
using pfalign::smith_waterman::smith_waterman_jax_regular_backward;
using pfalign::smith_waterman::SWConfig;

namespace {

std::string make_sequence_label(const SequenceEmbeddings* seq, int seq_idx) {
    if (seq && !seq->identifier.empty()) {
        return seq->identifier;
    }
    return "sequence_" + std::to_string(seq_idx);
}

void write_wrapped_sequence(std::ofstream& out, const std::string& sequence) {
    constexpr size_t kWidth = 80;
    for (size_t pos = 0; pos < sequence.size(); pos += kWidth) {
        out << sequence.substr(pos, kWidth) << "\n";
    }
}

char chain_id_for_index(int idx) {
    const int offset = idx % 26;
    return static_cast<char>('A' + offset);
}

struct SuperposedSequenceScratch {
    std::vector<float> transformed_coords;
    float rmsd = 0.0f;
    int aligned_pairs = 0;
    bool has_transform = false;
};

struct SuperposeWorkItem {
    std::vector<int> reference_positions;
    std::vector<int> sequence_positions;
};

void write_msa_model(std::ofstream& out, int model_index, char chain_id, const std::string& label,
                     const float* coords, int length, const std::vector<uint8_t>& aligned_flags,
                     float rmsd, int aligned_pairs) {
    out << "MODEL     " << std::setw(4) << model_index << "\n";
    out << "REMARK   2 SEQUENCE: " << label << "\n";
    out << "REMARK   2 LENGTH: " << length << " RESIDUES\n";
    out << "REMARK   2 RMSD TO REFERENCE: " << std::fixed << std::setprecision(4) << rmsd << "\n";
    out << "REMARK   2 MATCHED POSITIONS: " << aligned_pairs << "\n";

    static constexpr const char* kAtomNames[] = {"N", "CA", "C", "O"};
    int atom_serial = 1;

    for (int res = 0; res < length; ++res) {
        const bool aligned = (res < static_cast<int>(aligned_flags.size())) && aligned_flags[res];
        const float bfactor = aligned ? 1.0f : 0.0f;

        for (int atom = 0; atom < 4; ++atom) {
            const float x = coords[(res * 4 + atom) * 3 + 0];
            const float y = coords[(res * 4 + atom) * 3 + 1];
            const float z = coords[(res * 4 + atom) * 3 + 2];

            if (x == 0.0f && y == 0.0f && z == 0.0f) {
                continue;
            }

            out << "ATOM  " << std::setw(5) << atom_serial << " " << std::setw(4) << std::left
                << kAtomNames[atom] << std::right << "ALA " << chain_id << std::setw(4) << (res + 1)
                << "    " << std::fixed << std::setprecision(3) << std::setw(8) << x << std::setw(8)
                << y << std::setw(8) << z << std::setw(6) << std::setprecision(2) << 1.00f
                << std::setw(6) << std::setprecision(2) << bfactor << "           "
                << kAtomNames[atom][0] << "\n";

            atom_serial++;
        }
    }

    out << "ENDMDL\n";
}

}  // namespace

// ============================================================================
// TempProfile - Temporary profile structure for thread-local computation
// ============================================================================

namespace {

/**
 * Temporary profile structure for parallel profile-profile alignment.
 *
 * Lives entirely in thread-local arena during parallel compute phase.
 * Converted to persistent Profile in main arena during sequential allocation phase.
 *
 * Key differences from Profile:
 * - All allocations from thread-local arena (not main arena)
 * - columns stored as std::vector (will be copied to arena during conversion)
 * - seq_indices stored as std::vector (will be copied during conversion)
 */
struct TempProfile {
    int length;
    int num_sequences;
    int hidden_dim;

    // Thread-arena-allocated arrays
    float* embeddings;  // [length * hidden_dim]
    float* weights;     // [length]
    float* sum_norm;    // [length * hidden_dim]
    int* gap_counts;    // [length]

    // Heap-allocated vectors (will be copied to main arena during conversion)
    std::vector<AlignmentColumn> columns;
    std::vector<int> seq_indices;

    /**
     * Create TempProfile from alignment of two profiles.
     *
     * Performs the same computation as Profile::from_alignment but allocates
     * everything from thread-local arena.
     *
     * @param profile1        First profile
     * @param profile2        Second profile
     * @param alignment       Alignment columns
     * @param aligned_length  Number of alignment columns
     * @param thread_arena    Thread-local arena for allocations
     * @return                TempProfile allocated from thread_arena
     */
    static TempProfile* create_from_alignment(const Profile& profile1, const Profile& profile2,
                                              const AlignmentColumn* alignment, int aligned_length,
                                              GrowableArena* thread_arena) {
        int num_sequences = profile1.num_sequences + profile2.num_sequences;
        int hidden_dim = profile1.hidden_dim;

        // Allocate TempProfile from thread arena
        TempProfile* temp = thread_arena->allocate<TempProfile>(1);
        new (temp) TempProfile();  // Placement-new to construct std::vectors

        temp->length = aligned_length;
        temp->num_sequences = num_sequences;
        temp->hidden_dim = hidden_dim;

        // Allocate arrays from thread arena
        temp->embeddings = thread_arena->allocate<float>(aligned_length * hidden_dim);
        temp->weights = thread_arena->allocate<float>(aligned_length);
        temp->sum_norm = thread_arena->allocate<float>(aligned_length * hidden_dim);
        temp->gap_counts = thread_arena->allocate<int>(aligned_length);

        // Reserve capacity for vectors
        temp->columns.reserve(aligned_length);
        temp->seq_indices.reserve(num_sequences);

        // Merge sequence indices from both profiles
        temp->seq_indices.insert(temp->seq_indices.end(), profile1.seq_indices.begin(),
                                 profile1.seq_indices.end());
        temp->seq_indices.insert(temp->seq_indices.end(), profile2.seq_indices.begin(),
                                 profile2.seq_indices.end());

        int next_profile1_col = 0;
        int next_profile2_col = 0;

        auto accumulate_from_profile = [&](const Profile& source_profile, int source_col,
                                           float* col_embedding, float* col_sum_norm,
                                           float& total_weight) {
            if (source_col < 0 || source_col >= source_profile.length) {
                fprintf(stderr, "ERROR: Invalid source_col=%d (length=%d)\n", source_col,
                        source_profile.length);
                return;
            }

            float weight = source_profile.weights[source_col];
            if (weight <= 0.0f) {
                return;
            }

            const float* source_emb = source_profile.get_embedding(source_col);
            const float* source_sum_norm = source_profile.get_sum_norm(source_col);

            for (int d = 0; d < hidden_dim; ++d) {
                col_embedding[d] += weight * source_emb[d];
                col_sum_norm[d] += source_sum_norm[d];
            }

            total_weight += weight;
        };

        auto column_has_residue = [](const AlignmentColumn& col, int start, int end) {
            int limit = static_cast<int>(col.positions.size());
            end = std::min(end, limit);
            for (int seq = start; seq < end; ++seq) {
                if (!col.positions[seq].is_gap()) {
                    return true;
                }
            }
            return false;
        };

        // Process each alignment column
        for (int col = 0; col < aligned_length; ++col) {
            const AlignmentColumn& align_col = alignment[col];

            // Copy alignment column (deep copy of std::vector)
            temp->columns.push_back(align_col);

            // Initialize column embedding, sum_norm, and weight
            float* col_embedding = &temp->embeddings[col * hidden_dim];
            float* col_sum_norm = &temp->sum_norm[col * hidden_dim];
            std::memset(col_embedding, 0, hidden_dim * sizeof(float));
            std::memset(col_sum_norm, 0, hidden_dim * sizeof(float));
            float total_weight = 0.0f;
            int gap_count = 0;

            bool uses_profile1 = column_has_residue(align_col, 0, profile1.num_sequences);
            bool uses_profile2 =
                column_has_residue(align_col, profile1.num_sequences, num_sequences);

            if (uses_profile1) {
                if (next_profile1_col < profile1.length) {
                    accumulate_from_profile(profile1, next_profile1_col, col_embedding,
                                            col_sum_norm, total_weight);
                } else {
                    fprintf(stderr, "ERROR: profile1 column index overflow (idx=%d, len=%d)\n",
                            next_profile1_col, profile1.length);
                }
                next_profile1_col++;
            }

            if (uses_profile2) {
                if (next_profile2_col < profile2.length) {
                    accumulate_from_profile(profile2, next_profile2_col, col_embedding,
                                            col_sum_norm, total_weight);
                } else {
                    fprintf(stderr, "ERROR: profile2 column index overflow (idx=%d, len=%d)\n",
                            next_profile2_col, profile2.length);
                }
                next_profile2_col++;
            }

            // Count gaps directly from alignment column
            for (const auto& pos : align_col.positions) {
                if (pos.is_gap()) {
                    gap_count++;
                }
            }

            // Normalize weighted embedding
            temp->weights[col] = total_weight;

            // DEBUG: Check for weight explosion
            static std::atomic<int> weight_check_count{0};
            if (col == aligned_length / 2 && weight_check_count.fetch_add(1) < 3) {
                fprintf(
                    stderr,
                    "[WEIGHT DEBUG] Merge col %d: total_weight=%.2e (should be <=200 for N=100)\n",
                    col, total_weight);
                fflush(stderr);
            }

            constexpr float eps = 1e-8f;
            if (total_weight > eps) {
                float inv_weight = 1.0f / total_weight;
                for (int d = 0; d < hidden_dim; ++d) {
                    col_embedding[d] *= inv_weight;
                }
                // sum_norm doesn't get divided - it accumulates across merges
            }

            temp->gap_counts[col] = gap_count;
        }

        return temp;
    }

    /**
     * Convert TempProfile to persistent Profile in main arena.
     *
     * Performs deep copy of all data from thread-local arena to main arena.
     * TempProfile should be destroyed (via thread_arena reset) after this.
     *
     * @param main_arena Main arena for persistent profile allocation
     * @return           Profile allocated from main_arena
     */
    Profile* to_profile(GrowableArena* main_arena) const {
        // Allocate Profile structure from main arena
        Profile* profile = main_arena->allocate<Profile>(1);
        new (profile) Profile(length, num_sequences, hidden_dim, main_arena);

        // Copy arrays from thread arena to main arena
        std::memcpy(profile->embeddings, embeddings, length * hidden_dim * sizeof(float));
        std::memcpy(profile->weights, weights, length * sizeof(float));
        std::memcpy(profile->sum_norm, sum_norm, length * hidden_dim * sizeof(float));
        std::memcpy(profile->gap_counts, gap_counts, length * sizeof(int));

        // DEBUG: Verify weights after copy
        static std::atomic<int> copy_check{0};
        if (copy_check.fetch_add(1) < 3 && length > 100) {
            int mid = length / 2;
            fprintf(stderr,
                    "[COPY DEBUG] TempProfile->Profile: len=%d, temp_weight[%d]=%.2e, "
                    "profile_weight[%d]=%.2e\n",
                    length, mid, weights[mid], mid, profile->weights[mid]);
            fflush(stderr);
        }

        // Deep copy alignment columns
        for (int i = 0; i < length; ++i) {
            profile->columns[i] = columns[i];  // std::vector copy (heap allocation)
        }

        // Copy sequence indices
        profile->seq_indices = seq_indices;  // std::vector copy (heap allocation)

        return profile;
    }
};

}  // namespace

// ============================================================================
// MSAWorkspace Implementation
// ============================================================================

MSAWorkspace* MSAWorkspace::create(int initial_L1, int initial_L2, int initial_aligned_len,
                                   bool use_affine, GrowableArena* arena) {
    MSAWorkspace* ws = arena->allocate<MSAWorkspace>(1);
    ws->arena = arena;
    ws->max_L1 = initial_L1;
    ws->max_L2 = initial_L2;
    ws->max_aligned_length = initial_aligned_len;
    ws->max_states = use_affine ? 3 : 1;

    // Allocate buffers
    ws->similarity_matrix = arena->allocate<float>(initial_L1 * initial_L2);
    ws->dp_matrix = arena->allocate<float>(initial_L1 * initial_L2 * ws->max_states);
    ws->posteriors = arena->allocate<float>(initial_L1 * initial_L2);
    ws->alignment_columns = arena->allocate<AlignmentColumn>(initial_aligned_len);

    // CRITICAL: Placement-new each AlignmentColumn to construct std::vector members
    for (int i = 0; i < initial_aligned_len; i++) {
        new (&ws->alignment_columns[i]) AlignmentColumn();
    }

    return ws;
}

void MSAWorkspace::ensure_capacity(int L1, int L2, int aligned_length) {
    bool need_realloc = false;

    // Save old capacity BEFORE modifying
    int old_max_aligned_length = max_aligned_length;

    if (L1 > max_L1 || L2 > max_L2) {
        // Use 1.5* growth (balances waste vs reallocations)
        // Instead of 2* doubling which overshoots capacity
        max_L1 = std::max(L1, static_cast<int>(max_L1 * 1.5f));
        max_L2 = std::max(L2, static_cast<int>(max_L2 * 1.5f));
        need_realloc = true;
    }

    if (aligned_length > max_aligned_length) {
        // Use 1.5* growth (balances waste vs reallocations)
        max_aligned_length = std::max(aligned_length, static_cast<int>(max_aligned_length * 1.5f));
        need_realloc = true;
    }

    if (need_realloc) {
        // Call destructors on old columns to free std::vector heap memory
        // Note: Arena keeps the raw storage, but we must free vector internals
        if (alignment_columns != nullptr && old_max_aligned_length > 0) {
            for (int i = 0; i < old_max_aligned_length; i++) {
                alignment_columns[i].~AlignmentColumn();
            }
        }

        // Reallocate from arena
        similarity_matrix = arena->allocate<float>(max_L1 * max_L2);
        dp_matrix = arena->allocate<float>(max_L1 * max_L2 * max_states);
        posteriors = arena->allocate<float>(max_L1 * max_L2);
        alignment_columns = arena->allocate<AlignmentColumn>(max_aligned_length);

        // CRITICAL: Placement-new each AlignmentColumn to construct std::vector members
        // Previous buffer is abandoned (arena doesn't free), but new slots need construction
        for (int i = 0; i < max_aligned_length; i++) {
            new (&alignment_columns[i]) AlignmentColumn();
        }
    }
}

void MSAWorkspace::destroy(MSAWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    if (workspace->alignment_columns != nullptr) {
        for (int i = 0; i < workspace->max_aligned_length; ++i) {
            workspace->alignment_columns[i].~AlignmentColumn();
        }
        workspace->alignment_columns = nullptr;
        workspace->max_aligned_length = 0;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute profile-aware similarity matrix using ECS weighting.
 */
template <typename Backend>
void compute_profile_similarity(const Profile& profile1, const Profile& profile2, float ecs_temp,
                                float* out_matrix) {
    MSA_PROFILE_SCOPE("compute_profile_similarity");

    int L1 = profile1.length;
    int L2 = profile2.length;
    int hidden_dim = profile1.hidden_dim;

    // Step 1: Compute raw dot-product similarity
    {
        MSA_PROFILE_SCOPE("similarity_compute_similarity");
        pfalign::similarity::compute_similarity<Backend>(profile1.embeddings, profile2.embeddings,
                                                         out_matrix, L1, L2, hidden_dim);
    }

    // Step 2: Weight by column coherence (ECS-based)
    // For each position pair (i,j), multiply by:
    //   exp((coherence_i + coherence_j) / ecs_temp)
    // This downweights alignments between low-coherence columns

    {
        MSA_PROFILE_SCOPE("ecs_weighting");

        // Compute coherence for each column
        float* coherence1 = static_cast<float*>(alloca(L1 * sizeof(float)));
        float* coherence2 = static_cast<float*>(alloca(L2 * sizeof(float)));

        {
            MSA_PROFILE_SCOPE("ecs_compute_coherence");
            for (int i = 0; i < L1; i++) {
                coherence1[i] = profile1.compute_column_coherence(i);
            }

            for (int j = 0; j < L2; j++) {
                coherence2[j] = profile2.compute_column_coherence(j);
            }
        }

        // Apply ECS weighting
        {
            MSA_PROFILE_SCOPE("ecs_apply_weights");
            for (int i = 0; i < L1; i++) {
                for (int j = 0; j < L2; j++) {
                    float ecs_weight = std::exp((coherence1[i] + coherence2[j]) / ecs_temp);
                    out_matrix[i * L2 + j] *= ecs_weight;
                }
            }
        }
    }
}

/**
 * Convert posteriors to alignment columns.
 *
 * Strategy: Greedy decoding from posteriors
 * 1. Find highest probability position pair (i*, j*)
 * 2. Add to alignment
 * 3. Mask out row i* and column j*
 * 4. Repeat until no pairs above threshold
 * 5. Add remaining unaligned positions as gaps
 */
/**
 * Convert AlignmentPair[] from decode_alignment() to AlignmentColumn[] for MSA.
 *
 * Maps pairwise alignment path to profile-profile alignment columns.
 * Each AlignmentPair represents one column in the merged alignment.
 *
 * @param alignment_path Array of alignment pairs from Viterbi decoder
 * @param path_length Number of pairs in alignment_path
 * @param profile1 First profile being merged
 * @param profile2 Second profile being merged
 * @param out_columns Output buffer for alignment columns
 * @param out_length Output: number of columns written
 */
void convert_alignment_path_to_columns(const AlignmentPair* alignment_path, int path_length,
                                       const Profile& profile1, const Profile& profile2,
                                       AlignmentColumn* out_columns, int* out_length) {
    int num_sequences = profile1.num_sequences + profile2.num_sequences;

    for (int k = 0; k < path_length; k++) {
        const AlignmentPair& pair = alignment_path[k];
        AlignmentColumn& col = out_columns[k];
        col.positions.resize(num_sequences);

        if (pair.i >= 0 && pair.j >= 0) {
            // Match: merge columns from both profiles
            for (int seq_idx = 0; seq_idx < profile1.num_sequences; seq_idx++) {
                col.positions[seq_idx] = profile1.columns[pair.i].positions[seq_idx];
            }
            for (int seq_idx = 0; seq_idx < profile2.num_sequences; seq_idx++) {
                col.positions[profile1.num_sequences + seq_idx] =
                    profile2.columns[pair.j].positions[seq_idx];
            }
        } else if (pair.i >= 0) {
            // Gap in profile2: copy profile1 column, add gaps for profile2
            for (int seq_idx = 0; seq_idx < profile1.num_sequences; seq_idx++) {
                col.positions[seq_idx] = profile1.columns[pair.i].positions[seq_idx];
            }
            for (int seq_idx = 0; seq_idx < profile2.num_sequences; seq_idx++) {
                col.positions[profile1.num_sequences + seq_idx] = {-1, -1};
            }
        } else {
            // Gap in profile1: add gaps for profile1, copy profile2 column
            for (int seq_idx = 0; seq_idx < profile1.num_sequences; seq_idx++) {
                col.positions[seq_idx] = {-1, -1};
            }
            for (int seq_idx = 0; seq_idx < profile2.num_sequences; seq_idx++) {
                col.positions[profile1.num_sequences + seq_idx] =
                    profile2.columns[pair.j].positions[seq_idx];
            }
        }
    }

    *out_length = path_length;
}

void posteriors_to_alignment(const float* posteriors, int L1, int L2, const Profile& profile1,
                             const Profile& profile2, AlignmentColumn* out_columns, int* out_length,
                             float threshold) {
    // Track which positions have been aligned
    bool* aligned1 = static_cast<bool*>(alloca(L1 * sizeof(bool)));
    bool* aligned2 = static_cast<bool*>(alloca(L2 * sizeof(bool)));
    std::memset(aligned1, 0, L1 * sizeof(bool));
    std::memset(aligned2, 0, L2 * sizeof(bool));

    int num_sequences = profile1.num_sequences + profile2.num_sequences;
    int col_idx = 0;

    // Greedy alignment: repeatedly pick highest-probability unaligned pair
    while (true) {
        // Find best unaligned pair
        int best_i = -1;
        int best_j = -1;
        float best_prob = threshold;

        for (int i = 0; i < L1; i++) {
            if (aligned1[i])
                continue;
            for (int j = 0; j < L2; j++) {
                if (aligned2[j])
                    continue;
                float prob = posteriors[i * L2 + j];
                if (prob > best_prob) {
                    best_prob = prob;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i < 0) {
            break;  // No more pairs above threshold
        }

        // Safety check: bounds
        if (best_i < 0 || best_i >= L1 || best_j < 0 || best_j >= L2) {
            fprintf(stderr, "ERROR: best_i=%d or best_j=%d out of bounds (L1=%d, L2=%d)\n", best_i,
                    best_j, L1, L2);
            abort();
        }

        // Create alignment column by merging positions from both profiles
        AlignmentColumn& col = out_columns[col_idx++];
        col.positions.resize(num_sequences);

        // Copy positions from profile1 at column best_i
        for (int seq_idx = 0; seq_idx < profile1.num_sequences; seq_idx++) {
            col.positions[seq_idx] = profile1.columns[best_i].positions[seq_idx];
        }

        // Copy positions from profile2 at column best_j
        for (int seq_idx = 0; seq_idx < profile2.num_sequences; seq_idx++) {
            col.positions[profile1.num_sequences + seq_idx] =
                profile2.columns[best_j].positions[seq_idx];
        }

        aligned1[best_i] = true;
        aligned2[best_j] = true;
    }

    // Add remaining unaligned positions from profile1 as gaps in profile2
    for (int i = 0; i < L1; i++) {
        if (aligned1[i])
            continue;

        AlignmentColumn& col = out_columns[col_idx++];
        col.positions.resize(num_sequences);

        // Copy from profile1
        for (int seq_idx = 0; seq_idx < profile1.num_sequences; seq_idx++) {
            col.positions[seq_idx] = profile1.columns[i].positions[seq_idx];
        }

        // Gaps for profile2
        for (int seq_idx = 0; seq_idx < profile2.num_sequences; seq_idx++) {
            col.positions[profile1.num_sequences + seq_idx] = {-1, -1};  // Gap
        }
    }

    // Add remaining unaligned positions from profile2 as gaps in profile1
    for (int j = 0; j < L2; j++) {
        if (aligned2[j])
            continue;

        AlignmentColumn& col = out_columns[col_idx++];
        col.positions.resize(num_sequences);

        // Gaps for profile1
        for (int seq_idx = 0; seq_idx < profile1.num_sequences; seq_idx++) {
            col.positions[seq_idx] = {-1, -1};  // Gap
        }

        // Copy from profile2
        for (int seq_idx = 0; seq_idx < profile2.num_sequences; seq_idx++) {
            col.positions[profile1.num_sequences + seq_idx] =
                profile2.columns[j].positions[seq_idx];
        }
    }

    *out_length = col_idx;
}

/**
 * Align two profiles and merge into new profile.
 */
template <typename Backend>
Profile* align_and_merge_profiles(const Profile& profile1, const Profile& profile2,
                                  const MSAConfig& config, MSAWorkspace* workspace,
                                  GrowableArena* arena) {
    int L1 = profile1.length;
    int L2 = profile2.length;

    // Ensure workspace has sufficient capacity
    int max_aligned = L1 + L2;  // Worst case: no matches, all gaps
    workspace->ensure_capacity(L1, L2, max_aligned);

    // Step 1: Compute profile-aware similarity
    compute_profile_similarity<Backend>(profile1, profile2, config.ecs_temperature,
                                        workspace->similarity_matrix);

    // Step 2: Configure Smith-Waterman
    SWConfig sw_config;
    sw_config.gap = config.gap_penalty;
    sw_config.gap_open = config.gap_open;
    sw_config.gap_extend = config.gap_extend;
    sw_config.temperature = config.temperature;
    sw_config.affine = config.use_affine_gaps;

    // Step 3: Run Smith-Waterman forward pass
    float partition;

    if (config.use_affine_gaps) {
        // Affine gaps: use flexible mode (allows gap transitions)
        smith_waterman_jax_affine_flexible<Backend>(workspace->similarity_matrix, L1, L2, sw_config,
                                                    workspace->dp_matrix, &partition);
    } else {
        // Regular gaps
        smith_waterman_jax_regular<Backend>(workspace->similarity_matrix, L1, L2, sw_config,
                                            workspace->dp_matrix, &partition);
    }

    // Step 4: Run Smith-Waterman backward pass (get posteriors)
    if (config.use_affine_gaps) {
        smith_waterman_jax_affine_flexible_backward<Backend>(
            workspace->dp_matrix, workspace->similarity_matrix, L1, L2, sw_config, partition,
            workspace->posteriors);
    } else {
        smith_waterman_jax_regular_backward<Backend>(workspace->dp_matrix,
                                                     workspace->similarity_matrix, L1, L2,
                                                     sw_config, partition, workspace->posteriors);
    }

    // Step 5: Convert posteriors to alignment columns
    int aligned_length;
    posteriors_to_alignment(workspace->posteriors, L1, L2, profile1, profile2,
                            workspace->alignment_columns, &aligned_length,
                            0.01f  // threshold
    );

    // Step 6: Merge profiles according to alignment
    Profile* merged = Profile::from_alignment(profile1, profile2, workspace->alignment_columns,
                                              aligned_length, arena);

    return merged;
}

/**
 * Align two profiles and merge into new TempProfile (for parallel computation).
 *
 * Same as align_and_merge_profiles but returns TempProfile allocated from
 * thread_arena instead of Profile from main arena.
 *
 * This allows parallel computation without mutex contention - each thread
 * computes into its own arena, then results are copied to main arena sequentially.
 */
template <typename Backend>
TempProfile* align_and_merge_profiles_temp(const Profile& profile1, const Profile& profile2,
                                           const MSAConfig& config, MSAWorkspace* workspace,
                                           GrowableArena* thread_arena) {
    int L1 = profile1.length;
    int L2 = profile2.length;

    MSA_PROFILE_SCOPE("align_and_merge_profiles_temp");

    // ========================================================================
    // INSTRUMENTATION: Phase 1 - Start timing
    // ========================================================================
    auto merge_start = std::chrono::high_resolution_clock::now();
    auto phase_start = merge_start;

    // Ensure workspace has sufficient capacity
    {
        MSA_PROFILE_SCOPE("workspace_ensure_capacity");
        int max_aligned = L1 + L2;  // Worst case: no matches, all gaps
        workspace->ensure_capacity(L1, L2, max_aligned);
    }

    // Step 1: Compute profile-aware similarity
    {
        MSA_PROFILE_SCOPE("phase1_similarity");
        compute_profile_similarity<Backend>(profile1, profile2, config.ecs_temperature,
                                            workspace->similarity_matrix);
    }

    // ========================================================================
    // INSTRUMENTATION: Phase 2 timing - Similarity computation complete
    // ========================================================================
    auto phase1_end = std::chrono::high_resolution_clock::now();
    auto phase1_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase_start);

    // Step 2: Configure Smith-Waterman
    SWConfig sw_config;
    sw_config.gap = config.gap_penalty;
    sw_config.gap_open = config.gap_open;
    sw_config.gap_extend = config.gap_extend;
    sw_config.temperature = config.temperature;
    sw_config.affine = config.use_affine_gaps;

    // Step 3: Run Smith-Waterman forward pass
    phase_start = std::chrono::high_resolution_clock::now();
    float partition;

    {
        MSA_PROFILE_SCOPE("phase2_sw_forward");
        if (config.use_affine_gaps) {
            // Affine gaps: use flexible mode (allows gap transitions)
            smith_waterman_jax_affine_flexible<Backend>(
                workspace->similarity_matrix, L1, L2, sw_config, workspace->dp_matrix, &partition);
        } else {
            // Regular gaps
            smith_waterman_jax_regular<Backend>(workspace->similarity_matrix, L1, L2, sw_config,
                                                workspace->dp_matrix, &partition);
        }
    }

    // ========================================================================
    // INSTRUMENTATION: Phase 2 timing - Forward pass complete
    // ========================================================================
    auto phase2_end = std::chrono::high_resolution_clock::now();
    auto phase2_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase2_end - phase_start);

    // Step 4: Run Smith-Waterman backward pass (get posteriors)
    phase_start = std::chrono::high_resolution_clock::now();

    {
        MSA_PROFILE_SCOPE("phase3_sw_backward");
        if (config.use_affine_gaps) {
            smith_waterman_jax_affine_flexible_backward<Backend>(
                workspace->dp_matrix, workspace->similarity_matrix, L1, L2, sw_config, partition,
                workspace->posteriors);
        } else {
            smith_waterman_jax_regular_backward<Backend>(
                workspace->dp_matrix, workspace->similarity_matrix, L1, L2, sw_config, partition,
                workspace->posteriors);
        }
    }

    // ========================================================================
    // INSTRUMENTATION: Phase 3 timing - Backward pass complete
    // ========================================================================
    auto phase3_end = std::chrono::high_resolution_clock::now();
    auto phase3_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase3_end - phase_start);

    // Step 5: Convert posteriors to alignment columns using Viterbi decoder
    phase_start = std::chrono::high_resolution_clock::now();
    int aligned_length;

    {
        MSA_PROFILE_SCOPE("phase4_decode_alignment");

        // Allocate scratch buffers from thread arena
        int max_path_length = L1 + L2;  // Worst case: all gaps
        AlignmentPair* alignment_path = thread_arena->allocate<AlignmentPair>(max_path_length);
        float* dp_score = thread_arena->allocate<float>((L1 + 1) * (L2 + 1));
        uint8_t* dp_traceback = thread_arena->allocate<uint8_t>((L1 + 1) * (L2 + 1));

        // Run Viterbi decoder to find MAP alignment path
        // Use config.decode_gap_penalty (in log-space)
        {
            MSA_PROFILE_SCOPE("viterbi_decode");
            int path_length = alignment_decode::decode_alignment<Backend>(
                workspace->posteriors, L1, L2,
                config.decode_gap_penalty,  // Log-probability penalty for gaps
                alignment_path, max_path_length, dp_score, dp_traceback);

            // Check for errors
            if (path_length < 0) {
                fprintf(stderr, "ERROR: decode_alignment failed (L1=%d, L2=%d)\n", L1, L2);
                abort();
            }

            // Convert alignment path to MSA columns
            {
                MSA_PROFILE_SCOPE("convert_to_columns");
                convert_alignment_path_to_columns(alignment_path, path_length, profile1, profile2,
                                                  workspace->alignment_columns, &aligned_length);
            }
        }
    }

    // ========================================================================
    // INSTRUMENTATION: Phase 4 timing - Posterior decoding complete
    // ========================================================================
    auto phase4_end = std::chrono::high_resolution_clock::now();
    auto phase4_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase4_end - phase_start);

    // Step 6: Merge profiles into TempProfile (thread-local allocation)
    phase_start = std::chrono::high_resolution_clock::now();
    TempProfile* merged;

    {
        MSA_PROFILE_SCOPE("phase5_profile_merge");
        merged = TempProfile::create_from_alignment(
            profile1, profile2, workspace->alignment_columns, aligned_length, thread_arena);
    }

    // ========================================================================
    // INSTRUMENTATION: Phase 5 timing - Profile merge complete
    // ========================================================================
    auto phase5_end = std::chrono::high_resolution_clock::now();
    auto phase5_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(phase5_end - phase_start);
    auto merge_end = phase5_end;
    auto merge_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(merge_end - merge_start);

    // ========================================================================
    // INSTRUMENTATION: Log expensive merges (threshold: 500ms)
    // ========================================================================
    long long total_ms = merge_duration.count();
    if (total_ms > 500) {
        // Get thread ID
        std::thread::id thread_id = std::this_thread::get_id();
        std::hash<std::thread::id> hasher;
        size_t thread_hash = hasher(thread_id) % 1000;  // Simplified thread ID

        // Phase 1: Basic merge info
        std::fprintf(stderr,
                     "[EXPENSIVE MERGE] thread=%zu, L1=%d, L2=%d, nseq1=%d, nseq2=%d, "
                     "aligned_len=%d, time=%lld ms\n",
                     thread_hash, L1, L2, profile1.num_sequences, profile2.num_sequences,
                     aligned_length, total_ms);

        // Phase 2: Per-phase breakdown
        std::fprintf(stderr,
                     "[MERGE PHASES] L1=%d, L2=%d, sim=%lld ms, fw=%lld ms, bw=%lld ms, "
                     "decode=%lld ms, merge=%lld ms, TOTAL=%lld ms\n",
                     L1, L2, (long long)phase1_duration.count(), (long long)phase2_duration.count(),
                     (long long)phase3_duration.count(), (long long)phase4_duration.count(),
                     (long long)phase5_duration.count(), total_ms);

        // Phase 3: Divergence diagnostics
        float avg_similarity = 0.0f;
        for (int i = 0; i < L1 * L2; i++) {
            avg_similarity += workspace->similarity_matrix[i];
        }
        avg_similarity /= (L1 * L2);

        int high_prob_count = 0;
        float max_posterior = 0.0f;
        for (int i = 0; i < L1 * L2; i++) {
            if (workspace->posteriors[i] > 0.01f) {
                high_prob_count++;
            }
            if (workspace->posteriors[i] > max_posterior) {
                max_posterior = workspace->posteriors[i];
            }
        }
        float sparsity = (float)high_prob_count / (L1 * L2);

        std::fprintf(stderr,
                     "[MERGE DIVERGENCE] L1=%d, L2=%d, avg_sim=%.4f, sparsity=%.4f, max_post=%.4f, "
                     "partition=%.2f\n",
                     L1, L2, avg_similarity, sparsity, max_posterior, partition);
    }

    return merged;
}

// ============================================================================
// Progressive MSA Algorithm
// ============================================================================

// Sequential implementation (used for small N or as fallback)
template <typename Backend>
static MSAResult progressive_msa_sequential(const SequenceCache& cache, const GuideTree& tree,
                                            const MSAConfig& config, GrowableArena* arena) {
    int N = cache.size();
    int hidden_dim = cache.hidden_dim();

    if (N == 0) {
        // Empty cache
        MSAResult result;
        result.alignment = nullptr;
        result.num_sequences = 0;
        result.aligned_length = 0;
        result.ecs = 0.0f;
        return result;
    }

    if (N == 1) {
        // Single sequence: trivial MSA
        const SequenceEmbeddings* seq = cache.get(0);
        Profile* profile =
            Profile::from_single_sequence(seq->get_embeddings(), seq->length, hidden_dim,
                                          0,  // seq_index
                                          arena);

        MSAResult result;
        result.alignment = profile;
        result.num_sequences = 1;
        result.aligned_length = profile->length;
        result.ecs = 1.0f;  // Perfect coherence for single sequence
        return result;
    }

    // Step 1: Compute reverse level-order traversal
    int* order = arena->allocate<int>(tree.num_nodes());
    int* level_offsets = arena->allocate<int>(tree.num_nodes() + 1);
    int depth;

    tree.compute_reverse_level_order(arena, order, level_offsets, &depth);

    // Step 2: Initialize profiles array (one per tree node)
    Profile** profiles = arena->allocate<Profile*>(tree.num_nodes());
    for (int i = 0; i < tree.num_nodes(); i++) {
        profiles[i] = nullptr;
    }

    // Step 3: Create workspace for alignment operations
    int max_initial_len = cache.max_length();
    MSAWorkspace* workspace = MSAWorkspace::create(
        max_initial_len, max_initial_len, max_initial_len * 2, config.use_affine_gaps, arena);

    // Step 4: Initialize leaf profiles from cached sequences
    for (int i = 0; i < N; i++) {
        const SequenceEmbeddings* seq = cache.get(i);
        profiles[i] = Profile::from_single_sequence(seq->get_embeddings(), seq->length, hidden_dim,
                                                    i,  // seq_index
                                                    arena);
    }

    // Progress tracking: N-1 total merges in binary tree
    const int total_merges = N - 1;
    int completed_merges = 0;

    // Step 5: Process each level bottom-up
    for (int level = 0; level < depth; level++) {
        int start = level_offsets[level];
        int end = level_offsets[level + 1];

        // Skip level 0 (leaves) - already initialized
        if (level == 0)
            continue;

        // Process all nodes at this level
        // NOTE: These can be parallelized in future GPU/SIMD implementation
        for (int i = start; i < end; i++) {
            int node_idx = order[i];
            if (node_idx < 0) {
                continue;
            }
            const GuideTreeNode& node = tree.node(node_idx);

            if (node.is_leaf) {
                // Leaf node (should not happen at level > 0, but handle gracefully)
                continue;
            }

            // Internal node: align and merge child profiles
            Profile* left = profiles[node.left_child];
            Profile* right = profiles[node.right_child];

            if (left == nullptr || right == nullptr) {
                // Child profile not initialized (should not happen)
                fprintf(stderr, "ERROR: Child profile missing for node %d\n", node_idx);
                continue;
            }

            // Align and merge
            profiles[node_idx] =
                align_and_merge_profiles<Backend>(*left, *right, config, workspace, arena);

            // Update progress
            completed_merges++;
            if (config.progress_callback) {
                config.progress_callback(completed_merges, total_merges);
            }

            // Child profiles no longer needed; destroy to release heap allocations
            Profile::destroy(left);
            Profile::destroy(right);
            profiles[node.left_child] = nullptr;
            profiles[node.right_child] = nullptr;
        }
    }

    // Step 6: Extract final MSA from root
    Profile* final_msa = profiles[tree.root_index()];

    if (final_msa == nullptr) {
        fprintf(stderr, "ERROR: Root profile is null\n");
        MSAResult result;
        result.alignment = nullptr;
        result.num_sequences = 0;
        result.aligned_length = 0;
        result.ecs = 0.0f;
        result.cache = &cache;
        MSAWorkspace::destroy(workspace);
        return result;
    }

    // Step 7: Compute ECS quality metric
    float ecs = final_msa->compute_ecs();

    MSAResult result;
    result.alignment = final_msa;
    result.num_sequences = final_msa->num_sequences;
    result.aligned_length = final_msa->length;
    result.ecs = ecs;
    result.cache = &cache;

    MSAWorkspace::destroy(workspace);

    return result;
}

// Parallel implementation using ThreadPool (default for N >= 10)
template <typename Backend>
static MSAResult progressive_msa_parallel(const SequenceCache& cache, const GuideTree& tree,
                                          const MSAConfig& config, GrowableArena* arena) {
    MSA_PROFILE_SCOPE("progressive_msa_parallel");

    int N = cache.size();
    int hidden_dim = cache.hidden_dim();

    if (N == 0) {
        MSAResult result;
        result.alignment = nullptr;
        result.num_sequences = 0;
        result.aligned_length = 0;
        result.ecs = 0.0f;
        return result;
    }

    if (N == 1) {
        const SequenceEmbeddings* seq = cache.get(0);
        Profile* profile =
            Profile::from_single_sequence(seq->get_embeddings(), seq->length, hidden_dim, 0, arena);

        MSAResult result;
        result.alignment = profile;
        result.num_sequences = 1;
        result.aligned_length = profile->length;
        result.ecs = 1.0f;
        result.cache = &cache;
        return result;
    }

    // Step 1: Compute reverse level-order traversal
    int* order;
    int* level_offsets;
    int depth;

    {
        MSA_PROFILE_SCOPE("compute_level_order");
        order = arena->allocate<int>(tree.num_nodes());
        level_offsets = arena->allocate<int>(tree.num_nodes() + 1);
        tree.compute_reverse_level_order(arena, order, level_offsets, &depth);
    }

    // Step 2: Initialize profiles array (one per tree node)
    Profile** profiles = arena->allocate<Profile*>(tree.num_nodes());
    for (int i = 0; i < tree.num_nodes(); i++) {
        profiles[i] = nullptr;
    }

    size_t requested_threads =
        config.thread_count > 0 ? static_cast<size_t>(config.thread_count) : 0;

    // Step 3: Create ThreadPool with per-thread arenas
    // Arena size: 100 MB per thread for temporary allocations
    // Increased from 50 MB to handle large profile-profile merges (N=45+)
    pfalign::threading::ThreadPool pool(requested_threads, 100);

    // DEBUG: Log actual thread count
    std::fprintf(stderr, "[DEBUG] Progressive MSA: N=%d, threads=%zu\n", N, pool.num_threads());

    // Step 4: TWO-PHASE leaf profile initialization
    // Phase 1 (Parallel): Create TempProfiles in thread-local arenas
    std::vector<TempProfile*> temp_leaves(N, nullptr);

    {
        MSA_PROFILE_SCOPE("leaf_init_phase1_parallel");
        MSA_CPU_TRACKER("leaf_init_parallel", pool.num_threads());

        pool.parallel_for(
            static_cast<size_t>(N),
            [&](int tid, size_t begin, size_t end, pfalign::memory::GrowableArena& thread_arena) {
                (void)tid;  // Unused

                for (size_t i = begin; i < end; i++) {
                    const SequenceEmbeddings* seq = cache.get(static_cast<int>(i));

                    // Allocate TempProfile from thread arena
                    TempProfile* temp = thread_arena.allocate<TempProfile>(1);
                    new (temp) TempProfile();

                    temp->length = seq->length;
                    temp->num_sequences = 1;
                    temp->hidden_dim = hidden_dim;

                    // Allocate arrays from thread arena
                    temp->embeddings = thread_arena.allocate<float>(seq->length * hidden_dim);
                    temp->weights = thread_arena.allocate<float>(seq->length);
                    temp->sum_norm = thread_arena.allocate<float>(seq->length * hidden_dim);
                    temp->gap_counts = thread_arena.allocate<int>(seq->length);

                    // Copy embeddings
                    std::memcpy(temp->embeddings, seq->get_embeddings(),
                                seq->length * hidden_dim * sizeof(float));

                    // Initialize weights (all 1.0 for single sequence)
                    for (int j = 0; j < seq->length; j++) {
                        temp->weights[j] = 1.0f;
                        temp->gap_counts[j] = 0;
                    }

                    // Compute normalized embeddings for sum_norm (NOT raw embeddings!)
                    // BUG FIX: sum_norm must contain unit vectors, not raw embeddings
                    constexpr float eps = 1e-8f;
                    for (int pos = 0; pos < seq->length; pos++) {
                        const float* emb = temp->embeddings + pos * hidden_dim;
                        float* sum_norm_pos = temp->sum_norm + pos * hidden_dim;

                        // Compute ||emb||
                        float norm_sq = 0.0f;
                        for (int d = 0; d < hidden_dim; d++) {
                            norm_sq += emb[d] * emb[d];
                        }

                        float norm = std::sqrt(norm_sq);
                        if (norm > eps) {
                            // Normalize: sum_norm = emb / ||emb||
                            for (int d = 0; d < hidden_dim; d++) {
                                sum_norm_pos[d] = emb[d] / norm;
                            }
                        } else {
                            // Zero embedding - leave sum_norm as zero
                            std::memset(sum_norm_pos, 0, hidden_dim * sizeof(float));
                        }
                    }

                    // Initialize columns (single residue per column)
                    temp->columns.reserve(seq->length);
                    for (int pos = 0; pos < seq->length; pos++) {
                        AlignmentColumn col;
                        col.positions.resize(1);
                        col.positions[0] = {static_cast<int>(i), pos};
                        temp->columns.push_back(col);
                    }

                    // Single sequence index
                    temp->seq_indices.push_back(static_cast<int>(i));

                    // Store in temp array (no lock needed - disjoint indices)
                    temp_leaves[i] = temp;
                }
            });
    }

    // Phase 2 (Sequential): Convert TempProfiles to Profiles in main arena
    {
        MSA_PROFILE_SCOPE("leaf_init_phase2_sequential");
        for (int i = 0; i < N; i++) {
            profiles[i] = temp_leaves[i]->to_profile(arena);
        }
    }

    // Step 5: TWO-PHASE level-by-level processing
    int max_initial_len = cache.max_length();

    // Progress tracking: N-1 total merges in binary tree
    const int total_merges = N - 1;
    std::atomic<int> completed_merges{0};

    std::fprintf(stderr, "[DEBUG] Tree depth: %d levels\n", depth);

    for (int level = 0; level < depth; level++) {
        int start = level_offsets[level];
        int end = level_offsets[level + 1];

        // Skip level 0 (leaves) - already initialized
        if (level == 0)
            continue;

        int num_nodes = end - start;

        auto level_start = std::chrono::high_resolution_clock::now();

        // Temporary storage for this level's TempProfiles
        std::vector<TempProfile*> temp_profiles(num_nodes, nullptr);

        // PHASE 1 (Parallel): Compute alignments into thread-local TempProfiles
        auto phase1_start = std::chrono::high_resolution_clock::now();

        {
            MSA_PROFILE_SCOPE("level_phase1_parallel_compute");
            MSA_CPU_TRACKER("level_parallel_compute", pool.num_threads());

            pool.parallel_for(
                static_cast<size_t>(num_nodes), [&](int tid, size_t begin, size_t end_range,
                                                    pfalign::memory::GrowableArena& thread_arena) {
                    (void)tid;  // Unused

                    // Create per-thread workspace
                    MSAWorkspace* workspace =
                        MSAWorkspace::create(max_initial_len, max_initial_len, max_initial_len * 2,
                                             config.use_affine_gaps, &thread_arena);

                    for (size_t i = begin; i < end_range; i++) {
                        int node_idx = order[start + i];
                        if (node_idx < 0) {
                            continue;
                        }
                        const GuideTreeNode& node = tree.node(node_idx);

                        if (node.is_leaf) {
                            continue;
                        }

                        // Read child profiles (SAFE: previous level is complete)
                        Profile* left = profiles[node.left_child];
                        Profile* right = profiles[node.right_child];

                        if (left == nullptr || right == nullptr) {
                            fprintf(stderr, "ERROR: Child profile missing for node %d\n", node_idx);
                            continue;
                        }

                        // Compute alignment into TempProfile (NO LOCK - parallel computation)
                        TempProfile* temp = align_and_merge_profiles_temp<Backend>(
                            *left, *right, config, workspace,
                            &thread_arena  // Use thread-local arena
                        );

                        // Update progress (thread-safe)
                        if (config.progress_callback) {
                            int count = completed_merges.fetch_add(1) + 1;
                            config.progress_callback(count, total_merges);
                        }

                        // Store in temp array (no lock needed - disjoint indices)
                        temp_profiles[i] = temp;
                    }

                    // Clean up thread workspace
                    MSAWorkspace::destroy(workspace);
                });
        }

        auto phase1_end = std::chrono::high_resolution_clock::now();
        auto phase1_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase1_start);

        // PHASE 2 (Sequential): Convert TempProfiles to Profiles and clean up children
        auto phase2_start = std::chrono::high_resolution_clock::now();

        {
            MSA_PROFILE_SCOPE("level_phase2_sequential_convert");

            for (size_t i = 0; i < static_cast<size_t>(num_nodes); i++) {
                int node_idx = order[start + i];
                if (node_idx < 0 || temp_profiles[i] == nullptr) {
                    continue;
                }

                const GuideTreeNode& node = tree.node(node_idx);

                // Convert TempProfile to Profile (SHORT LOCK - just allocation)
                profiles[node_idx] = temp_profiles[i]->to_profile(arena);

                // DEBUG: Check if weights are correct immediately after conversion
                if (profiles[node_idx]->length > 100 && !tree.node(node_idx).is_leaf) {
                    static std::atomic<int> post_convert_check{0};
                    if (post_convert_check.fetch_add(1) < 3) {
                        int mid = profiles[node_idx]->length / 2;
                        fprintf(stderr,
                                "[POST-CONVERT] Node %d (level %d, len=%d): weight[%d]=%.2e\n",
                                node_idx, level, profiles[node_idx]->length, mid,
                                profiles[node_idx]->weights[mid]);
                        fflush(stderr);
                    }
                }

                // Child profiles no longer needed; destroy to release heap allocations
                Profile* left = profiles[node.left_child];
                Profile* right = profiles[node.right_child];

                if (left != nullptr) {
                    Profile::destroy(left);
                    profiles[node.left_child] = nullptr;
                }
                if (right != nullptr) {
                    Profile::destroy(right);
                    profiles[node.right_child] = nullptr;
                }
            }
        }

        // DEBUG: Check all newly created profiles before arena reset
        // Only check levels near the root (last few levels)
        if (level >= depth - 3) {
            for (size_t i = 0; i < static_cast<size_t>(num_nodes); i++) {
                int node_idx = order[start + i];
                if (profiles[node_idx] != nullptr && profiles[node_idx]->length > 100) {
                    int mid = profiles[node_idx]->length / 2;
                    fprintf(stderr, "[PRE-RESET] Level %d, node %d (len=%d): weight[%d]=%.2e\n",
                            level, node_idx, profiles[node_idx]->length, mid,
                            profiles[node_idx]->weights[mid]);
                }
            }
            fflush(stderr);
        }

        // Reset thread arenas after each level to prevent cumulative waste
        // This is safe because Phase 2 has converted all TempProfiles to Profiles in main arena
        pool.reset_arenas();

        // DEBUG: Check all profiles after arena reset
        if (level >= depth - 3) {
            for (size_t i = 0; i < static_cast<size_t>(num_nodes); i++) {
                int node_idx = order[start + i];
                if (profiles[node_idx] != nullptr && profiles[node_idx]->length > 100) {
                    int mid = profiles[node_idx]->length / 2;
                    fprintf(stderr, "[POST-RESET] Level %d, node %d (len=%d): weight[%d]=%.2e\n",
                            level, node_idx, profiles[node_idx]->length, mid,
                            profiles[node_idx]->weights[mid]);
                }
            }
            fflush(stderr);
        }

        auto phase2_end = std::chrono::high_resolution_clock::now();
        auto phase2_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(phase2_end - phase2_start);

        // DEBUG: Log level timing and profile stats
        auto level_end = std::chrono::high_resolution_clock::now();
        auto level_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(level_end - level_start);

        // Calculate average profile length at this level
        int total_length = 0;
        int valid_profiles = 0;
        for (size_t i = 0; i < static_cast<size_t>(num_nodes); i++) {
            int node_idx = order[start + i];
            if (node_idx >= 0 && profiles[node_idx] != nullptr) {
                total_length += profiles[node_idx]->length;
                valid_profiles++;
            }
        }
        int avg_length = (valid_profiles > 0) ? (total_length / valid_profiles) : 0;

        std::fprintf(stderr,
                     "[DEBUG] Level %d: %d merges, Phase1=%lld ms, Phase2=%lld ms, total=%lld ms, "
                     "avg_len=%d\n",
                     level, num_nodes, (long long)phase1_duration.count(),
                     (long long)phase2_duration.count(), (long long)level_duration.count(),
                     avg_length);
    }

    // Step 6: Extract final MSA from root
    Profile* final_msa = profiles[tree.root_index()];

    if (final_msa == nullptr) {
        fprintf(stderr, "ERROR: Root profile is null\n");
        MSAResult result;
        result.alignment = nullptr;
        result.num_sequences = 0;
        result.aligned_length = 0;
        result.ecs = 0.0f;
        result.cache = &cache;
        return result;
    }

    // Step 7: Compute ECS quality metric
    float ecs;

    {
        MSA_PROFILE_SCOPE("compute_ecs");

        // DEBUG: Check final_msa weights immediately before ECS computation
        if (final_msa->length > 100) {
            int mid = final_msa->length / 2;
            fprintf(stderr, "[BEFORE ECS] final_msa=%p, weights=%p, len=%d\n",
                    static_cast<void*>(final_msa), static_cast<void*>(final_msa->weights),
                    final_msa->length);
            fprintf(stderr, "[BEFORE ECS] weight[0]=%.2e, weight[%d]=%.2e, weight[%d]=%.2e\n",
                    final_msa->weights[0], mid, final_msa->weights[mid], final_msa->length - 1,
                    final_msa->weights[final_msa->length - 1]);
            fflush(stderr);
        }

        ecs = final_msa->compute_ecs();
    }

    MSAResult result;
    result.alignment = final_msa;
    result.num_sequences = final_msa->num_sequences;
    result.aligned_length = final_msa->length;
    result.ecs = ecs;
    result.cache = &cache;

#if ENABLE_PROFILING
    // Print profiling reports
    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "  Progressive MSA Profiling Report\n");
    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "N=%d sequences, aligned_length=%d, ECS=%.4f\n", N, result.aligned_length, ecs);
    fprintf(stderr, "\n");

    // Print timing breakdown
    pfalign::ProfilingData::instance().print_report();

    // Print CPU utilization
    pfalign::CPUUtilizationData::instance().print_report();

    fprintf(stderr, "================================================================\n");
    fprintf(stderr, "\n");
#endif

    return result;
}

// Public API with auto-dispatch (sequential for small N, parallel for large N)
template <typename Backend>
MSAResult progressive_msa(const SequenceCache& cache, const GuideTree& tree,
                          const MSAConfig& config, GrowableArena* arena) {
    int N = cache.size();

    // Use parallel implementation for N >= 10
    // For smaller N, sequential avoids thread overhead
    if (N >= 10) {
        return progressive_msa_parallel<Backend>(cache, tree, config, arena);
    } else {
        return progressive_msa_sequential<Backend>(cache, tree, config, arena);
    }
}

bool MSAResult::write_fasta(const std::string& output_path) const {
    if (alignment == nullptr || alignment->columns == nullptr || cache == nullptr) {
        return false;
    }
    if (num_sequences <= 0) {
        return false;
    }

    std::ofstream out(output_path);
    if (!out.is_open()) {
        return false;
    }

    if (static_cast<int>(alignment->seq_indices.size()) < num_sequences) {
        return false;
    }

    std::vector<const SequenceEmbeddings*> seq_entries;
    seq_entries.reserve(num_sequences);
    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const int cache_id = alignment->seq_indices[seq_idx];
        const SequenceEmbeddings* seq = cache->get(cache_id);
        if (seq == nullptr) {
            return false;
        }
        seq_entries.push_back(seq);
    }

    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const SequenceEmbeddings* seq = seq_entries[seq_idx];
        const std::string label = make_sequence_label(seq, seq_idx);

        out << ">" << label << " | Length: " << seq->length << "\n";
        std::string aligned_seq;
        aligned_seq.reserve(aligned_length);

        for (int col = 0; col < aligned_length; ++col) {
            const AlignmentPosition& pos = alignment->columns[col].positions[seq_idx];
            if (pos.is_gap()) {
                aligned_seq.push_back('-');
                continue;
            }

            char residue = 'X';
            if (!seq->sequence.empty() && pos.pos >= 0 &&
                pos.pos < static_cast<int>(seq->sequence.size())) {
                residue = seq->sequence[pos.pos];
            }
            aligned_seq.push_back(residue);
        }

        write_wrapped_sequence(out, aligned_seq);
    }

    return true;
}

int MSAResult::select_reference_sequence() const {
    if (alignment == nullptr || alignment->columns == nullptr || num_sequences <= 0) {
        return 0;
    }

    int best_idx = 0;
    int best_non_gaps = -1;

    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        int non_gaps = 0;
        for (int col = 0; col < aligned_length; ++col) {
            if (!alignment->columns[col].positions[seq_idx].is_gap()) {
                non_gaps++;
            }
        }

        if (non_gaps > best_non_gaps) {
            best_non_gaps = non_gaps;
            best_idx = seq_idx;
        }
    }

    return best_idx;
}

bool MSAResult::write_superposed_pdb(const std::string& output_path, int reference_seq_idx,
                                     pfalign::memory::GrowableArena* arena) const {
    (void)arena;

    if (alignment == nullptr || alignment->columns == nullptr || cache == nullptr) {
        return false;
    }
    if (num_sequences <= 0) {
        return false;
    }
    if (static_cast<int>(alignment->seq_indices.size()) < num_sequences) {
        return false;
    }

    if (reference_seq_idx < 0 || reference_seq_idx >= num_sequences) {
        reference_seq_idx = select_reference_sequence();
    }

    std::vector<const SequenceEmbeddings*> seq_entries;
    seq_entries.reserve(num_sequences);
    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const int cache_id = alignment->seq_indices[seq_idx];
        const SequenceEmbeddings* seq = cache->get(cache_id);
        if (seq == nullptr || seq->coords == nullptr) {
            return false;
        }
        seq_entries.push_back(seq);
    }

    const SequenceEmbeddings* ref_seq = seq_entries[reference_seq_idx];
    if (ref_seq == nullptr || ref_seq->coords == nullptr) {
        return false;
    }

    std::vector<std::vector<uint8_t>> aligned_flags(num_sequences);
    std::vector<SuperposeWorkItem> work_items(num_sequences);
    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        aligned_flags[seq_idx].assign(seq_entries[seq_idx]->length, 0);
    }

    for (int col = 0; col < aligned_length; ++col) {
        const AlignmentPosition& ref_pos = alignment->columns[col].positions[reference_seq_idx];
        if (ref_pos.is_gap() || ref_pos.pos < 0 || ref_pos.pos >= ref_seq->length) {
            continue;
        }
        aligned_flags[reference_seq_idx][ref_pos.pos] = 1;

        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
            if (seq_idx == reference_seq_idx) {
                continue;
            }

            const AlignmentPosition& cur_pos = alignment->columns[col].positions[seq_idx];
            if (cur_pos.is_gap() || cur_pos.pos < 0) {
                continue;
            }

            const SequenceEmbeddings* seq = seq_entries[seq_idx];
            if (cur_pos.pos >= seq->length) {
                continue;
            }

            aligned_flags[seq_idx][cur_pos.pos] = 1;
            work_items[seq_idx].reference_positions.push_back(ref_pos.pos);
            work_items[seq_idx].sequence_positions.push_back(cur_pos.pos);
        }
    }

    std::vector<SuperposedSequenceScratch> scratch(num_sequences);
    pfalign::threading::ThreadPool pool(0, 32);
    pool.parallel_for(
        static_cast<size_t>(num_sequences),
        [&](int /*tid*/, size_t begin, size_t end, pfalign::memory::GrowableArena& thread_arena) {
            (void)thread_arena;
            for (size_t idx = begin; idx < end; ++idx) {
                if (static_cast<int>(idx) == reference_seq_idx) {
                    continue;
                }

                const SequenceEmbeddings* seq = seq_entries[idx];
                auto& seq_scratch = scratch[idx];
                const auto& work = work_items[idx];
                seq_scratch.aligned_pairs = static_cast<int>(work.reference_positions.size());

                if (seq_scratch.aligned_pairs < 3) {
                    continue;
                }

                std::vector<float> ref_ca;
                std::vector<float> cur_ca;
                ref_ca.reserve(static_cast<size_t>(work.reference_positions.size()) * 3);
                cur_ca.reserve(static_cast<size_t>(work.reference_positions.size()) * 3);

                for (size_t pair_idx = 0; pair_idx < work.reference_positions.size(); ++pair_idx) {
                    const int ref_pos = work.reference_positions[pair_idx];
                    const int cur_pos = work.sequence_positions[pair_idx];
                    const float* ref_atom = ref_seq->coords + ref_pos * 12 + 3;
                    const float* cur_atom = seq->coords + cur_pos * 12 + 3;
                    for (int dim = 0; dim < 3; ++dim) {
                        ref_ca.push_back(ref_atom[dim]);
                        cur_ca.push_back(cur_atom[dim]);
                    }
                }

                float R[9];
                float t[3];
                float rmsd = 0.0f;

                kabsch::kabsch_align<ScalarBackend>(cur_ca.data(), ref_ca.data(),
                                                    seq_scratch.aligned_pairs, R, t, &rmsd);

                seq_scratch.transformed_coords.resize(static_cast<size_t>(seq->length) * 12);
                kabsch::apply_transformation<ScalarBackend>(
                    R, t, seq->coords, seq_scratch.transformed_coords.data(), seq->length);

                seq_scratch.rmsd = rmsd;
                seq_scratch.has_transform = true;
            }
        });

    std::ofstream out(output_path);
    if (!out.is_open()) {
        return false;
    }

    const std::string ref_label = make_sequence_label(ref_seq, reference_seq_idx);

    out << "HEADER    PROTEIN-FORGE MSA SUPERPOSITION\n";
    out << "REMARK   1 REFERENCE INDEX: " << reference_seq_idx << "\n";
    out << "REMARK   1 REFERENCE ID: " << ref_label << "\n";
    out << "REMARK   1 TOTAL SEQUENCES: " << num_sequences << "\n";
    out << "REMARK   1 ALIGNED COLUMNS: " << aligned_length << "\n";
    out << "REMARK   1 ECS: " << std::fixed << std::setprecision(5) << ecs << "\n";
    out << "REMARK   1 NOTE: B-FACTORS MARK ALIGNED RESIDUES (1.0 = ALIGNED)\n";

    int model_counter = 1;
    const int ref_aligned =
        std::count(aligned_flags[reference_seq_idx].begin(), aligned_flags[reference_seq_idx].end(),
                   static_cast<uint8_t>(1));

    write_msa_model(out, model_counter++, chain_id_for_index(reference_seq_idx), ref_label,
                    ref_seq->coords, ref_seq->length, aligned_flags[reference_seq_idx], 0.0f,
                    ref_aligned);

    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        if (seq_idx == reference_seq_idx) {
            continue;
        }

        const SequenceEmbeddings* seq = seq_entries[seq_idx];
        const float* coords_ptr = nullptr;

        if (scratch[seq_idx].has_transform) {
            coords_ptr = scratch[seq_idx].transformed_coords.data();
        } else {
            coords_ptr = seq->coords;
        }

        if (coords_ptr == nullptr) {
            return false;
        }

        write_msa_model(out, model_counter++, chain_id_for_index(seq_idx),
                        make_sequence_label(seq, seq_idx), coords_ptr, seq->length,
                        aligned_flags[seq_idx], scratch[seq_idx].rmsd,
                        scratch[seq_idx].aligned_pairs);
    }

    out << "END\n";
    return true;
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicitly instantiate for ScalarBackend
template MSAResult progressive_msa<pfalign::ScalarBackend>(const SequenceCache& cache,
                                                           const GuideTree& tree,
                                                           const MSAConfig& config,
                                                           GrowableArena* arena);

template Profile* align_and_merge_profiles<pfalign::ScalarBackend>(const Profile& profile1,
                                                                   const Profile& profile2,
                                                                   const MSAConfig& config,
                                                                   MSAWorkspace* workspace,
                                                                   GrowableArena* arena);

template void compute_profile_similarity<pfalign::ScalarBackend>(const Profile& profile1,
                                                                 const Profile& profile2,
                                                                 float ecs_temp, float* out_matrix);

}  // namespace msa
}  // namespace pfalign
