#include "profile.h"
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <atomic>

namespace pfalign {
namespace msa {

// AlignmentColumn implementation
int AlignmentColumn::count_residues() const {
    int count = 0;
    for (const auto& pos : positions) {
        if (!pos.is_gap()) {
            ++count;
        }
    }
    return count;
}

float AlignmentColumn::gap_fraction() const {
    if (positions.empty()) {
        return 0.0f;
    }
    int gaps = static_cast<int>(positions.size()) - count_residues();
    return static_cast<float>(gaps) / positions.size();
}

// Profile implementation
Profile::Profile(int len, int num_seqs, int hidden, pfalign::memory::GrowableArena* a)
    : length(len), num_sequences(num_seqs), hidden_dim(hidden), arena(a) {
    // Allocate memory from arena using template method
    embeddings = arena->allocate<float>(length * hidden_dim);
    weights = arena->allocate<float>(length);
    sum_norm = arena->allocate<float>(length * hidden_dim);
    gap_counts = arena->allocate<int>(length);
    columns = arena->allocate<AlignmentColumn>(length);

    // Initialize arrays
    std::memset(embeddings, 0, static_cast<size_t>(length * hidden_dim) * sizeof(float));
    std::fill_n(weights, length, 1.0f);  // Default weight = 1.0
    std::memset(sum_norm, 0, static_cast<size_t>(length * hidden_dim) * sizeof(float));
    std::memset(gap_counts, 0, static_cast<size_t>(length) * sizeof(int));

    // Placement new for columns
    for (int i = 0; i < length; ++i) {
        new (&columns[i]) AlignmentColumn();
    }
}

Profile::~Profile() {
    if (columns != nullptr) {
        for (int i = 0; i < length; ++i) {
            columns[i].~AlignmentColumn();
        }
    }
}

void Profile::destroy(Profile* profile) {
    if (profile == nullptr) {
        return;
    }
    profile->~Profile();
    profile->length = 0;
    profile->columns = nullptr;
}

Profile* Profile::from_single_sequence(const float* seq_embeddings, int length, int hidden_dim,
                                       int seq_index, pfalign::memory::GrowableArena* arena) {
    // Create profile with single sequence (using placement new)
    Profile* profile = arena->allocate<Profile>(1);
    new (profile) Profile(length, 1, hidden_dim, arena);

    // Copy embeddings and compute normalized sums for ECS
    constexpr float eps = 1e-8f;
    for (int i = 0; i < length; ++i) {
        const float* emb = seq_embeddings + i * hidden_dim;
        float* prof_emb = profile->get_embedding_mut(i);
        float* prof_sum_norm = profile->get_sum_norm_mut(i);

        // Copy embedding
        std::memcpy(prof_emb, emb, static_cast<size_t>(hidden_dim) * sizeof(float));

        // Compute normalized embedding for sum_norm
        // For single sequence: sum_norm = emb / ||emb|| * 1.0
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            norm_sq += emb[d] * emb[d];
        }

        float norm = std::sqrt(norm_sq);
        if (norm > eps) {
            for (int d = 0; d < hidden_dim; ++d) {
                prof_sum_norm[d] = emb[d] / norm;
            }
        }
        // else: sum_norm stays zero (from constructor)

        // DEBUG: Verify normalization for first position
        if (i == 0) {
            float check_norm_sq = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                check_norm_sq += prof_sum_norm[d] * prof_sum_norm[d];
            }
            // ALWAYS print to verify (remove conditional)
            static std::atomic<int> leaf_count{0};
            int count = leaf_count.fetch_add(1);
            if (count < 5) {
                fprintf(stderr,
                        "[LEAF DEBUG] Seq %d: ||emb||=%.6f, ||sum_norm||=%.6f, weight[0]=%.2f\n",
                        seq_index, static_cast<double>(norm),
                        std::sqrt(static_cast<double>(check_norm_sq)),
                        static_cast<double>(profile->weights[0]));
                fflush(stderr);
            }
        }
    }

    // Set up alignment columns (no gaps for single sequence)
    for (int i = 0; i < length; ++i) {
        profile->columns[i].positions.resize(1);
        profile->columns[i].positions[0].seq_idx = seq_index;
        profile->columns[i].positions[0].pos = i;
    }

    // Add sequence index
    profile->seq_indices.push_back(seq_index);

    return profile;
}

Profile* Profile::from_alignment(const Profile& profile1, const Profile& profile2,
                                 const AlignmentColumn* alignment, int aligned_length,
                                 pfalign::memory::GrowableArena* arena) {
    // Create merged profile (using placement new)
    int total_seqs = profile1.num_sequences + profile2.num_sequences;
    int hidden_dim = profile1.hidden_dim;

    Profile* merged = arena->allocate<Profile>(1);
    new (merged) Profile(aligned_length, total_seqs, hidden_dim, arena);

    // Merge sequence indices
    merged->seq_indices = profile1.seq_indices;
    merged->seq_indices.insert(merged->seq_indices.end(), profile2.seq_indices.begin(),
                               profile2.seq_indices.end());

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

    auto accumulate_from_profile = [&](const Profile& source_profile, int source_col,
                                       float* merged_emb, float* merged_sum_norm,
                                       float& total_weight) {
        if (source_col < 0 || source_col >= source_profile.length) {
            fprintf(stderr, "ERROR: source_col=%d out of range (length=%d)\n", source_col,
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
            merged_emb[d] += source_emb[d] * weight;
            merged_sum_norm[d] += source_sum_norm[d];
        }

        total_weight += weight;
    };

    // Process each column in the alignment
    constexpr float eps = 1e-8f;
    int next_profile1_col = 0;
    int next_profile2_col = 0;

    for (int col = 0; col < aligned_length; ++col) {
        const AlignmentColumn& align_col = alignment[col];

        // Resize column to hold all sequences
        merged->columns[col].positions.resize(static_cast<size_t>(total_seqs));

        // Initialize accumulators
        float* merged_emb = merged->get_embedding_mut(col);
        float* merged_sum_norm = merged->get_sum_norm_mut(col);
        std::memset(merged_emb, 0, static_cast<size_t>(hidden_dim) * sizeof(float));
        std::memset(merged_sum_norm, 0, static_cast<size_t>(hidden_dim) * sizeof(float));

        float total_weight = 0.0f;  // Sum of child weights (residue counts)
        int gap_count = 0;

        // Preserve alignment structure
        for (int seq_slot = 0; seq_slot < total_seqs; ++seq_slot) {
            const AlignmentPosition& pos = align_col.positions[seq_slot];
            merged->columns[col].positions[seq_slot] = pos;
            if (pos.is_gap()) {
                gap_count++;
            }
        }

        bool uses_profile1 = column_has_residue(align_col, 0, profile1.num_sequences);
        bool uses_profile2 = column_has_residue(align_col, profile1.num_sequences, total_seqs);

        if (uses_profile1) {
            if (next_profile1_col < profile1.length) {
                accumulate_from_profile(profile1, next_profile1_col, merged_emb, merged_sum_norm,
                                        total_weight);
            } else {
                fprintf(stderr, "ERROR: profile1 column index overflow (idx=%d, len=%d)\n",
                        next_profile1_col, profile1.length);
            }
            next_profile1_col++;
        }

        if (uses_profile2) {
            if (next_profile2_col < profile2.length) {
                accumulate_from_profile(profile2, next_profile2_col, merged_emb, merged_sum_norm,
                                        total_weight);
            } else {
                fprintf(stderr, "ERROR: profile2 column index overflow (idx=%d, len=%d)\n",
                        next_profile2_col, profile2.length);
            }
            next_profile2_col++;
        }

        // Finalize column: divide by total weight
        if (total_weight > eps) {
            float scale = 1.0f / total_weight;
            for (int d = 0; d < hidden_dim; ++d) {
                merged_emb[d] *= scale;
            }
            // sum_norm doesn't get divided - it accumulates across merges
        }

        // Set gap count and weight
        merged->gap_counts[col] = gap_count;
        merged->weights[col] = total_weight;  // Total residue count

        // DEBUG: Verify sum_norm accumulation for middle column in all merges
        if (col == aligned_length / 2 && total_seqs >= 2) {
            float sum_norm_magnitude_sq = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                sum_norm_magnitude_sq += merged_sum_norm[d] * merged_sum_norm[d];
            }
            float prof1_weight = (next_profile1_col > 0 && next_profile1_col - 1 < profile1.length)
                                     ? profile1.weights[next_profile1_col - 1]
                                     : 0.0f;
            float prof2_weight = (next_profile2_col > 0 && next_profile2_col - 1 < profile2.length)
                                     ? profile2.weights[next_profile2_col - 1]
                                     : 0.0f;
            fprintf(stderr,
                    "[MERGE DEBUG] Col %d: seqs=%d, weight=%.2f (p1=%.2f,p2=%.2f), gaps=%d, "
                    "||sum_norm||=%.4f (max=%.2f)\n",
                    col, total_seqs, static_cast<double>(total_weight),
                    static_cast<double>(prof1_weight), static_cast<double>(prof2_weight), gap_count,
                    std::sqrt(static_cast<double>(sum_norm_magnitude_sq)),
                    static_cast<double>(total_weight));
            fflush(stderr);
        }
    }

    return merged;
}

// ============================================================================
// Embedding Coherence Score (ECS) Implementation
// ============================================================================

float Profile::compute_column_coherence(int col) const {
    // Get residue count for this column
    float n = weights[col];

    // Need at least 2 residues to compute coherence
    if (n < 2.0f) {
        return 0.0f;
    }

    // Compute ||sum_norm||^2
    const float* sum_norm_col = get_sum_norm(col);
    float sum_norm_sq = 0.0f;
    for (int d = 0; d < hidden_dim; ++d) {
        sum_norm_sq += sum_norm_col[d] * sum_norm_col[d];
    }

    // Coherence formula: (||u||^2 - n) / (n * (n - 1))
    // This is the average pairwise cosine similarity
    float coherence = (sum_norm_sq - n) / (n * (n - 1.0f));

    return coherence;
}

float Profile::compute_ecs() const {
    float weighted_coherence = 0.0f;
    float total_weight = 0.0f;

    // DEBUG: Sample a few columns for detailed output
    bool debug_mode = (length > 100);  // Only debug for large alignments
    int sample_cols[5] = {0, length / 4, length / 2, 3 * length / 4, length - 1};

    // DEBUG: Check weights array immediately at ECS entry
    if (debug_mode) {
        static std::atomic<int> ecs_entry_count{0};
        if (ecs_entry_count.fetch_add(1) < 3) {
            int mid = length / 2;
            fprintf(stderr, "[ECS ENTRY] Profile=%p, weights=%p, len=%d\n",
                    static_cast<const void*>(this), static_cast<const void*>(weights), length);
            fprintf(stderr, "[ECS ENTRY] weights[0]=%.2e, weights[%d]=%.2e, weights[%d]=%.2e\n",
                    static_cast<double>(weights[0]), mid, static_cast<double>(weights[mid]),
                    length - 1, static_cast<double>(weights[length - 1]));
            fflush(stderr);
        }
    }

    for (int col = 0; col < length; ++col) {
        float n = weights[col];
        if (n >= 2.0f) {
            // Compute ||sum_norm||^2
            const float* sum_norm_col = get_sum_norm(col);
            float sum_norm_sq = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                sum_norm_sq += sum_norm_col[d] * sum_norm_col[d];
            }

            float coherence = compute_column_coherence(col);

            // DEBUG: Print sample columns with safer formatting
            if (debug_mode) {
                for (int i = 0; i < 5; ++i) {
                    if (col == sample_cols[i]) {
                        // Use scientific notation to handle large values
                        fprintf(stderr,
                                "[ECS DEBUG] Col %d: n=%.2e, ||sum_norm||^2=%.4e, "
                                "||sum_norm||=%.4e, coherence=%.6f\n",
                                col, static_cast<double>(n), static_cast<double>(sum_norm_sq),
                                std::sqrt(static_cast<double>(sum_norm_sq)),
                                static_cast<double>(coherence));

                        // Sanity check: weights should be <= num_sequences
                        if (n > num_sequences * 1.1f) {
                            fprintf(
                                stderr,
                                "[ECS WARNING] Weight %.2e exceeds num_sequences=%d by >10%%!\n",
                                static_cast<double>(n), num_sequences);
                        }
                        break;
                    }
                }
            }

            weighted_coherence += coherence * n;
            total_weight += n;
        }
    }

    float ecs = 0.0f;
    if (total_weight > 0.0f) {
        ecs = weighted_coherence / total_weight;
    }

    // DEBUG: Final ECS summary
    if (debug_mode) {
        fprintf(stderr, "[ECS DEBUG] Final: weighted_coherence=%.4f, total_weight=%.4f, ECS=%.6f\n",
                static_cast<double>(weighted_coherence), static_cast<double>(total_weight),
                static_cast<double>(ecs));
        fprintf(stderr, "[ECS DEBUG] Profile: %d sequences, %d columns\n", num_sequences, length);
    }

    return ecs;
}

float Profile::compute_ecs_detailed(float* out_column_coherences) const {
    float weighted_coherence = 0.0f;
    float total_weight = 0.0f;

    for (int col = 0; col < length; ++col) {
        float n = weights[col];
        float coherence = 0.0f;

        if (n >= 2.0f) {
            coherence = compute_column_coherence(col);
            weighted_coherence += coherence * n;
            total_weight += n;
        }

        // Store per-column coherence
        out_column_coherences[col] = coherence;
    }

    if (total_weight > 0.0f) {
        return weighted_coherence / total_weight;
    }

    return 0.0f;
}

}  // namespace msa
}  // namespace pfalign
