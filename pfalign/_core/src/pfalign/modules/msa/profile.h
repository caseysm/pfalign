#pragma once

#include "pfalign/common/growable_arena.h"
#include <vector>
#include <cstddef>

namespace pfalign {
namespace msa {

/**
 * Represents a position in an aligned sequence within a profile.
 * pos = -1 indicates a gap at this column.
 */
struct AlignmentPosition {
    int seq_idx;  // Index of sequence in original set
    int pos;      // Position in original sequence (-1 for gap)

    bool is_gap() const {
        return pos < 0;
    }
};

/**
 * Represents a column in a multiple sequence alignment.
 * Contains one position per sequence (gap or residue).
 */
struct AlignmentColumn {
    std::vector<AlignmentPosition> positions;

    AlignmentColumn() = default;
    explicit AlignmentColumn(int num_seqs) : positions(static_cast<size_t>(num_seqs)) {
    }

    // Count non-gap positions in this column
    int count_residues() const;

    // Get gap fraction (0.0 = no gaps, 1.0 = all gaps)
    float gap_fraction() const;
};

/**
 * Profile: Represents a group of aligned sequences.
 *
 * A profile stores:
 * - Average MPNN embeddings for each aligned column
 * - Position-specific weights
 * - Gap statistics
 * - Original sequence indices
 *
 * Profiles can be created from:
 * - Single sequences (leaf nodes in guide tree)
 * - Pairwise alignments (merging two profiles/sequences)
 */
struct Profile {
    // Core data
    int length;         // Aligned profile length (number of columns)
    int num_sequences;  // Number of sequences in this profile
    int hidden_dim;     // MPNN embedding dimension (typically 64)

    // Embeddings: [length * hidden_dim]
    // For each column, weighted average of non-gap embeddings
    // Average is computed as: sum(emb[i] * weight[i]) / sum(weight[i])
    float* embeddings;

    // Position weights: [length]
    // Number of non-gap residues contributing to each column
    // For single sequences: weight = 1.0
    // For merged profiles: weight = sum of child weights
    float* weights;

    // Normalized embedding sums: [length * hidden_dim]
    // Sum of unit-normalized embeddings for ECS computation
    // sum_norm[col] = sum(emb[i] / ||emb[i]|| * weight[i])
    float* sum_norm;

    // Gap counts: [length]
    // Number of gaps at each position
    int* gap_counts;

    // Alignment structure
    AlignmentColumn* columns;  // [length] - alignment columns

    // Metadata
    std::vector<int> seq_indices;  // Original sequence indices in this profile

    // Memory management
    pfalign::memory::GrowableArena* arena;

    // Constructor (private - use factory methods)
    Profile(int len, int num_seqs, int hidden, pfalign::memory::GrowableArena* arena);
    ~Profile();

    // Factory method: Create profile from single sequence
    static Profile* from_single_sequence(const float* embeddings,  // [length * hidden_dim]
                                         int length, int hidden_dim, int seq_index,
                                         pfalign::memory::GrowableArena* arena);

    // Factory method: Create profile from alignment of two profiles
    static Profile* from_alignment(const Profile& profile1, const Profile& profile2,
                                   const AlignmentColumn* alignment,  // [aligned_length]
                                   int aligned_length, pfalign::memory::GrowableArena* arena);

    // Destroy profile (calls destructor but does not release arena storage)
    static void destroy(Profile* profile);

    // Get embedding for a specific column
    const float* get_embedding(int col) const {
        return embeddings + col * hidden_dim;
    }

    // Get mutable embedding pointer for a specific column
    float* get_embedding_mut(int col) {
        return embeddings + col * hidden_dim;
    }

    // Check if column has gaps
    bool has_gaps(int col) const {
        return gap_counts[col] > 0;
    }

    // Get gap fraction for a column (0.0 = no gaps, 1.0 = all gaps)
    float gap_fraction(int col) const {
        return static_cast<float>(gap_counts[col]) / num_sequences;
    }

    // Get number of non-gap positions in column
    // NOTE: This is different from weights[col]!
    // residue_count = number of sequences with residues at this column
    // weights[col] = total contributing residue count (can be > num_sequences for merged profiles)
    int residue_count(int col) const {
        return num_sequences - gap_counts[col];
    }

    // Get sum_norm pointer for a column (for ECS computation)
    const float* get_sum_norm(int col) const {
        return sum_norm + col * hidden_dim;
    }

    // Get mutable sum_norm pointer
    float* get_sum_norm_mut(int col) {
        return sum_norm + col * hidden_dim;
    }

    // ========================================================================
    // Embedding Coherence Score (ECS) Methods
    // ========================================================================

    /**
     * Compute coherence for a single column.
     *
     * Coherence measures how similar the embeddings are within a column.
     * Formula: (||sum_norm||^2 - n) / (n * (n - 1))
     * where n = weights[col] (number of residues contributing)
     *
     * Returns value in [-1, 1]:
     * - 1.0: All embeddings identical (perfect alignment)
     * - 0.0: Embeddings uncorrelated (random)
     * - -1.0: Embeddings opposite (structural conflict)
     *
     * @param col Column index
     * @return Coherence score, or 0.0 if n < 2
     */
    float compute_column_coherence(int col) const;

    /**
     * Compute global Embedding Coherence Score (ECS).
     *
     * ECS is the weighted average of column coherences, weighted by
     * the number of residues contributing to each column.
     *
     * Formula: ECS = sum(coherence[col] * weights[col]) / sum(weights[col])
     *
     * @return Global ECS in [-1, 1]
     */
    float compute_ecs() const;

    /**
     * Compute ECS with detailed per-column breakdown.
     *
     * @param out_column_coherences Output array [length] for per-column scores
     * @return Global ECS
     */
    float compute_ecs_detailed(float* out_column_coherences) const;
};

}  // namespace msa
}  // namespace pfalign
