/**
 * Profile-Aware Similarity Computation
 *
 * Extends the similarity module to support:
 * - Sequence-Sequence: dot(emb1[i], emb2[j])
 * - Sequence-Profile: dot(seq_emb[i], profile.embeddings[j])
 * - Profile-Sequence: dot(profile.embeddings[i], seq_emb[j])
 * - Profile-Profile: dot(profile1.embeddings[i], profile2.embeddings[j])
 *
 * All modes reuse the same underlying dot-product primitives.
 * Profile embeddings are pre-averaged [L * hidden_dim], so computation
 * is identical to sequence-sequence similarity.
 *
 * Optional weighting: Can apply profile.weights[col] to emphasize
 * well-supported columns (high residue counts) and de-emphasize
 * gappy columns (low residue counts).
 */

#pragma once

#include "similarity.h"
#include "pfalign/modules/msa/profile.h"
#include "pfalign/modules/mpnn/sequence_cache.h"

namespace pfalign {
namespace similarity {

/**
 * Compute similarity between sequence embeddings and profile.
 *
 * Used for aligning a new sequence to an existing profile during
 * progressive MSA.
 *
 * @param seq_embeddings    Sequence embeddings [L1 * hidden_dim]
 * @param profile           Profile with averaged embeddings [L2 * hidden_dim]
 * @param similarity        Output similarity matrix [L1 * L2]
 * @param L1                Sequence length
 * @param L2                Profile length
 * @param hidden_dim        Embedding dimension (typically 64)
 * @param apply_weights     If true, weight by profile.weights (default: false)
 *
 * Formula:
 *   similarity[i,j] = dot(seq_embeddings[i], profile.embeddings[j])
 *   if apply_weights: similarity[i,j] *= profile.weights[j]
 *
 * Example:
 *   Profile* profile = ...;  // Existing profile, length 100
 *   float* seq_emb = ...;    // New sequence embeddings, length 80
 *   float sim[80 * 100];
 *
 *   compute_sequence_profile_similarity<ScalarBackend>(
 *       seq_emb, *profile, sim, 80, 100, 64
 *   );
 */
template <typename Backend>
void compute_sequence_profile_similarity(const float* seq_embeddings, const msa::Profile& profile,
                                         float* similarity, int L1, int L2, int hidden_dim,
                                         bool apply_weights = false);

/**
 * Compute similarity between profile and sequence embeddings.
 *
 * Same as compute_sequence_profile_similarity but with arguments swapped.
 * Provided for API symmetry.
 *
 * @param profile           Profile with averaged embeddings [L1 * hidden_dim]
 * @param seq_embeddings    Sequence embeddings [L2 * hidden_dim]
 * @param similarity        Output similarity matrix [L1 * L2]
 * @param L1                Profile length
 * @param L2                Sequence length
 * @param hidden_dim        Embedding dimension
 * @param apply_weights     If true, weight by profile.weights (default: false)
 *
 * Formula:
 *   similarity[i,j] = dot(profile.embeddings[i], seq_embeddings[j])
 *   if apply_weights: similarity[i,j] *= profile.weights[i]
 */
template <typename Backend>
void compute_profile_sequence_similarity(const msa::Profile& profile, const float* seq_embeddings,
                                         float* similarity, int L1, int L2, int hidden_dim,
                                         bool apply_weights = false);

/**
 * Compute similarity between two profiles.
 *
 * Uses averaged column embeddings from each profile.
 * Optionally applies position-specific weights for gap-aware scoring.
 *
 * @param profile1          First profile [L1 * hidden_dim]
 * @param profile2          Second profile [L2 * hidden_dim]
 * @param similarity        Output similarity matrix [L1 * L2]
 * @param L1                Profile 1 length
 * @param L2                Profile 2 length
 * @param hidden_dim        Embedding dimension
 * @param apply_weights     If true, weight by profile weights (default: false)
 *
 * Formula (apply_weights = false):
 *   similarity[i,j] = dot(profile1.embeddings[i], profile2.embeddings[j])
 *
 * Formula (apply_weights = true):
 *   similarity[i,j] = dot(...) * sqrt(weights1[i] * weights2[j])
 *
 * The sqrt weighting scheme balances contributions from both profiles
 * and ensures symmetry: S[i,j] = S[j,i] when profiles are swapped.
 *
 * Example:
 *   Profile* p1 = ...;  // Profile 1, length 100
 *   Profile* p2 = ...;  // Profile 2, length 150
 *   float sim[100 * 150];
 *
 *   compute_profile_profile_similarity<ScalarBackend>(
 *       *p1, *p2, sim, 100, 150, 64, true  // with weighting
 *   );
 */
template <typename Backend>
void compute_profile_profile_similarity(const msa::Profile& profile1, const msa::Profile& profile2,
                                        float* similarity, int L1, int L2, int hidden_dim,
                                        bool apply_weights = false);

// ============================================================================
// Convenience Wrappers Using SequenceEmbeddings
// ============================================================================

/**
 * Compute similarity between two cached sequences.
 *
 * Convenience wrapper that extracts embeddings from SequenceEmbeddings.
 *
 * @param seq1          First sequence (from SequenceCache)
 * @param seq2          Second sequence (from SequenceCache)
 * @param similarity    Output similarity matrix [L1 * L2]
 */
template <typename Backend>
void compute_sequence_similarity(const SequenceEmbeddings& seq1, const SequenceEmbeddings& seq2,
                                 float* similarity);

/**
 * Compute similarity between cached sequence and profile.
 */
template <typename Backend>
void compute_sequence_profile_similarity(const SequenceEmbeddings& seq, const msa::Profile& profile,
                                         float* similarity, bool apply_weights = false);

/**
 * Compute similarity between profile and cached sequence.
 */
template <typename Backend>
void compute_profile_sequence_similarity(const msa::Profile& profile, const SequenceEmbeddings& seq,
                                         float* similarity, bool apply_weights = false);

// ============================================================================
// Generic Dispatcher (Strategy Pattern)
// ============================================================================

/**
 * Similarity computation mode.
 *
 * Determines which similarity function to use based on input types.
 */
enum class SimilarityMode {
    SEQUENCE_SEQUENCE,  // Sequence * Sequence
    SEQUENCE_PROFILE,   // Sequence * Profile
    PROFILE_SEQUENCE,   // Profile * Sequence
    PROFILE_PROFILE     // Profile * Profile
};

/**
 * Generic similarity computer with automatic dispatch.
 *
 * Selects the appropriate similarity function based on mode.
 * Useful for generic alignment code that works with both
 * sequences and profiles.
 *
 * Example:
 *   SimilarityComputer<ScalarBackend> computer(SimilarityMode::PROFILE_PROFILE);
 *
 *   // Compute similarity (automatically dispatches to profile-profile)
 *   computer.compute(
 *       &profile1, &profile2, similarity,
 *       L1, L2, hidden_dim, apply_weights
 *   );
 */
template <typename Backend>
class SimilarityComputer {
public:
    explicit SimilarityComputer(SimilarityMode mode) : mode_(mode) {
    }

    /**
     * Compute similarity with automatic dispatch.
     *
     * Input types must match the configured mode:
     * - SEQUENCE_SEQUENCE: input1 = float*, input2 = float*
     * - SEQUENCE_PROFILE: input1 = float*, input2 = Profile*
     * - PROFILE_SEQUENCE: input1 = Profile*, input2 = float*
     * - PROFILE_PROFILE: input1 = Profile*, input2 = Profile*
     *
     * @param input1        First input (sequence embeddings or Profile*)
     * @param input2        Second input (sequence embeddings or Profile*)
     * @param similarity    Output similarity matrix [L1 * L2]
     * @param L1            First input length
     * @param L2            Second input length
     * @param hidden_dim    Embedding dimension
     * @param apply_weights Apply profile weights (only for profile modes)
     */
    void compute(const void* input1, const void* input2, float* similarity, int L1, int L2,
                 int hidden_dim, bool apply_weights = false);

    SimilarityMode get_mode() const {
        return mode_;
    }
    void set_mode(SimilarityMode mode) {
        mode_ = mode;
    }

private:
    SimilarityMode mode_;
};

}  // namespace similarity
}  // namespace pfalign
