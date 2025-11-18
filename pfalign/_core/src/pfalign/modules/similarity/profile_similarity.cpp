#include "profile_similarity.h"
#include "pfalign/dispatch/backend_traits.h"
#include <cmath>
#include <stdexcept>

namespace pfalign {
namespace similarity {

// ============================================================================
// Sequence-Profile Similarity
// ============================================================================

template <typename Backend>
void compute_sequence_profile_similarity(const float* seq_embeddings, const msa::Profile& profile,
                                         float* similarity, int L1, int L2, int hidden_dim,
                                         bool apply_weights) {
    // Profile embeddings are pre-averaged [L2 * hidden_dim]
    // Just use the standard similarity computation
    pfalign::similarity::compute_similarity<Backend>(seq_embeddings, profile.embeddings, similarity,
                                                     L1, L2, hidden_dim);

    // Optionally apply column weights
    if (apply_weights) {
        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                similarity[i * L2 + j] *= profile.weights[j];
            }
        }
    }
}

// ============================================================================
// Profile-Sequence Similarity
// ============================================================================

template <typename Backend>
void compute_profile_sequence_similarity(const msa::Profile& profile, const float* seq_embeddings,
                                         float* similarity, int L1, int L2, int hidden_dim,
                                         bool apply_weights) {
    // Profile embeddings are pre-averaged [L1 * hidden_dim]
    pfalign::similarity::compute_similarity<Backend>(profile.embeddings, seq_embeddings, similarity,
                                                     L1, L2, hidden_dim);

    // Optionally apply row weights (from profile)
    if (apply_weights) {
        for (int i = 0; i < L1; i++) {
            float weight_i = profile.weights[i];
            for (int j = 0; j < L2; j++) {
                similarity[i * L2 + j] *= weight_i;
            }
        }
    }
}

// ============================================================================
// Profile-Profile Similarity
// ============================================================================

template <typename Backend>
void compute_profile_profile_similarity(const msa::Profile& profile1, const msa::Profile& profile2,
                                        float* similarity, int L1, int L2, int hidden_dim,
                                        bool apply_weights) {
    // Both profiles have averaged embeddings [L * hidden_dim]
    pfalign::similarity::compute_similarity<Backend>(profile1.embeddings, profile2.embeddings,
                                                     similarity, L1, L2, hidden_dim);

    // Optionally apply weights from both profiles
    // Use geometric mean: sqrt(w1[i] * w2[j])
    // This ensures symmetry and balances both profile contributions
    if (apply_weights) {
        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                float combined_weight = std::sqrt(profile1.weights[i] * profile2.weights[j]);
                similarity[i * L2 + j] *= combined_weight;
            }
        }
    }
}

// ============================================================================
// Convenience Wrappers Using SequenceEmbeddings
// ============================================================================

template <typename Backend>
void compute_sequence_similarity(const SequenceEmbeddings& seq1, const SequenceEmbeddings& seq2,
                                 float* similarity) {
    pfalign::similarity::compute_similarity<Backend>(seq1.get_embeddings(), seq2.get_embeddings(),
                                                     similarity, seq1.length, seq2.length,
                                                     seq1.hidden_dim);
}

template <typename Backend>
void compute_sequence_profile_similarity(const SequenceEmbeddings& seq, const msa::Profile& profile,
                                         float* similarity, bool apply_weights) {
    compute_sequence_profile_similarity<Backend>(seq.get_embeddings(), profile, similarity,
                                                 seq.length, profile.length, seq.hidden_dim,
                                                 apply_weights);
}

template <typename Backend>
void compute_profile_sequence_similarity(const msa::Profile& profile, const SequenceEmbeddings& seq,
                                         float* similarity, bool apply_weights) {
    compute_profile_sequence_similarity<Backend>(profile, seq.get_embeddings(), similarity,
                                                 profile.length, seq.length, profile.hidden_dim,
                                                 apply_weights);
}

// ============================================================================
// Generic Dispatcher Implementation
// ============================================================================

template <typename Backend>
void SimilarityComputer<Backend>::compute(const void* input1, const void* input2, float* similarity,
                                          int L1, int L2, int hidden_dim, bool apply_weights) {
    switch (mode_) {
        case SimilarityMode::SEQUENCE_SEQUENCE: {
            const float* seq1 = static_cast<const float*>(input1);
            const float* seq2 = static_cast<const float*>(input2);
            pfalign::similarity::compute_similarity<Backend>(seq1, seq2, similarity, L1, L2,
                                                             hidden_dim);
            break;
        }

        case SimilarityMode::SEQUENCE_PROFILE: {
            const float* seq = static_cast<const float*>(input1);
            const msa::Profile* profile = static_cast<const msa::Profile*>(input2);
            compute_sequence_profile_similarity<Backend>(seq, *profile, similarity, L1, L2,
                                                         hidden_dim, apply_weights);
            break;
        }

        case SimilarityMode::PROFILE_SEQUENCE: {
            const msa::Profile* profile = static_cast<const msa::Profile*>(input1);
            const float* seq = static_cast<const float*>(input2);
            compute_profile_sequence_similarity<Backend>(*profile, seq, similarity, L1, L2,
                                                         hidden_dim, apply_weights);
            break;
        }

        case SimilarityMode::PROFILE_PROFILE: {
            const msa::Profile* profile1 = static_cast<const msa::Profile*>(input1);
            const msa::Profile* profile2 = static_cast<const msa::Profile*>(input2);
            compute_profile_profile_similarity<Backend>(*profile1, *profile2, similarity, L1, L2,
                                                        hidden_dim, apply_weights);
            break;
        }

        default:
            throw std::invalid_argument("Unknown SimilarityMode");
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// Instantiate for ScalarBackend (always available)
template void compute_sequence_profile_similarity<ScalarBackend>(const float*, const msa::Profile&,
                                                                 float*, int, int, int, bool);

template void compute_profile_sequence_similarity<ScalarBackend>(const msa::Profile&, const float*,
                                                                 float*, int, int, int, bool);

template void compute_profile_profile_similarity<ScalarBackend>(const msa::Profile&,
                                                                const msa::Profile&, float*, int,
                                                                int, int, bool);

template void compute_sequence_similarity<ScalarBackend>(const SequenceEmbeddings&,
                                                         const SequenceEmbeddings&, float*);

template void compute_sequence_profile_similarity<ScalarBackend>(const SequenceEmbeddings&,
                                                                 const msa::Profile&, float*, bool);

template void compute_profile_sequence_similarity<ScalarBackend>(const msa::Profile&,
                                                                 const SequenceEmbeddings&, float*,
                                                                 bool);

template class SimilarityComputer<ScalarBackend>;

}  // namespace similarity
}  // namespace pfalign
