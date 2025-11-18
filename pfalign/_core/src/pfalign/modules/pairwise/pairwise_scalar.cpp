/**
 * Scalar implementation of pairwise protein alignment pipeline.
 */

#include "pairwise_align.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/modules/alignment/alignment.h"
#include "pfalign/primitives/reduce/reduce.h"
#include "pfalign/primitives/alignment_decode/alignment_decode.h"
#include "pfalign/common/thread_pool.h"
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace {

inline void validate_finite_array(const float* data, std::size_t count, const char* label) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            std::ostringstream oss;
            oss << label << " contains non-finite value at index " << i;
            throw std::invalid_argument(oss.str());
        }
    }
}

template <typename Backend>
void RunMpnnEncoders(const pfalign::pairwise::PairwiseConfig& config,
                     const pfalign::mpnn::MPNNWeights& weights, const float* coords1, int L1,
                     const float* coords2, int L2, float* embeddings1, float* embeddings2,
                     pfalign::mpnn::MPNNWorkspace* workspace1,
                     pfalign::mpnn::MPNNWorkspace* workspace2) {
    auto encode = [&](const float* coords, int length, float* embeddings,
                      pfalign::mpnn::MPNNWorkspace* ws) {
        pfalign::mpnn::mpnn_forward<Backend>(coords, length, weights, config.mpnn_config,
                                             embeddings, ws);
    };

    if (config.parallel_mpnn) {
        // Use ThreadPool for parallel MPNN encoding (2 encoders)
        pfalign::threading::ThreadPool pool(2, 100);  // 2 threads, 100MB per arena

        pool.parallel_for(
            2, [&](int tid, size_t begin, size_t end, pfalign::memory::GrowableArena& arena) {
                (void)begin;  // Unused - we process exactly 2 items
                (void)end;
                (void)arena;  // MPNNWorkspace has its own arena

                if (tid == 0) {
                    encode(coords1, L1, embeddings1, workspace1);
                } else {
                    encode(coords2, L2, embeddings2, workspace2);
                }
            });
    } else {
        encode(coords1, L1, embeddings1, workspace1);
        encode(coords2, L2, embeddings2, workspace2);
    }
}

}  // namespace

namespace pfalign {
namespace pairwise {

/**
 * Full pipeline: Coords -> MPNN -> Similarity -> Smith-Waterman
 */
template <>
void pairwise_align<ScalarBackend>(const float* coords1, int L1, const float* coords2, int L2,
                                   const PairwiseConfig& config, const mpnn::MPNNWeights& weights,
                                   PairwiseWorkspace* workspace, float* partition) {
    if (coords1 == nullptr) {
        throw std::invalid_argument("pairwise_align: coords1 pointer is null");
    }
    if (coords2 == nullptr) {
        throw std::invalid_argument("pairwise_align: coords2 pointer is null");
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("pairwise_align: workspace pointer is null");
    }
    if (partition == nullptr) {
        throw std::invalid_argument("pairwise_align: partition pointer is null");
    }
    if (L1 <= 0 || L2 <= 0) {
        throw std::invalid_argument("pairwise_align: sequence lengths must be positive");
    }
    if (config.mpnn_config.hidden_dim <= 0) {
        throw std::invalid_argument(
            "pairwise_align: config.mpnn_config.hidden_dim must be positive");
    }
    if (config.mpnn_config.k_neighbors <= 0) {
        throw std::invalid_argument(
            "pairwise_align: config.mpnn_config.k_neighbors must be positive");
    }
    if (config.mpnn_config.num_layers <= 0) {
        throw std::invalid_argument(
            "pairwise_align: config.mpnn_config.num_layers must be positive");
    }
    if (weights.num_layers != config.mpnn_config.num_layers) {
        std::ostringstream oss;
        oss << "pairwise_align: weights.num_layers (" << weights.num_layers
            << ") does not match config.mpnn_config.num_layers (" << config.mpnn_config.num_layers
            << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->mpnn_ws1 == nullptr || workspace->mpnn_ws2 == nullptr) {
        throw std::invalid_argument("pairwise_align: workspace MPNN workspaces are null");
    }
    if (workspace->hidden_dim != config.mpnn_config.hidden_dim) {
        std::ostringstream oss;
        oss << "pairwise_align: workspace hidden_dim (" << workspace->hidden_dim
            << ") does not match config.mpnn_config.hidden_dim (" << config.mpnn_config.hidden_dim
            << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->mpnn_ws1->hidden_dim != config.mpnn_config.hidden_dim ||
        workspace->mpnn_ws2->hidden_dim != config.mpnn_config.hidden_dim) {
        throw std::invalid_argument(
            "pairwise_align: MPNN workspace hidden_dim does not match "
            "config.mpnn_config.hidden_dim");
    }
    if (workspace->mpnn_ws1->k < config.mpnn_config.k_neighbors ||
        workspace->mpnn_ws2->k < config.mpnn_config.k_neighbors) {
        std::ostringstream oss;
        oss << "pairwise_align: workspace k_neighbors ("
            << std::min(workspace->mpnn_ws1->k, workspace->mpnn_ws2->k)
            << ") is smaller than config.mpnn_config.k_neighbors ("
            << config.mpnn_config.k_neighbors << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->mpnn_ws1->L < L1) {
        std::ostringstream oss;
        oss << "pairwise_align: workspace mpnn_ws1 length (" << workspace->mpnn_ws1->L
            << ") is smaller than L1 (" << L1 << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->mpnn_ws2->L < L2) {
        std::ostringstream oss;
        oss << "pairwise_align: workspace mpnn_ws2 length (" << workspace->mpnn_ws2->L
            << ") is smaller than L2 (" << L2 << ")";
        throw std::invalid_argument(oss.str());
    }

    if (workspace->embeddings1 == nullptr || workspace->embeddings2 == nullptr ||
        workspace->similarity == nullptr || workspace->sw_matrix == nullptr) {
        throw std::invalid_argument("pairwise_align: workspace buffers are null");
    }

    // Coords are backbone-only: (N, CA, C, O) = 4 atoms * 3 coordinates per residue
    const std::size_t coords1_count = static_cast<std::size_t>(L1) * 4u * 3u;
    const std::size_t coords2_count = static_cast<std::size_t>(L2) * 4u * 3u;
    validate_finite_array(coords1, coords1_count, "pairwise_align: coords1");
    validate_finite_array(coords2, coords2_count, "pairwise_align: coords2");

    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        *partition = smith_waterman::NINF;
        return;
    }

    // Step 1-2: Encode both proteins (optionally in parallel)
    RunMpnnEncoders<ScalarBackend>(config, weights, coords1, L1, coords2, L2,
                                   workspace->embeddings1, workspace->embeddings2,
                                   workspace->mpnn_ws1, workspace->mpnn_ws2);

    // Step 3: Compute similarity matrix
    similarity::compute_similarity<ScalarBackend>(workspace->embeddings1, workspace->embeddings2,
                                                  workspace->similarity, L1, L2,
                                                  config.mpnn_config.hidden_dim);

    // Step 4: Smith-Waterman alignment
    switch (config.sw_mode) {
        case PairwiseConfig::SWMode::DIRECT_REGULAR:
            smith_waterman::smith_waterman_direct_regular<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::DIRECT_AFFINE:
            smith_waterman::smith_waterman_direct_affine<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE:
            smith_waterman::smith_waterman_direct_affine_flexible<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::JAX_REGULAR:
            smith_waterman::smith_waterman_jax_regular<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::JAX_AFFINE_STANDARD:
            smith_waterman::smith_waterman_jax_affine<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE:
            smith_waterman::smith_waterman_jax_affine_flexible<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;
    }
}

/**
 * From pre-computed embeddings: Embeddings -> Similarity -> Smith-Waterman
 */
template <>
void pairwise_align_from_embeddings<ScalarBackend>(const float* embeddings1, int L1,
                                                   const float* embeddings2, int L2, int hidden_dim,
                                                   const PairwiseConfig& config,
                                                   PairwiseWorkspace* workspace, float* partition) {
    if (embeddings1 == nullptr) {
        throw std::invalid_argument("pairwise_align_from_embeddings: embeddings1 pointer is null");
    }
    if (embeddings2 == nullptr) {
        throw std::invalid_argument("pairwise_align_from_embeddings: embeddings2 pointer is null");
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("pairwise_align_from_embeddings: workspace pointer is null");
    }
    if (partition == nullptr) {
        throw std::invalid_argument("pairwise_align_from_embeddings: partition pointer is null");
    }
    if (L1 <= 0 || L2 <= 0) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings: sequence lengths must be positive");
    }
    if (hidden_dim <= 0) {
        throw std::invalid_argument("pairwise_align_from_embeddings: hidden_dim must be positive");
    }
    if (config.mpnn_config.hidden_dim != hidden_dim) {
        std::ostringstream oss;
        oss << "pairwise_align_from_embeddings: config.mpnn_config.hidden_dim ("
            << config.mpnn_config.hidden_dim << ") does not match hidden_dim (" << hidden_dim
            << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->hidden_dim != hidden_dim) {
        std::ostringstream oss;
        oss << "pairwise_align_from_embeddings: workspace hidden_dim (" << workspace->hidden_dim
            << ") does not match hidden_dim (" << hidden_dim << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->similarity == nullptr || workspace->sw_matrix == nullptr) {
        throw std::invalid_argument("pairwise_align_from_embeddings: workspace buffers are null");
    }

    if (workspace->similarity == nullptr || workspace->sw_matrix == nullptr) {
        throw std::invalid_argument("pairwise_align_from_embeddings: workspace buffers are null");
    }

    const std::size_t emb1_count =
        static_cast<std::size_t>(L1) * static_cast<std::size_t>(hidden_dim);
    const std::size_t emb2_count =
        static_cast<std::size_t>(L2) * static_cast<std::size_t>(hidden_dim);
    validate_finite_array(embeddings1, emb1_count, "pairwise_align_from_embeddings: embeddings1");
    validate_finite_array(embeddings2, emb2_count, "pairwise_align_from_embeddings: embeddings2");

    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        *partition = smith_waterman::NINF;
        return;
    }

    // Step 1: Compute similarity matrix
    similarity::compute_similarity<ScalarBackend>(embeddings1, embeddings2, workspace->similarity,
                                                  L1, L2, hidden_dim);

    // Step 2: Smith-Waterman alignment
    switch (config.sw_mode) {
        case PairwiseConfig::SWMode::DIRECT_REGULAR:
            smith_waterman::smith_waterman_direct_regular<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::DIRECT_AFFINE:
            smith_waterman::smith_waterman_direct_affine<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE:
            smith_waterman::smith_waterman_direct_affine_flexible<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::JAX_REGULAR:
            smith_waterman::smith_waterman_jax_regular<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::JAX_AFFINE_STANDARD:
            smith_waterman::smith_waterman_jax_affine<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;

        case PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE:
            smith_waterman::smith_waterman_jax_affine_flexible<ScalarBackend>(
                workspace->similarity, L1, L2, config.sw_config, workspace->sw_matrix, partition);
            break;
    }
}

/**
 * From embeddings with score: Embeddings -> Similarity -> SW (fwd+bwd) -> Score
 */
template <>
void pairwise_align_from_embeddings_with_score<ScalarBackend>(
    const float* embeddings1, int L1, const float* embeddings2, int L2, int hidden_dim,
    const PairwiseConfig& config, PairwiseWorkspace* workspace, float* partition, float* score,
    pfalign::memory::GrowableArena* arena, float* score_magnitude) {
    if (embeddings1 == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: embeddings1 pointer is null");
    }
    if (embeddings2 == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: embeddings2 pointer is null");
    }
    if (workspace == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: workspace pointer is null");
    }
    if (partition == nullptr || score == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: output pointers are null");
    }
    if (arena == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: arena pointer is null");
    }
    if (L1 <= 0 || L2 <= 0) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: sequence lengths must be positive");
    }
    if (hidden_dim <= 0) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: hidden_dim must be positive");
    }
    if (workspace->hidden_dim != hidden_dim) {
        std::ostringstream oss;
        oss << "pairwise_align_from_embeddings_with_score: workspace hidden_dim ("
            << workspace->hidden_dim << ") does not match hidden_dim (" << hidden_dim << ")";
        throw std::invalid_argument(oss.str());
    }
    if (workspace->similarity == nullptr || workspace->sw_matrix == nullptr ||
        workspace->posteriors == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_score: workspace buffers are null");
    }

    const std::size_t emb1_count =
        static_cast<std::size_t>(L1) * static_cast<std::size_t>(hidden_dim);
    const std::size_t emb2_count =
        static_cast<std::size_t>(L2) * static_cast<std::size_t>(hidden_dim);
    validate_finite_array(embeddings1, emb1_count,
                          "pairwise_align_from_embeddings_with_score: embeddings1");
    validate_finite_array(embeddings2, emb2_count,
                          "pairwise_align_from_embeddings_with_score: embeddings2");

    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        *partition = smith_waterman::NINF;
        *score = 0.0f;
        if (score_magnitude)
            *score_magnitude = 0.0f;
        return;
    }

    // Scoped arena for automatic cleanup of scratch allocations
    pfalign::memory::ScopedGrowableArena scope(*arena);

    const int matrix_size = L1 * L2;

    // Step 1: Compute similarity matrix
    similarity::compute_similarity<ScalarBackend>(embeddings1, embeddings2, workspace->similarity,
                                                  L1, L2, hidden_dim);

    // Step 2: Convert SW mode to alignment mode
    alignment::AlignmentMode align_mode = alignment::AlignmentMode::JAX_REGULAR;  // Default initialization
    switch (config.sw_mode) {
        case PairwiseConfig::SWMode::DIRECT_REGULAR:
            align_mode = alignment::AlignmentMode::DIRECT_REGULAR;
            break;
        case PairwiseConfig::SWMode::DIRECT_AFFINE:
            align_mode = alignment::AlignmentMode::DIRECT_AFFINE;
            break;
        case PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE:
            align_mode = alignment::AlignmentMode::DIRECT_AFFINE_FLEXIBLE;
            break;
        case PairwiseConfig::SWMode::JAX_REGULAR:
            align_mode = alignment::AlignmentMode::JAX_REGULAR;
            break;
        case PairwiseConfig::SWMode::JAX_AFFINE_STANDARD:
            align_mode = alignment::AlignmentMode::JAX_AFFINE;
            break;
        case PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE:
            align_mode = alignment::AlignmentMode::JAX_AFFINE_FLEXIBLE;
            break;
    }

    alignment::AlignmentConfig align_config;
    align_config.mode = align_mode;
    align_config.sw_config = config.sw_config;

    // Step 3: Smith-Waterman forward + backward (computes posteriors)
    alignment::compute_alignment_with_posteriors<ScalarBackend>(
        workspace->similarity, L1, L2, align_config, workspace->sw_matrix, workspace->posteriors,
        partition, arena);

    // Step 4: Normalize posteriors to sum to 1 (they represent P(i,j | alignment))
    float posterior_sum = reduce::matrix_sum<ScalarBackend>(workspace->posteriors, matrix_size);
    if (posterior_sum < 1e-10f) {
        *score = 0.0f;  // No valid alignment
        if (score_magnitude)
            *score_magnitude = 0.0f;
        return;
    }

    // Normalize posteriors in-place
    for (int i = 0; i < matrix_size; i++) {
        workspace->posteriors[i] /= posterior_sum;
    }

    // Step 5: Compute embedding norms for cosine similarity
    // We need ||e1[i]|| for each position i in sequence 1
    // and ||e2[j]|| for each position j in sequence 2
    float* norms1 = scope.allocate<float>(static_cast<size_t>(L1));
    float* norms2 = scope.allocate<float>(static_cast<size_t>(L2));
    if (!norms1 || !norms2) {
        *score = 0.0f;  // Arena allocation failed
        if (score_magnitude)
            *score_magnitude = 0.0f;
        return;
    }

    // Compute norms: ||e1[i]|| = sqrt(sum(e1[i,k]^2)) for k in [0, hidden_dim)
    for (int i = 0; i < L1; i++) {
        float sum_sq = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            float val = embeddings1[i * hidden_dim + k];
            sum_sq += val * val;
        }
        norms1[i] = std::sqrt(sum_sq);
    }

    for (int j = 0; j < L2; j++) {
        float sum_sq = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            float val = embeddings2[j * hidden_dim + k];
            sum_sq += val * val;
        }
        norms2[j] = std::sqrt(sum_sq);
    }

    // Step 6: Allocate temporary buffer for cosine similarity
    float* cosine_similarity = scope.allocate<float>(static_cast<size_t>(matrix_size));
    if (!cosine_similarity) {
        *score = 0.0f;  // Arena allocation failed
        if (score_magnitude)
            *score_magnitude = 0.0f;
        return;
    }

    // Step 7: Convert dot product similarity to cosine similarity
    // cosine[i,j] = dot[i,j] / (||e1[i]|| * ||e2[j]||)
    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L2; j++) {
            float norm_product = norms1[i] * norms2[j];
            if (norm_product > 1e-10f) {
                cosine_similarity[i * L2 + j] = workspace->similarity[i * L2 + j] / norm_product;
            } else {
                cosine_similarity[i * L2 + j] = 0.0f;  // Handle zero-norm vectors
            }
        }
    }

    // Step 8: Allocate weighted buffer
    float* weighted = scope.allocate<float>(static_cast<size_t>(matrix_size));
    if (!weighted) {
        *score = 0.0f;  // Arena allocation failed
        if (score_magnitude)
            *score_magnitude = 0.0f;
        return;
    }

    // Step 9: Element-wise multiply: weighted = cosine_similarity ⊙ posteriors
    reduce::elementwise_multiply<ScalarBackend>(cosine_similarity, workspace->posteriors, weighted,
                                                matrix_size);

    // Step 10: Sum to get final cosine score
    *score = reduce::matrix_sum<ScalarBackend>(weighted, matrix_size);

    // Step 11: Optionally compute magnitude-aware score
    if (score_magnitude) {
        // Allocate buffers for magnitude score computation
        float* weighted_dot = scope.allocate<float>(static_cast<size_t>(matrix_size));
        float* norm_products = scope.allocate<float>(static_cast<size_t>(matrix_size));

        if (!weighted_dot || !norm_products) {
            *score_magnitude = 0.0f;  // Arena allocation failed
            return;
        }

        // Compute norm products: norm_products[i,j] = ||e1[i]|| * ||e2[j]||
        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                norm_products[i * L2 + j] = norms1[i] * norms2[j];
            }
        }

        // Compute magnitude-aware score: Sigma P[i,j] * dot[i,j] / Sigma P[i,j] * ||e1[i]|| *
        // ||e2[j]||
        reduce::elementwise_multiply<ScalarBackend>(workspace->similarity, workspace->posteriors,
                                                    weighted_dot, matrix_size);
        float numerator = reduce::matrix_sum<ScalarBackend>(weighted_dot, matrix_size);

        float denominator = 0.0f;
        for (int idx = 0; idx < matrix_size; idx++) {
            denominator += workspace->posteriors[idx] * norm_products[idx];
        }

        *score_magnitude = (denominator > 1e-10f) ? (numerator / denominator) : 0.0f;
    }
}

/**
 * Full pipeline with score: Coords -> MPNN -> Similarity -> SW (fwd+bwd) -> Score
 */
template <>
void pairwise_align_with_score<ScalarBackend>(const float* coords1, int L1, const float* coords2,
                                              int L2, const PairwiseConfig& config,
                                              const mpnn::MPNNWeights& weights,
                                              PairwiseWorkspace* workspace, float* partition,
                                              float* score, pfalign::memory::GrowableArena* arena,
                                              float* score_magnitude) {
    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        *partition = smith_waterman::NINF;
        *score = 0.0f;
        if (score_magnitude)
            *score_magnitude = 0.0f;
        return;
    }

    // Step 1-2: Encode both proteins (optionally in parallel)
    RunMpnnEncoders<ScalarBackend>(config, weights, coords1, L1, coords2, L2,
                                   workspace->embeddings1, workspace->embeddings2,
                                   workspace->mpnn_ws1, workspace->mpnn_ws2);

    // Step 3-6: Compute score from embeddings (optionally dual-score)
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        workspace->embeddings1, L1, workspace->embeddings2, L2, config.mpnn_config.hidden_dim,
        config, workspace, partition, score, arena, score_magnitude);
}

/**
 * Dual-score from embeddings: Embeddings -> Similarity -> SW -> Both Scores
 */
template <>
void pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
    const float* embeddings1, int L1, const float* embeddings2, int L2, int hidden_dim,
    const PairwiseConfig& config, PairwiseWorkspace* workspace, float* partition,
    float* score_cosine, float* score_magnitude, pfalign::memory::GrowableArena* arena) {
    if (embeddings1 == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: embeddings1 pointer is null");
    }
    if (embeddings2 == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: embeddings2 pointer is null");
    }
    if (workspace == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: workspace pointer is null");
    }
    if (partition == nullptr || score_cosine == nullptr || score_magnitude == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: output pointers are null");
    }
    if (arena == nullptr) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: arena pointer is null");
    }
    if (L1 <= 0 || L2 <= 0) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: sequence lengths must be positive");
    }
    if (hidden_dim <= 0) {
        throw std::invalid_argument(
            "pairwise_align_from_embeddings_with_dual_score: hidden_dim must be positive");
    }
    if (workspace->hidden_dim != hidden_dim) {
        std::ostringstream oss;
        oss << "pairwise_align_from_embeddings_with_dual_score: workspace hidden_dim ("
            << workspace->hidden_dim << ") does not match hidden_dim (" << hidden_dim << ")";
        throw std::invalid_argument(oss.str());
    }

    const std::size_t emb1_count =
        static_cast<std::size_t>(L1) * static_cast<std::size_t>(hidden_dim);
    const std::size_t emb2_count =
        static_cast<std::size_t>(L2) * static_cast<std::size_t>(hidden_dim);
    validate_finite_array(embeddings1, emb1_count,
                          "pairwise_align_from_embeddings_with_dual_score: embeddings1");
    validate_finite_array(embeddings2, emb2_count,
                          "pairwise_align_from_embeddings_with_dual_score: embeddings2");

    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        *partition = smith_waterman::NINF;
        *score_cosine = 0.0f;
        *score_magnitude = 0.0f;
        return;
    }

    // Scoped arena for automatic cleanup of scratch allocations
    pfalign::memory::ScopedGrowableArena scope(*arena);

    const int matrix_size = L1 * L2;

    // Step 1: Compute similarity matrix (dot product)
    similarity::compute_similarity<ScalarBackend>(embeddings1, embeddings2, workspace->similarity,
                                                  L1, L2, hidden_dim);

    // Step 2: Convert SW mode to alignment mode
    alignment::AlignmentMode align_mode = alignment::AlignmentMode::JAX_REGULAR;  // Default initialization
    switch (config.sw_mode) {
        case PairwiseConfig::SWMode::DIRECT_REGULAR:
            align_mode = alignment::AlignmentMode::DIRECT_REGULAR;
            break;
        case PairwiseConfig::SWMode::DIRECT_AFFINE:
            align_mode = alignment::AlignmentMode::DIRECT_AFFINE;
            break;
        case PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE:
            align_mode = alignment::AlignmentMode::DIRECT_AFFINE_FLEXIBLE;
            break;
        case PairwiseConfig::SWMode::JAX_REGULAR:
            align_mode = alignment::AlignmentMode::JAX_REGULAR;
            break;
        case PairwiseConfig::SWMode::JAX_AFFINE_STANDARD:
            align_mode = alignment::AlignmentMode::JAX_AFFINE;
            break;
        case PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE:
            align_mode = alignment::AlignmentMode::JAX_AFFINE_FLEXIBLE;
            break;
    }

    alignment::AlignmentConfig align_config;
    align_config.mode = align_mode;
    align_config.sw_config = config.sw_config;

    // Step 3: Smith-Waterman forward + backward (computes posteriors)
    alignment::compute_alignment_with_posteriors<ScalarBackend>(
        workspace->similarity, L1, L2, align_config, workspace->sw_matrix, workspace->posteriors,
        partition, arena);

    // Step 4: Normalize posteriors to sum to 1
    float posterior_sum = reduce::matrix_sum<ScalarBackend>(workspace->posteriors, matrix_size);
    if (posterior_sum < 1e-10f) {
        *score_cosine = 0.0f;
        *score_magnitude = 0.0f;
        return;
    }
    for (int i = 0; i < matrix_size; i++) {
        workspace->posteriors[i] /= posterior_sum;
    }

    // Step 5: Compute embedding norms for both metrics
    float* norms1 = scope.allocate<float>(static_cast<size_t>(L1));
    float* norms2 = scope.allocate<float>(static_cast<size_t>(L2));
    if (!norms1 || !norms2) {
        *score_cosine = 0.0f;
        *score_magnitude = 0.0f;
        return;
    }

    for (int i = 0; i < L1; i++) {
        float sum_sq = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            float val = embeddings1[i * hidden_dim + k];
            sum_sq += val * val;
        }
        norms1[i] = std::sqrt(sum_sq);
    }

    for (int j = 0; j < L2; j++) {
        float sum_sq = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            float val = embeddings2[j * hidden_dim + k];
            sum_sq += val * val;
        }
        norms2[j] = std::sqrt(sum_sq);
    }

    // Step 6: Compute cosine similarity and norm products
    float* cosine_similarity = scope.allocate<float>(static_cast<size_t>(matrix_size));
    float* norm_products = scope.allocate<float>(static_cast<size_t>(matrix_size));
    if (!cosine_similarity || !norm_products) {
        *score_cosine = 0.0f;
        *score_magnitude = 0.0f;
        return;
    }

    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L2; j++) {
            int idx = i * L2 + j;
            float norm_product = norms1[i] * norms2[j];
            norm_products[idx] = norm_product;
            if (norm_product > 1e-10f) {
                cosine_similarity[idx] = workspace->similarity[idx] / norm_product;
            } else {
                cosine_similarity[idx] = 0.0f;
            }
        }
    }

    // Step 7: Compute cosine-based score
    float* weighted_cos = scope.allocate<float>(static_cast<size_t>(matrix_size));
    if (!weighted_cos) {
        *score_cosine = 0.0f;
        *score_magnitude = 0.0f;
        return;
    }

    reduce::elementwise_multiply<ScalarBackend>(cosine_similarity, workspace->posteriors,
                                                weighted_cos, matrix_size);
    *score_cosine = reduce::matrix_sum<ScalarBackend>(weighted_cos, matrix_size);

    // Step 8: Compute magnitude-aware score
    // score_mag = Sigma P[i,j] * dot[i,j] / Sigma P[i,j] * ||e1[i]|| * ||e2[j]||
    float* weighted_dot = scope.allocate<float>(static_cast<size_t>(matrix_size));
    if (!weighted_dot) {
        *score_magnitude = 0.0f;
        return;
    }

    reduce::elementwise_multiply<ScalarBackend>(workspace->similarity,  // dot products
                                                workspace->posteriors, weighted_dot, matrix_size);

    float numerator = reduce::matrix_sum<ScalarBackend>(weighted_dot, matrix_size);

    // Compute denominator: Sigma P[i,j] * ||e1[i]|| * ||e2[j]||
    float denominator = 0.0f;
    for (int idx = 0; idx < matrix_size; idx++) {
        denominator += workspace->posteriors[idx] * norm_products[idx];
    }

    *score_magnitude = (denominator > 1e-10f) ? (numerator / denominator) : 0.0f;
}

/**
 * Dual-score from coordinates: Coords -> MPNN -> Embeddings -> Both Scores
 */
template <>
void pairwise_align_with_dual_score<ScalarBackend>(
    const float* coords1, int L1, const float* coords2, int L2, const PairwiseConfig& config,
    const mpnn::MPNNWeights& weights, PairwiseWorkspace* workspace, float* partition,
    float* score_cosine, float* score_magnitude, pfalign::memory::GrowableArena* arena) {
    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        *partition = smith_waterman::NINF;
        *score_cosine = 0.0f;
        *score_magnitude = 0.0f;
        return;
    }

    // Step 1-2: Encode both proteins (optionally in parallel)
    RunMpnnEncoders<ScalarBackend>(config, weights, coords1, L1, coords2, L2,
                                   workspace->embeddings1, workspace->embeddings2,
                                   workspace->mpnn_ws1, workspace->mpnn_ws2);

    // Step 3: Compute both scores from embeddings
    pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
        workspace->embeddings1, L1, workspace->embeddings2, L2, config.mpnn_config.hidden_dim,
        config, workspace, partition, score_cosine, score_magnitude, arena);
}

/**
 * Full alignment from embeddings: Embeddings -> Similarity -> SW -> Score -> Decode
 */
template <>
void pairwise_align_from_embeddings_full<ScalarBackend>(
    const float* embeddings1, int L1, const float* embeddings2, int L2, int hidden_dim,
    const PairwiseConfig& config, PairwiseWorkspace* workspace, AlignmentResult* result,
    pfalign::memory::GrowableArena* arena, float gap_penalty) {
    // Validate inputs
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        result->partition = smith_waterman::NINF;
        result->score = 0.0f;
        result->path_length = 0;
        return;
    }

    if (!result->posteriors || !result->alignment_path || result->max_path_length < (L1 + L2)) {
        result->partition = smith_waterman::NINF;
        result->score = 0.0f;
        result->path_length = 0;
        return;
    }

    // Set sequence lengths
    result->L1 = L1;
    result->L2 = L2;

    // Step 1: Call pairwise_align_with_score to get partition + score + posteriors
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1, L1, embeddings2, L2, hidden_dim, config, workspace, &result->partition,
        &result->score, arena);

    // Step 2: Copy posteriors from workspace to result
    const int matrix_size = L1 * L2;
    std::memcpy(result->posteriors, workspace->posteriors,
                static_cast<size_t>(matrix_size) * sizeof(float));

    // Step 3: Decode alignment path from posteriors
    // Allocate temporary buffers for decode DP in arena
    pfalign::memory::ScopedGrowableArena scope(*arena);

    const int dp_size = (L1 + 1) * (L2 + 1);
    float* dp_score = scope.allocate<float>(static_cast<size_t>(dp_size));
    uint8_t* dp_traceback = scope.allocate<uint8_t>(static_cast<size_t>(dp_size));

    if (!dp_score || !dp_traceback) {
        result->path_length = 0;  // Decode failed - arena exhausted
        return;
    }

    // Decode using alignment_decode primitive
    int path_len = alignment_decode::decode_alignment<ScalarBackend>(
        result->posteriors, L1, L2, gap_penalty, result->alignment_path, result->max_path_length,
        dp_score, dp_traceback);

    result->path_length = (path_len > 0) ? path_len : 0;
}

/**
 * Full alignment from coordinates: Coords -> MPNN -> Full Alignment
 */
template <>
void pairwise_align_full<ScalarBackend>(const float* coords1, int L1, const float* coords2, int L2,
                                        const PairwiseConfig& config,
                                        const mpnn::MPNNWeights& weights,
                                        PairwiseWorkspace* workspace, AlignmentResult* result,
                                        pfalign::memory::GrowableArena* arena, float gap_penalty) {
    // Validate workspace capacity
    if (L1 > workspace->L1_max || L2 > workspace->L2_max) {
        result->partition = smith_waterman::NINF;
        result->score = 0.0f;
        result->path_length = 0;
        return;
    }

    // Preserve coordinate pointers for downstream output utilities.
    result->coords1 = coords1;
    result->coords2 = coords2;

    // Step 1-2: Encode both proteins (optionally in parallel)
    RunMpnnEncoders<ScalarBackend>(config, weights, coords1, L1, coords2, L2,
                                   workspace->embeddings1, workspace->embeddings2,
                                   workspace->mpnn_ws1, workspace->mpnn_ws2);

    // Step 3: Full alignment from embeddings
    pairwise_align_from_embeddings_full<ScalarBackend>(
        workspace->embeddings1, L1, workspace->embeddings2, L2, config.mpnn_config.hidden_dim,
        config, workspace, result, arena, gap_penalty);
}

}  // namespace pairwise
}  // namespace pfalign
