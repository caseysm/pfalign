/**
 * Scalar backend implementation for alignment module.
 */

#include "alignment.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"

namespace pfalign {
namespace alignment {

template <>
void compute_alignment<ScalarBackend>(const float* scores, int L1, int L2,
                                      const AlignmentConfig& config, float* dp_matrix,
                                      float* partition,
                                      [[maybe_unused]] pfalign::memory::GrowableArena* temp_arena) {
    using namespace smith_waterman;

    switch (config.mode) {
        case AlignmentMode::JAX_REGULAR:
            smith_waterman_jax_regular<ScalarBackend>(scores, L1, L2, config.sw_config, dp_matrix,
                                                      partition);
            break;

        case AlignmentMode::JAX_AFFINE:
            smith_waterman_jax_affine<ScalarBackend>(scores, L1, L2, config.sw_config, dp_matrix,
                                                     partition);
            break;

        case AlignmentMode::JAX_AFFINE_FLEXIBLE:
            smith_waterman_jax_affine_flexible<ScalarBackend>(scores, L1, L2, config.sw_config,
                                                              dp_matrix, partition);
            break;

        case AlignmentMode::DIRECT_REGULAR:
            smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config.sw_config,
                                                         dp_matrix, partition);
            break;

        case AlignmentMode::DIRECT_AFFINE:
            smith_waterman_direct_affine<ScalarBackend>(scores, L1, L2, config.sw_config, dp_matrix,
                                                        partition);
            break;

        case AlignmentMode::DIRECT_AFFINE_FLEXIBLE:
            smith_waterman_direct_affine_flexible<ScalarBackend>(scores, L1, L2, config.sw_config,
                                                                 dp_matrix, partition);
            break;
    }
}

template <>
void compute_alignment_with_posteriors<ScalarBackend>(const float* scores, int L1, int L2,
                                                      const AlignmentConfig& config,
                                                      float* dp_matrix, float* posteriors,
                                                      float* partition,
                                                      pfalign::memory::GrowableArena* temp_arena) {
    using namespace smith_waterman;

    // Forward pass (populates dp_matrix and partition)
    compute_alignment<ScalarBackend>(scores, L1, L2, config, dp_matrix, partition, temp_arena);

    // Backward pass (computes posteriors from dp_matrix)
    switch (config.mode) {
        case AlignmentMode::JAX_REGULAR:
            smith_waterman_jax_regular_backward<ScalarBackend>(
                dp_matrix, scores, L1, L2, config.sw_config, *partition, posteriors, temp_arena);
            break;

        case AlignmentMode::JAX_AFFINE:
            smith_waterman_jax_affine_backward<ScalarBackend>(
                dp_matrix, scores, L1, L2, config.sw_config, *partition, posteriors, temp_arena);
            break;

        case AlignmentMode::JAX_AFFINE_FLEXIBLE:
            smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
                dp_matrix, scores, L1, L2, config.sw_config, *partition, posteriors, temp_arena);
            break;

        case AlignmentMode::DIRECT_REGULAR:
            smith_waterman_direct_regular_backward<ScalarBackend>(
                dp_matrix, scores, L1, L2, config.sw_config, *partition, posteriors, temp_arena);
            break;

        case AlignmentMode::DIRECT_AFFINE:
            smith_waterman_direct_affine_backward<ScalarBackend>(
                dp_matrix, scores, L1, L2, config.sw_config, *partition, posteriors, temp_arena);
            break;

        case AlignmentMode::DIRECT_AFFINE_FLEXIBLE:
            smith_waterman_direct_affine_flexible_backward<ScalarBackend>(
                dp_matrix, scores, L1, L2, config.sw_config, *partition, posteriors, temp_arena);
            break;
    }
}

}  // namespace alignment
}  // namespace pfalign
