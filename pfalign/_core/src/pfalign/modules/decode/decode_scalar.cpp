/**
 * Scalar backend implementation for decode module.
 */

#include "decode.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/primitives/alignment_decode/alignment_decode.h"

namespace pfalign {
namespace decode {

template <>
int decode_alignment<ScalarBackend>(const float* posteriors, int L1, int L2,
                                    const DecodeConfig& config, AlignmentPair* alignment_path,
                                    int max_path_length, pfalign::memory::GrowableArena* arena) {
    // Validate inputs
    if (!posteriors || !alignment_path || !arena) {
        return -1;
    }
    if (L1 <= 0 || L2 <= 0 || max_path_length <= 0) {
        return -1;
    }

    // Allocate scratch buffers from arena
    const size_t dp_size_elements = static_cast<size_t>((L1 + 1) * (L2 + 1));

    float* dp_score = arena->allocate<float>(dp_size_elements);
    if (!dp_score) {
        return -1;  // Arena allocation failed
    }

    uint8_t* dp_traceback = arena->allocate<uint8_t>(dp_size_elements);
    if (!dp_traceback) {
        return -1;  // Arena allocation failed
    }

    // Call primitive
    return alignment_decode::decode_alignment<ScalarBackend>(posteriors, L1, L2, config.gap_penalty,
                                                             alignment_path, max_path_length,
                                                             dp_score, dp_traceback);
}

}  // namespace decode
}  // namespace pfalign
