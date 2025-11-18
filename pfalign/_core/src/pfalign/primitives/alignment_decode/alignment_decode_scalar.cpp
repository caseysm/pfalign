/**
 * Scalar backend implementation for alignment decoding.
 */

#include "alignment_decode.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>
#include <algorithm>

namespace pfalign {
namespace alignment_decode {

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * Safe log for posteriors: clamp zero to LOG_ZERO to avoid -inf.
 */
static inline float safe_log(float x) {
    return (x > 0.0f) ? std::log(x) : LOG_ZERO;
}

/**
 * Index into DP matrices: (i, j) in (L1+1) * (L2+1) matrix.
 */
static inline int dp_index(int i, int j, int stride) {
    return i * stride + j;
}

//==============================================================================
// Scalar Backend Implementation
//==============================================================================

template <>
int decode_alignment<ScalarBackend>(const float* posteriors, int L1, int L2, float gap_penalty,
                                    AlignmentPair* alignment_path, int max_path_length,
                                    float* dp_score, uint8_t* dp_traceback) {
    // Validate inputs
    if (L1 <= 0 || L2 <= 0 || !posteriors || !alignment_path || !dp_score || !dp_traceback ||
        max_path_length <= 0) {
        return -1;
    }

    const int dp_stride = L2 + 1;

    //--------------------------------------------------------------------------
    // Step 1: Initialize DP boundaries
    //--------------------------------------------------------------------------
    // V[0, 0] = 0 (log-space: log(1) = 0)
    dp_score[dp_index(0, 0, dp_stride)] = 0.0f;
    dp_traceback[dp_index(0, 0, dp_stride)] = START;

    // First row: V[0, j] = j * gap_penalty (gaps in seq1)
    for (int j = 1; j <= L2; j++) {
        dp_score[dp_index(0, j, dp_stride)] = j * gap_penalty;
        dp_traceback[dp_index(0, j, dp_stride)] = GAP_SEQ1;
    }

    // First column: V[i, 0] = i * gap_penalty (gaps in seq2)
    for (int i = 1; i <= L1; i++) {
        dp_score[dp_index(i, 0, dp_stride)] = i * gap_penalty;
        dp_traceback[dp_index(i, 0, dp_stride)] = GAP_SEQ2;
    }

    //--------------------------------------------------------------------------
    // Step 2: Fill DP matrix
    //--------------------------------------------------------------------------
    // V[i, j] = max(
    //     V[i-1, j-1] + log(P[i-1, j-1]),  // match
    //     V[i-1, j] + gap_penalty,          // gap in seq2
    //     V[i, j-1] + gap_penalty           // gap in seq1
    // )

    for (int i = 1; i <= L1; i++) {
        for (int j = 1; j <= L2; j++) {
            // Posterior for match (i-1, j-1) in 0-indexed posteriors
            float posterior = posteriors[(i - 1) * L2 + (j - 1)];
            float log_posterior = safe_log(posterior);

            // Three transitions
            float score_match = dp_score[dp_index(i - 1, j - 1, dp_stride)] + log_posterior;
            float score_gap_seq2 = dp_score[dp_index(i - 1, j, dp_stride)] + gap_penalty;
            float score_gap_seq1 = dp_score[dp_index(i, j - 1, dp_stride)] + gap_penalty;

            // Take maximum
            float max_score = score_match;
            uint8_t direction = MATCH;

            if (score_gap_seq2 > max_score) {
                max_score = score_gap_seq2;
                direction = GAP_SEQ2;
            }

            if (score_gap_seq1 > max_score) {
                max_score = score_gap_seq1;
                direction = GAP_SEQ1;
            }

            dp_score[dp_index(i, j, dp_stride)] = max_score;
            dp_traceback[dp_index(i, j, dp_stride)] = direction;
        }
    }

    //--------------------------------------------------------------------------
    // Step 3: Traceback from V[L1, L2] to V[0, 0]
    //--------------------------------------------------------------------------
    // Build path backwards, then reverse

    int path_len = 0;
    int i = L1;
    int j = L2;

    // Temporary buffer for reversed path (worst case: L1 + L2)
    // We'll write forwards into alignment_path, then reverse at the end
    AlignmentPair* temp_path = alignment_path;  // Reuse output buffer

    while (i > 0 || j > 0) {
        if (path_len >= max_path_length) {
            // Buffer overflow
            return -1;
        }

        uint8_t direction = dp_traceback[dp_index(i, j, dp_stride)];

        if (direction == MATCH) {
            // Match: move diagonally, record (i-1, j-1) in 0-indexed coords
            temp_path[path_len].i = i - 1;
            temp_path[path_len].j = j - 1;
            temp_path[path_len].posterior = posteriors[(i - 1) * L2 + (j - 1)];
            path_len++;
            i--;
            j--;
        } else if (direction == GAP_SEQ2) {
            // Gap in seq2: move up, record (i-1, -1)
            temp_path[path_len].i = i - 1;
            temp_path[path_len].j = -1;
            temp_path[path_len].posterior = 0.0f;
            path_len++;
            i--;
        } else if (direction == GAP_SEQ1) {
            // Gap in seq1: move left, record (-1, j-1)
            temp_path[path_len].i = -1;
            temp_path[path_len].j = j - 1;
            temp_path[path_len].posterior = 0.0f;
            path_len++;
            j--;
        } else {
            // START or invalid direction - should only happen at (0, 0)
            break;
        }
    }

    //--------------------------------------------------------------------------
    // Step 4: Reverse path (was built backwards)
    //--------------------------------------------------------------------------
    std::reverse(temp_path, temp_path + path_len);

    return path_len;
}

}  // namespace alignment_decode
}  // namespace pfalign
