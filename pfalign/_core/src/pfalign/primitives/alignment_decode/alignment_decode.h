/**
 * Alignment decoding: convert posterior probability matrix to discrete path.
 *
 * Implements MAP (Maximum A Posteriori) decoding via Viterbi dynamic
 * programming in log-space. Converts soft alignment posteriors from
 * Smith-Waterman backward pass into a discrete alignment path with gaps.
 *
 * All operations support backend dispatch (Scalar, NEON, CUDA).
 */

#pragma once

#include "pfalign/adapters/alignment_types.h"
#include <cmath>

namespace pfalign {
namespace alignment_decode {

/**
 * Sentinel value for log(0) to avoid -inf in log-space computation.
 *
 * log(1e-9) ~= -20.7, chosen to be:
 * - Large enough that exp(LOG_ZERO) ~= 0 (negligible probability)
 * - Small enough to avoid underflow in subsequent additions
 * - Dominated by typical gap penalties (-2 to -5) in DP
 */
constexpr float LOG_ZERO = -20.0f;

/**
 * Decode posterior probability matrix into discrete alignment path.
 *
 * Uses MAP (Maximum A Posteriori) decoding via Viterbi DP in log-space.
 * The algorithm finds the single most likely alignment path given the
 * posterior probability matrix from Smith-Waterman backward pass.
 *
 * All memory must be pre-allocated by caller (no internal allocation).
 * This is a primitive operation designed for zero-allocation contexts
 * where scratch buffers are managed by an arena allocator.
 *
 * @tparam Backend Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param posteriors Posterior matrix [L1 * L2] from SW backward pass (row-major)
 *                   Must sum to 1.0 (probability distribution). Zero entries are
 *                   treated as LOG_ZERO in log-space.
 * @param L1 Length of sequence 1 (number of rows in posteriors)
 * @param L2 Length of sequence 2 (number of columns in posteriors)
 * @param gap_penalty Log-probability penalty for inserting a gap.
 *                    Typical values: -2.0 (lenient) to -5.0 (stringent).
 *                    Interpretation: gap_penalty = log(P_gap)
 * @param alignment_path Output buffer [max_path_length] for decoded alignment.
 *                       Caller allocated. Will be filled with AlignmentPair
 *                       structs representing matches and gaps.
 * @param max_path_length Maximum length of output buffer, typically L1 + L2
 *                        (worst case: all gaps). Function returns -1 if
 *                        actual path exceeds this length.
 * @param dp_score Scratch buffer [(L1+1) * (L2+1)] for DP scores in log-space.
 *                 Size: (L1+1) * (L2+1) * sizeof(float) bytes.
 *                 For L1=L2=200: 201 * 201 * 4 ~= 162 KB.
 *                 Contents undefined after return (can be reused).
 * @param dp_traceback Scratch buffer [(L1+1) * (L2+1)] for traceback directions.
 *                     Size: (L1+1) * (L2+1) * sizeof(uint8_t) bytes.
 *                     For L1=L2=200: 201 * 201 * 1 ~= 40 KB.
 *                     Contents undefined after return (can be reused).
 *
 * @return Actual length of alignment path (number of AlignmentPairs written),
 *         or -1 on error:
 *         - Buffer overflow (path length > max_path_length)
 *         - Invalid input (L1 <= 0, L2 <= 0, null pointers)
 *
 * Algorithm outline:
 *   1. Initialize DP boundaries with gap penalties in log-space
 *   2. Fill DP matrix: V[i,j] = max(match, gap_seq1, gap_seq2)
 *   3. Traceback from V[L1,L2] to V[0,0]
 *   4. Reverse path (built backwards)
 *
 * Indexing:
 *   - posteriors[i * L2 + j]: posterior probability P(i aligns to j)
 *   - dp_score[(i) * (L2+1) + (j)]: DP score at position (i, j)
 *   - dp_traceback[(i) * (L2+1) + (j)]: traceback direction at (i, j)
 *
 * Gap representation in alignment_path:
 *   - Gap in seq1: AlignmentPair{-1, j, 0.0}
 *   - Gap in seq2: AlignmentPair{i, -1, 0.0}
 *   - Match: AlignmentPair{i, j, posteriors[i*L2 + j]}
 *
 * Monotonicity guarantee:
 *   Output path is monotonic: both i and j are non-decreasing
 *   (accounting for gaps: -1 positions are skipped in monotonicity check).
 *
 * Example usage:
 * ```cpp
 *   // Setup (L1=3, L2=3)
 *   float posteriors[9] = {...};     // 3*3 posterior matrix
 *   AlignmentPair path[6];            // Max length 3+3
 *   float dp_score[16];               // (3+1) * (3+1)
 *   uint8_t dp_traceback[16];
 *
 *   // Decode with gap penalty = log(0.1) ~= -2.3
 *   int path_len = decode_alignment<ScalarBackend>(
 *       posteriors, 3, 3, -2.3f,
 *       path, 6,
 *       dp_score, dp_traceback
 *   );
 *
 *   if (path_len > 0) {
 *       for (int k = 0; k < path_len; k++) {
 *           if (path[k].i == -1) {
 *               printf("Gap in seq1\n");
 *           } else if (path[k].j == -1) {
 *               printf("Gap in seq2\n");
 *           } else {
 *               printf("Match: %d -> %d (p=%.3f)\n",
 *                      path[k].i, path[k].j, path[k].posterior);
 *           }
 *       }
 *   }
 * ```
 *
 * Performance notes:
 *   - Scalar backend: O(L1 * L2) time, cache-friendly row-major access
 *   - NEON/SIMD: Horizontal max reduction limits vectorization, but
 *     log/exp can be vectorized in preprocessing
 *   - CUDA: Limited parallelism (DP is inherently sequential), but
 *     multiple sequences can be decoded in parallel
 *
 * Numerical stability:
 *   - All computation in log-space to avoid underflow
 *   - Zero posteriors clamped to LOG_ZERO before log()
 *   - Gap penalty typically dominates LOG_ZERO in DP
 */
template <typename Backend>
int decode_alignment(const float* posteriors, int L1, int L2, float gap_penalty,
                     AlignmentPair* alignment_path, int max_path_length, float* dp_score,
                     uint8_t* dp_traceback);

}  // namespace alignment_decode
}  // namespace pfalign
