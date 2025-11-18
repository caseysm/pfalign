/**
 * Decode module for extracting discrete alignments from posteriors.
 *
 * Provides clean interface to alignment_decode primitive with:
 * - MAP (Viterbi) decoding from posterior probability matrices
 * - Gap penalty configuration
 * - Memory management helpers
 *
 * This module wraps the alignment_decode primitive similar to how
 * alignment wraps the smith_waterman primitive.
 */

#pragma once

#include "pfalign/primitives/alignment_decode/alignment_decode.h"
#include "pfalign/adapters/alignment_types.h"
#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace decode {

/**
 * Decode configuration.
 */
struct DecodeConfig {
    float gap_penalty;  ///< Log-probability penalty for gaps (typical: -2.0 to -5.0)

    DecodeConfig()
        : gap_penalty(-2.3f)  // log(0.1) ~= -2.3
    {
    }

    DecodeConfig(float gap_pen) : gap_penalty(gap_pen) {
    }
};

/**
 * Decode posterior matrix into discrete alignment path.
 *
 * This is a thin wrapper around the alignment_decode primitive that:
 * - Provides clean interface for callers
 * - Manages scratch buffer allocation via arena
 * - Returns structured result with path length
 *
 * Uses MAP (Maximum A Posteriori) decoding via Viterbi DP in log-space
 * to find the single most likely alignment path given the posterior
 * probability matrix from Smith-Waterman backward pass.
 *
 * @tparam Backend      Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param posteriors    Posterior matrix [L1 * L2] from SW backward pass (row-major)
 *                      Must sum to 1.0 (probability distribution).
 * @param L1            Length of sequence 1
 * @param L2            Length of sequence 2
 * @param config        Decode configuration (gap penalty)
 * @param alignment_path Output buffer [max_path_length] for decoded alignment
 *                      (caller allocated)
 * @param max_path_length Maximum length of output buffer (typically L1 + L2)
 * @param arena         Arena allocator for scratch buffers (~200KB for L=200)
 *
 * @return Actual length of alignment path, or -1 on error:
 *         - Buffer overflow (path length > max_path_length)
 *         - Invalid input (L1 <= 0, L2 <= 0, null pointers)
 *         - Arena allocation failure
 *
 * Scratch buffer allocation (from arena):
 *   - DP score matrix: (L1+1) * (L2+1) * sizeof(float) bytes
 *   - DP traceback: (L1+1) * (L2+1) * sizeof(uint8_t) bytes
 *   - For L1=L2=200: ~162 KB + ~40 KB = ~200 KB total
 *
 * Example:
 * ```cpp
 *   using pfalign::ScalarBackend;
 *   using pfalign::decode::decode_alignment;
 *   using pfalign::decode::DecodeConfig;
 *   using pfalign::AlignmentPair;
 *
 *   float posteriors[100 * 150];  // From SW backward pass
 *   AlignmentPair path[250];       // Max length: 100 + 150
 *   pfalign::memory::Arena arena(1024 * 1024);  // 1 MB
 *
 *   DecodeConfig config(-2.3f);  // Gap penalty = log(0.1)
 *
 *   int path_len = decode_alignment<ScalarBackend>(
 *       posteriors, 100, 150,
 *       config,
 *       path, 250,
 *       &arena
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
 */
template <typename Backend>
int decode_alignment(const float* posteriors, int L1, int L2, const DecodeConfig& config,
                     AlignmentPair* alignment_path, int max_path_length,
                     pfalign::memory::GrowableArena* arena);

/**
 * Get required scratch buffer size for decoding.
 *
 * Utility function to help callers size their arena allocators.
 *
 * @param L1 Length of first sequence
 * @param L2 Length of second sequence
 * @return   Required scratch size in bytes
 *
 * Example:
 * ```cpp
 *   size_t scratch_size = get_decode_scratch_size(100, 150);
 *   pfalign::memory::Arena arena(scratch_size);
 * ```
 */
inline size_t get_decode_scratch_size(int L1, int L2) {
    size_t dp_score_size = static_cast<size_t>((L1 + 1) * (L2 + 1)) * sizeof(float);
    size_t dp_traceback_size = static_cast<size_t>((L1 + 1) * (L2 + 1)) * sizeof(uint8_t);
    return dp_score_size + dp_traceback_size;
}

}  // namespace decode
}  // namespace pfalign
