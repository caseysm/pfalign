/**
 * Alignment Format Adapters
 *
 * Utilities for converting between different alignment representations:
 * - Pairwise edge list (AlignmentPair) -> MSA column format (AlignmentColumn)
 * - Alignment paths -> Profile-ready structures
 *
 * These adapters enable seamless integration between pairwise alignment
 * and progressive MSA by bridging data format gaps.
 */

#pragma once

#include "alignment_types.h"
#include "pfalign/modules/msa/profile.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace adapters {

/**
 * Convert pairwise alignment path to MSA column format.
 *
 * Transforms AlignmentPair edge list (i, j, posterior) into AlignmentColumn
 * array suitable for Profile::from_alignment().
 *
 * Edge list format (pairwise):
 *   - Sparse representation of aligned positions
 *   - Gaps represented as i=-1 or j=-1
 *   - Sequential order maintained
 *
 * Column format (MSA):
 *   - Dense representation with one entry per sequence per column
 *   - Each column contains positions for all sequences
 *   - Gaps represented as pos=-1 in AlignmentPosition
 *
 * Example conversion:
 *   Input (edge list):
 *     AlignmentPair path[] = {
 *         {0, 0, 0.9},   // Match: seq1[0] ↔ seq2[0]
 *         {1, -1, 0.0},  // Gap in seq2: seq1[1] ↔ gap
 *         {-1, 1, 0.0},  // Gap in seq1: gap ↔ seq2[1]
 *         {2, 2, 0.85}   // Match: seq1[2] ↔ seq2[2]
 *     };
 *
 *   Output (columns):
 *     columns[0].positions = [{0, 0}, {1, 0}]   // Both at pos 0
 *     columns[1].positions = [{0, 1}, {1, -1}]  // seq2 has gap
 *     columns[2].positions = [{0, -1}, {1, 1}]  // seq1 has gap
 *     columns[3].positions = [{0, 2}, {1, 2}]   // Both at pos 2
 *
 * @param path          Input pairwise alignment path (edge list)
 * @param path_length   Number of AlignmentPairs in path
 * @param seq1_idx      Global sequence index for first sequence
 * @param seq2_idx      Global sequence index for second sequence
 * @param columns       Output MSA columns [path_length] (pre-allocated)
 */
void pairwise_to_columns(const AlignmentPair* path, int path_length, int seq1_idx, int seq2_idx,
                         msa::AlignmentColumn* columns);

/**
 * Convert pairwise AlignmentResult to profile-ready column format.
 *
 * Convenience wrapper that:
 * 1. Extracts alignment_path from AlignmentResult
 * 2. Allocates columns array from arena
 * 3. Calls pairwise_to_columns() to populate columns
 *
 * This is the recommended interface for MSA code as it handles
 * memory management automatically.
 *
 * Usage:
 *   AlignmentResult result = pairwise_align(...);
 *
 *   int aligned_length;
 *   AlignmentColumn* columns = alignment_result_to_columns(
 *       result, seq1_idx, seq2_idx, &arena, &aligned_length
 *   );
 *
 *   Profile* merged = Profile::from_alignment(
 *       *profile1, *profile2, columns, aligned_length, &arena
 *   );
 *
 * @param result            Pairwise alignment result (contains path)
 * @param seq1_idx          Global sequence index for first sequence
 * @param seq2_idx          Global sequence index for second sequence
 * @param arena             Arena for allocating columns array
 * @param out_aligned_length Output: number of columns (= path_length)
 * @return                  Pointer to allocated columns array
 */
msa::AlignmentColumn* alignment_result_to_columns(const pairwise::AlignmentResult& result,
                                                  int seq1_idx, int seq2_idx,
                                                  pfalign::memory::GrowableArena* arena,
                                                  int* out_aligned_length);

/**
 * Validate alignment path for conversion.
 *
 * Checks that the alignment path is well-formed:
 * - No invalid indices (i < -1 or j < -1)
 * - Both i and j not simultaneously -1 (empty column)
 * - Sequential ordering maintained
 *
 * Throws std::invalid_argument if validation fails.
 *
 * @param path        Alignment path to validate
 * @param path_length Number of pairs in path
 * @param L1          Length of sequence 1 (for bounds checking)
 * @param L2          Length of sequence 2 (for bounds checking)
 */
void validate_alignment_path(const AlignmentPair* path, int path_length, int L1, int L2);

}  // namespace adapters
}  // namespace pfalign
