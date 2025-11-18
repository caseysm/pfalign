/**
 * FASTA Alignment Writer
 *
 * Writes pairwise protein alignments to FASTA format with gap characters.
 * Takes AlignmentResult with decoded path and converts to gapped sequences.
 *
 * Output format:
 * ```
 * >protein1_id | Score: 0.845 | Partition: 123.45 | Length: 150
 * ACDEFG-HIKLMN---PQRSTVWY
 * >protein2_id | Length: 145
 * ACDEFGHIKLMN-PQRST---VWY
 * ```
 *
 * Gap character: '-' (standard FASTA gap)
 * Line wrapping: 80 characters per line (FASTA convention)
 */

#pragma once

#include "pfalign/adapters/alignment_types.h"
#include <string>
#include <vector>
#include <fstream>

namespace pfalign::io {

/**
 * Convert decoded alignment path to gapped sequences.
 *
 * Takes the alignment path from AlignmentResult and original sequences,
 * produces two gapped sequences where gaps are inserted as '-' characters.
 *
 * Algorithm:
 * - Iterate through alignment path
 * - For matches (i >= 0, j >= 0): add both residues
 * - For gap in seq1 (i == -1): add '-' to seq1, residue to seq2
 * - For gap in seq2 (j == -1): add residue to seq1, '-' to seq2
 *
 * Output sequences will have equal length (matches + gaps).
 *
 * @param alignment_path Decoded alignment path [path_length]
 * @param path_length Length of alignment path
 * @param seq1 Original sequence 1 (ungapped)
 * @param seq2 Original sequence 2 (ungapped)
 * @param gapped_seq1 Output: sequence 1 with gaps inserted
 * @param gapped_seq2 Output: sequence 2 with gaps inserted
 *
 * Example:
 *   seq1 = "ACDEFG"
 *   seq2 = "ACDEG"
 *   path = [(0,0), (1,1), (2,2), (3,-1), (4,3), (5,4)]
 *   gapped_seq1 = "ACDEFG"
 *   gapped_seq2 = "ACDE-G"
 */
void alignment_path_to_gapped_sequences(const AlignmentPair* alignment_path, int path_length,
                                        const std::string& seq1, const std::string& seq2,
                                        std::string& gapped_seq1, std::string& gapped_seq2);

/**
 * Write aligned sequences to FASTA format.
 *
 * Outputs two-sequence alignment in standard FASTA format with gaps.
 * Includes alignment metadata in headers (score, partition, lengths).
 *
 * Format:
 * ```
 * >id1 | Score: 0.845 | Partition: 123.45 | Length: 150
 * ACDEFGHIKLMNPQRSTVWY...
 * >id2 | Length: 145
 * ACDEFGHIKLMNPQRSTVWY...
 * ```
 *
 * Features:
 * - Line wrapping at 80 characters (FASTA standard)
 * - Gap character: '-'
 * - Metadata in header line
 * - Both sequences have equal length after gap insertion
 *
 * @param output_path File path to write (will overwrite if exists)
 * @param id1 Identifier for protein 1
 * @param id2 Identifier for protein 2
 * @param gapped_seq1 Gapped sequence 1
 * @param gapped_seq2 Gapped sequence 2
 * @param score Alignment score [0, 1]
 * @param partition Partition function (log-space)
 * @param original_length1 Original length of sequence 1 (before gaps)
 * @param original_length2 Original length of sequence 2 (before gaps)
 * @return true on success, false on file I/O error
 */
bool write_fasta_alignment(const std::string& output_path, const std::string& id1,
                           const std::string& id2, const std::string& gapped_seq1,
                           const std::string& gapped_seq2, float score, float partition,
                           int original_length1, int original_length2);

/**
 * High-level convenience function: write alignment from AlignmentResult.
 *
 * Takes full AlignmentResult and original sequences, handles all conversion
 * and formatting internally.
 *
 * This is the primary API for writing alignment output.
 *
 * @param output_path File path to write
 * @param id1 Identifier for protein 1
 * @param id2 Identifier for protein 2
 * @param seq1 Original ungapped sequence 1
 * @param seq2 Original ungapped sequence 2
 * @param alignment_path Decoded alignment path
 * @param path_length Length of alignment path
 * @param score Alignment score [0, 1]
 * @param partition Partition function
 * @return true on success, false on error
 *
 * Example usage:
 * ```cpp
 * AlignmentResult result;
 * // ... run pairwise_align_full ...
 *
 * std::string seq1 = extract_sequence(protein1);
 * std::string seq2 = extract_sequence(protein2);
 *
 * write_alignment_fasta(
 *     "output.fasta",
 *     "protein1", "protein2",
 *     seq1, seq2,
 *     result.alignment_path, result.path_length,
 *     result.score, result.partition
 * );
 * ```
 */
bool write_alignment_fasta(const std::string& output_path, const std::string& id1,
                           const std::string& id2, const std::string& seq1, const std::string& seq2,
                           const AlignmentPair* alignment_path, int path_length, float score,
                           float partition);

}  // namespace pfalign::io
