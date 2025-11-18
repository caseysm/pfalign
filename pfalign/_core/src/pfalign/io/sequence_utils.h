/**
 * Sequence utilities for amino acid conversion and manipulation.
 *
 * Provides utilities for:
 * - Converting 3-letter amino acid codes to 1-letter codes
 * - Extracting sequences from PDB residues
 * - Inserting gaps into sequences based on alignment paths
 *
 * Mappings are BioPython-compatible (Bio.Data.IUPACData).
 */

#pragma once

#include "protein_structure.h"
#include "amino_acids.h"
#include "pfalign/adapters/alignment_types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace pfalign {
namespace io {

using pfalign::io::three_to_one;

/**
 * Extract sequence from residues as 1-letter codes.
 *
 * Converts each residue's 3-letter code to 1-letter code.
 * Unknown residues are represented as 'X'.
 *
 * @param residues Vector of Residue structs
 * @return Sequence string (1-letter codes)
 *
 * Example:
 * ```cpp
 *   std::vector<Residue> residues = {
 *       Residue(1, ' ', "Ala"),
 *       Residue(2, ' ', "Cys"),
 *       Residue(3, ' ', "Glu")
 *   };
 *   std::string seq = extract_sequence(residues);  // -> "ACE"
 * ```
 */
inline std::string extract_sequence(const std::vector<Residue>& residues) {
    std::string sequence;
    sequence.reserve(residues.size());

    for (const auto& res : residues) {
        sequence += three_to_one(res.resn);
    }

    return sequence;
}

/**
 * Extract sequence from a specific chain.
 *
 * @param chain Chain struct containing residues
 * @return Sequence string (1-letter codes)
 *
 * Example:
 * ```cpp
 *   const Chain& chain = protein.get_chain(0);
 *   std::string seq = extract_sequence(chain);
 * ```
 */
inline std::string extract_sequence(const Chain& chain) {
    return extract_sequence(chain.residues);
}

/**
 * Insert gaps into a sequence based on alignment path.
 *
 * Creates an aligned sequence string with gaps ('-') inserted at positions
 * where the alignment path indicates gaps.
 *
 * For sequence 1 (query): gaps are inserted where path[k].i == -1
 * For sequence 2 (target): gaps are inserted where path[k].j == -1
 *
 * @param sequence Original sequence (no gaps)
 * @param alignment_path Alignment path from Viterbi decode
 * @param path_length Number of entries in alignment_path
 * @param is_seq1 True for sequence 1 (query), false for sequence 2 (target)
 * @return Aligned sequence with gaps
 *
 * Example:
 * ```cpp
 *   // Original sequences
 *   std::string seq1 = "ACE";
 *   std::string seq2 = "ABCD";
 *
 *   // Alignment path: [(0,0), (1,-1), (2,1), (?,2), (?,3)]
 *   AlignmentPair path[] = {
 *       {0, 0, 0.9},   // A-A match
 *       {1, -1, 0.0},  // Gap in seq2
 *       {2, 1, 0.8}    // E-B match
 *   };
 *
 *   std::string aligned1 = insert_gaps(seq1, path, 3, true);   // -> "ACE"
 *   std::string aligned2 = insert_gaps(seq2, path, 3, false);  // -> "A-B"
 * ```
 */
inline std::string insert_gaps(const std::string& sequence, const AlignmentPair* alignment_path,
                               int path_length, bool is_seq1) {
    std::string aligned;
    aligned.reserve(static_cast<size_t>(path_length));  // Upper bound: path length

    for (int k = 0; k < path_length; k++) {
        const AlignmentPair& pair = alignment_path[k];

        if (is_seq1) {
            // Sequence 1: check if position i is valid
            if (pair.i == -1) {
                aligned += '-';  // Gap in seq1
            } else {
                if (pair.i >= 0 && pair.i < static_cast<int>(sequence.size())) {
                    aligned += sequence[static_cast<size_t>(pair.i)];
                } else {
                    aligned += 'X';  // Out of bounds (shouldn't happen)
                }
            }
        } else {
            // Sequence 2: check if position j is valid
            if (pair.j == -1) {
                aligned += '-';  // Gap in seq2
            } else {
                if (pair.j >= 0 && pair.j < static_cast<int>(sequence.size())) {
                    aligned += sequence[static_cast<size_t>(pair.j)];
                } else {
                    aligned += 'X';  // Out of bounds (shouldn't happen)
                }
            }
        }
    }

    return aligned;
}

}  // namespace io
}  // namespace pfalign
