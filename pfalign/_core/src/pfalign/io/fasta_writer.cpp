/**
 * FASTA Alignment Writer - Implementation
 */

#include "fasta_writer.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#include "pfalign/adapters/alignment_types.h"

namespace pfalign::io {

void alignment_path_to_gapped_sequences(const AlignmentPair* alignment_path, int path_length,
                                        const std::string& seq1, const std::string& seq2,
                                        std::string& gapped_seq1, std::string& gapped_seq2) {
    // Reserve approximate space (original length + some gaps)
    gapped_seq1.clear();
    gapped_seq2.clear();
    gapped_seq1.reserve(static_cast<size_t>(path_length));
    gapped_seq2.reserve(static_cast<size_t>(path_length));

    for (int k = 0; k < path_length; k++) {
        int i = alignment_path[k].i;
        int j = alignment_path[k].j;

        if (i == -1) {
            // Gap in seq1
            gapped_seq1 += '-';
            if (j >= 0 && j < static_cast<int>(seq2.length())) {
                gapped_seq2 += seq2[static_cast<size_t>(j)];
            } else {
                gapped_seq2 += 'X';  // Error: invalid index
            }
        } else if (j == -1) {
            // Gap in seq2
            if (i >= 0 && i < static_cast<int>(seq1.length())) {
                gapped_seq1 += seq1[static_cast<size_t>(i)];
            } else {
                gapped_seq1 += 'X';  // Error: invalid index
            }
            gapped_seq2 += '-';
        } else {
            // Match: both residues present
            if (i >= 0 && i < static_cast<int>(seq1.length())) {
                gapped_seq1 += seq1[static_cast<size_t>(i)];
            } else {
                gapped_seq1 += 'X';  // Error: invalid index
            }

            if (j >= 0 && j < static_cast<int>(seq2.length())) {
                gapped_seq2 += seq2[static_cast<size_t>(j)];
            } else {
                gapped_seq2 += 'X';  // Error: invalid index
            }
        }
    }
}

bool write_fasta_alignment(const std::string& output_path, const std::string& id1,
                           const std::string& id2, const std::string& gapped_seq1,
                           const std::string& gapped_seq2, float score, float partition,
                           int original_length1, int original_length2) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_path << std::endl;
        return false;
    }

    // Verify gapped sequences have equal length
    if (gapped_seq1.length() != gapped_seq2.length()) {
        std::cerr << "Error: Gapped sequences have different lengths (" << gapped_seq1.length()
                  << " vs " << gapped_seq2.length() << ")" << std::endl;
        return false;
    }

    // Write header for sequence 1 with metadata
    out << ">" << id1 << " | Score: " << std::fixed << std::setprecision(3) << score
        << " | Partition: " << std::fixed << std::setprecision(2) << partition
        << " | Length: " << original_length1 << " | Gapped: " << gapped_seq1.length() << "\n";

    // Write gapped sequence 1 with line wrapping (80 chars)
    const int line_width = 80;
    for (size_t i = 0; i < gapped_seq1.length(); i += line_width) {
        out << gapped_seq1.substr(i, line_width) << "\n";
    }

    // Write header for sequence 2
    out << ">" << id2 << " | Length: " << original_length2 << " | Gapped: " << gapped_seq2.length()
        << "\n";

    // Write gapped sequence 2 with line wrapping
    for (size_t i = 0; i < gapped_seq2.length(); i += line_width) {
        out << gapped_seq2.substr(i, line_width) << "\n";
    }

    out.close();
    return true;
}

bool write_alignment_fasta(const std::string& output_path, const std::string& id1,
                           const std::string& id2, const std::string& seq1, const std::string& seq2,
                           const AlignmentPair* alignment_path, int path_length, float score,
                           float partition) {
    // Step 1: Convert alignment path to gapped sequences
    std::string gapped_seq1, gapped_seq2;
    alignment_path_to_gapped_sequences(alignment_path, path_length, seq1, seq2, gapped_seq1,
                                       gapped_seq2);

    // Step 2: Write to FASTA format
    return write_fasta_alignment(output_path, id1, id2, gapped_seq1, gapped_seq2, score, partition,
                                 static_cast<int>(seq1.length()), static_cast<int>(seq2.length()));
}

}  // namespace pfalign::io
