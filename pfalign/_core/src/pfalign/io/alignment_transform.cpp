/**
 * Alignment transformation utilities implementation.
 */

#include "alignment_formats.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace pfalign {
namespace io {

// ============================================================================
// AlignmentTransform
// ============================================================================

MultipleAlignment AlignmentTransform::fill_a3m_gaps(const MultipleAlignment& aln) {
    if (aln.sequences.empty()) return aln;

    MultipleAlignment result = aln;

    // Find maximum insert length after each match state
    std::vector<size_t> max_insert_len;

    for (const auto& seq : result.sequences) {
        std::vector<size_t> insert_lens;
        std::string current_insert;

        for (char c : seq) {
            if (std::isupper(c) || c == '-' || c == '~' || std::isdigit(c)) {
                // Match state - save insert length and start new insert
                insert_lens.push_back(current_insert.length());
                current_insert.clear();
            } else if (std::islower(c)) {
                // Insert state
                current_insert += c;
            }
        }
        // Final insert
        insert_lens.push_back(current_insert.length());

        // Update max lengths
        if (max_insert_len.size() < insert_lens.size()) {
            max_insert_len.resize(insert_lens.size(), 0);
        }
        for (size_t i = 0; i < insert_lens.size(); ++i) {
            max_insert_len[i] = std::max(max_insert_len[i], insert_lens[i]);
        }
    }

    // Fill gaps in all sequences
    for (auto& seq : result.sequences) {
        std::string filled;
        std::string current_insert;
        size_t insert_idx = 0;

        for (char c : seq) {
            if (std::isupper(c) || c == '-' || c == '~' || std::isdigit(c)) {
                // Add gaps to pad current insert
                size_t insert_len = current_insert.length();
                filled += current_insert;
                for (size_t i = insert_len; i < max_insert_len[insert_idx]; ++i) {
                    filled += '.';
                }
                filled += c;
                current_insert.clear();
                insert_idx++;
            } else if (std::islower(c)) {
                current_insert += c;
            }
        }

        // Handle final insert
        filled += current_insert;
        for (size_t i = current_insert.length(); i < max_insert_len[insert_idx]; ++i) {
            filled += '.';
        }

        seq = filled;
    }

    return result;
}

MultipleAlignment AlignmentTransform::assign_match_states_first(const MultipleAlignment& aln) {
    if (aln.sequences.empty()) return aln;

    MultipleAlignment result = aln;

    // Find first non-ss/sa sequence
    size_t first_seq_idx = 0;
    for (size_t i = 0; i < result.names.size(); ++i) {
        if (result.names[i].find("ss_") != 0 && result.names[i].find("sa_") != 0) {
            first_seq_idx = i;
            break;
        }
    }

    const std::string& first_seq = result.sequences[first_seq_idx];

    // Determine which columns are match states (have residue in first sequence)
    std::vector<bool> is_match(first_seq.length());
    for (size_t i = 0; i < first_seq.length(); ++i) {
        is_match[i] = (first_seq[i] != '.' && first_seq[i] != '-');
    }

    // Apply match state assignment to all sequences
    for (auto& seq : result.sequences) {
        for (size_t i = 0; i < seq.length() && i < is_match.size(); ++i) {
            if (is_match[i]) {
                // Match column
                if (seq[i] == '.') {
                    seq[i] = '-';
                } else if (std::islower(seq[i])) {
                    seq[i] = std::toupper(seq[i]);
                }
            } else {
                // Insert column
                if (seq[i] == '-') {
                    seq[i] = '.';
                } else if (std::isupper(seq[i])) {
                    seq[i] = std::tolower(seq[i]);
                }
            }
        }
    }

    return result;
}

MultipleAlignment AlignmentTransform::assign_match_states_gap_rule(
    const MultipleAlignment& aln,
    int threshold_percent
) {
    if (aln.sequences.empty()) return aln;

    MultipleAlignment result = aln;
    size_t alignment_len = result.alignment_length();
    size_t num_seqs = result.num_sequences();

    // Count gaps per column
    std::vector<size_t> gap_counts(alignment_len, 0);
    for (const auto& seq : result.sequences) {
        for (size_t i = 0; i < seq.length(); ++i) {
            if (seq[i] == '.' || seq[i] == '-') {
                gap_counts[i]++;
            }
        }
    }

    // Determine match columns (< threshold% gaps)
    std::vector<bool> is_match(alignment_len);
    for (size_t i = 0; i < alignment_len; ++i) {
        double gap_percent = (100.0 * gap_counts[i]) / num_seqs;
        is_match[i] = (gap_percent < threshold_percent);
    }

    // Apply match state assignment
    for (auto& seq : result.sequences) {
        for (size_t i = 0; i < seq.length(); ++i) {
            if (is_match[i]) {
                // Match column
                if (seq[i] == '.') {
                    seq[i] = '-';
                } else if (std::islower(seq[i])) {
                    seq[i] = std::toupper(seq[i]);
                }
            } else {
                // Insert column
                if (seq[i] == '-') {
                    seq[i] = '.';
                } else if (std::isupper(seq[i])) {
                    seq[i] = std::tolower(seq[i]);
                }
            }
        }
    }

    return result;
}

MultipleAlignment AlignmentTransform::remove_gapped_columns(
    const MultipleAlignment& aln,
    int threshold_percent
) {
    if (aln.sequences.empty()) return aln;

    size_t alignment_len = aln.alignment_length();
    size_t num_seqs = aln.num_sequences();

    // Count gaps per column
    std::vector<size_t> gap_counts(alignment_len, 0);
    for (const auto& seq : aln.sequences) {
        for (size_t i = 0; i < seq.length(); ++i) {
            if (seq[i] == '.' || seq[i] == '-') {
                gap_counts[i]++;
            }
        }
    }

    // Determine which columns to keep
    std::vector<bool> keep_column(alignment_len);
    for (size_t i = 0; i < alignment_len; ++i) {
        double gap_percent = (100.0 * gap_counts[i]) / num_seqs;
        keep_column[i] = (gap_percent < threshold_percent);
    }

    // Build new sequences with only kept columns
    MultipleAlignment result = aln;
    for (auto& seq : result.sequences) {
        std::string new_seq;
        for (size_t i = 0; i < seq.length(); ++i) {
            if (keep_column[i]) {
                new_seq += seq[i];
            }
        }
        seq = new_seq;
    }

    return result;
}

MultipleAlignment AlignmentTransform::remove_inserts(const MultipleAlignment& aln) {
    MultipleAlignment result = aln;

    for (auto& seq : result.sequences) {
        // Remove lowercase letters and '.' characters
        seq.erase(
            std::remove_if(seq.begin(), seq.end(), [](char c) {
                return std::islower(c) || c == '.';
            }),
            seq.end()
        );
    }

    return result;
}

MultipleAlignment AlignmentTransform::to_uppercase(const MultipleAlignment& aln) {
    MultipleAlignment result = aln;

    for (auto& seq : result.sequences) {
        std::transform(seq.begin(), seq.end(), seq.begin(), ::toupper);
    }

    return result;
}

MultipleAlignment AlignmentTransform::to_lowercase(const MultipleAlignment& aln) {
    MultipleAlignment result = aln;

    for (auto& seq : result.sequences) {
        std::transform(seq.begin(), seq.end(), seq.begin(), ::tolower);
    }

    return result;
}

MultipleAlignment AlignmentTransform::remove_secondary_structure(const MultipleAlignment& aln) {
    MultipleAlignment result;
    result.title = aln.title;

    // Only copy non-secondary-structure sequences
    for (size_t i = 0; i < aln.num_sequences(); ++i) {
        const std::string& name = aln.names[i];
        if (name.find("ss_") != 0 && name.find("sa_") != 0 && name.find("aa_") != 0) {
            result.names.push_back(aln.names[i]);
            result.sequences.push_back(aln.sequences[i]);
        }
    }

    // Don't copy secondary structure annotations
    return result;
}

MultipleAlignment AlignmentTransform::apply_options(
    MultipleAlignment aln,
    const AlignmentFormatWriter::Options& opts,
    AlignmentFormat target_format
) {
    // 1. Fill A3M gaps if converting FROM A3M to other format
    // (Not needed for writing - just for format conversion)

    // 2. Match state assignment
    if (opts.match_mode == AlignmentFormatWriter::Options::MatchMode::FIRST_SEQUENCE) {
        aln = assign_match_states_first(aln);
    } else if (opts.match_mode == AlignmentFormatWriter::Options::MatchMode::GAP_THRESHOLD) {
        aln = assign_match_states_gap_rule(aln, opts.gap_threshold_percent);
    }

    // 3. Remove gapped columns
    if (opts.remove_gapped_threshold > 0) {
        aln = remove_gapped_columns(aln, opts.remove_gapped_threshold);
    }

    // 4. Remove inserts
    if (opts.remove_inserts) {
        aln = remove_inserts(aln);
    }

    // 5. Gap character replacement
    if (opts.gap_char != "default") {
        for (auto& seq : aln.sequences) {
            if (opts.gap_char.empty()) {
                // Remove all gaps
                seq.erase(std::remove(seq.begin(), seq.end(), '.'), seq.end());
                seq.erase(std::remove(seq.begin(), seq.end(), '-'), seq.end());
            } else {
                // Replace gaps with specified character
                std::replace(seq.begin(), seq.end(), '.', opts.gap_char[0]);
                std::replace(seq.begin(), seq.end(), '-', opts.gap_char[0]);
            }
        }
    }

    // 6. Format-specific gap handling
    if (target_format == AlignmentFormat::FASTA ||
        target_format == AlignmentFormat::CLUSTAL ||
        target_format == AlignmentFormat::STOCKHOLM ||
        target_format == AlignmentFormat::PSI) {
        // Convert all '.' to '-'
        for (auto& seq : aln.sequences) {
            std::replace(seq.begin(), seq.end(), '.', '-');
        }
    }
    // Note: A3M format will remove '.' in the writer itself

    // 7. Case conversion
    if (opts.uppercase) {
        aln = to_uppercase(aln);
    } else if (opts.lowercase) {
        aln = to_lowercase(aln);
    }

    // 8. Remove secondary structure
    if (opts.remove_secondary_structure) {
        aln = remove_secondary_structure(aln);
    }

    return aln;
}

// ============================================================================
// Alignment statistics
// ============================================================================

AlignmentStats compute_alignment_stats(
    const std::string& seq1,
    const std::string& seq2
) {
    AlignmentStats stats = {0.0, 0.0, 0.0, 0, 0.0, 0};

    if (seq1.length() != seq2.length()) {
        throw std::runtime_error("Sequences must have same length for statistics");
    }

    int matches = 0;
    int total_aligned = 0;
    int gaps = 0;

    for (size_t i = 0; i < seq1.length(); ++i) {
        char c1 = seq1[i];
        char c2 = seq2[i];

        if (c1 == '-' || c2 == '-' || c1 == '.' || c2 == '.') {
            gaps++;
        } else {
            total_aligned++;
            if (std::toupper(c1) == std::toupper(c2)) {
                matches++;
            }
        }
    }

    int length = seq1.length();
    stats.length = length;
    stats.gaps = gaps;
    stats.gap_percentage = (double)gaps / length;
    stats.identity = total_aligned > 0 ? (double)matches / total_aligned : 0.0;
    stats.similarity = stats.identity;  // For now, same as identity
    stats.coverage = (double)total_aligned / length;

    return stats;
}

}  // namespace io
}  // namespace pfalign
