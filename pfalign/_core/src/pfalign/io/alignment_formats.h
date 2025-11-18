/**
 * Alignment format conversion library.
 *
 * Supports reading/writing multiple sequence alignments in various formats:
 * - FASTA (fas): Standard aligned FASTA
 * - A2M: Match/insert states (uppercase=match, lowercase=insert, '.'=insert-gap)
 * - A3M: Compressed A2M (insert-gaps omitted, variable-length sequences)
 * - Stockholm (sto): HMMER format
 * - PSI-BLAST (psi): PSI-BLAST input format
 * - Clustal (clu): Clustal format
 *
 * Based on HHsuite's reformat.pl functionality.
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iosfwd>

namespace pfalign {
namespace io {

/**
 * Multiple sequence alignment representation.
 */
struct MultipleAlignment {
    std::vector<std::string> names;      // Sequence names/descriptions
    std::vector<std::string> sequences;  // Aligned sequences
    std::string title;                   // Optional title line (from #-comments)

    // Optional secondary structure annotations
    std::optional<std::string> ss_dssp;  // DSSP secondary structure
    std::optional<std::string> ss_pred;  // Predicted secondary structure
    std::optional<std::string> ss_conf;  // Confidence values
    std::optional<std::string> sa_dssp;  // Solvent accessibility

    size_t num_sequences() const { return names.size(); }
    size_t alignment_length() const { return sequences.empty() ? 0 : sequences[0].length(); }

    // Validate that all sequences have same length (except for A3M)
    bool validate_lengths() const;
};

/**
 * Supported alignment formats.
 */
enum class AlignmentFormat {
    FASTA,     // fas, fasta, fa, afa
    A2M,       // a2m
    A3M,       // a3m
    STOCKHOLM, // sto, stockholm
    PSI,       // psi
    CLUSTAL    // clu, aln
};

/**
 * Convert format string to enum.
 */
AlignmentFormat parse_format(const std::string& format_str);

/**
 * Get format name string.
 */
std::string format_to_string(AlignmentFormat format);

/**
 * Alignment format parser.
 */
class AlignmentFormatParser {
public:
    /**
     * Parse alignment from file.
     *
     * @param path File path
     * @param format Format (auto-detect from extension if not specified)
     * @return Parsed alignment
     */
    static MultipleAlignment parse_file(
        const std::string& path,
        std::optional<AlignmentFormat> format = std::nullopt
    );

    /**
     * Parse alignment from stream.
     *
     * @param in Input stream
     * @param format Format (must be specified)
     * @return Parsed alignment
     */
    static MultipleAlignment parse_stream(
        std::istream& in,
        AlignmentFormat format
    );

    /**
     * Auto-detect format from file extension.
     *
     * @param path File path
     * @return Detected format (defaults to FASTA if unknown)
     */
    static AlignmentFormat detect_format(const std::string& path);

private:
    static MultipleAlignment parse_fasta(std::istream& in);
    static MultipleAlignment parse_a2m(std::istream& in);
    static MultipleAlignment parse_a3m(std::istream& in);
    static MultipleAlignment parse_stockholm(std::istream& in);
    static MultipleAlignment parse_psi(std::istream& in);
    static MultipleAlignment parse_clustal(std::istream& in);

    // Helper: clean up sequence (remove non-alphanumeric except .- )
    static void clean_sequence(std::string& seq);
};

/**
 * Alignment format writer.
 */
class AlignmentFormatWriter {
public:
    /**
     * Writer options.
     */
    struct Options {
        int residues_per_line;           // Residues per line (FASTA, Clustal)
        int max_name_length;             // Maximum name length
        int name_field_width;            // Name field width (Stockholm, PSI, Clustal)

        // Match state assignment (for A2M/A3M output)
        enum class MatchMode {
            NONE,           // Don't change match states
            FIRST_SEQUENCE, // Use first sequence (-M first)
            GAP_THRESHOLD   // Use gap percentage threshold (-M <int>)
        };
        MatchMode match_mode;
        int gap_threshold_percent;       // For GAP_THRESHOLD mode

        // Gap removal options
        bool remove_inserts;             // Remove lowercase residues (-r)
        int remove_gapped_threshold;     // Remove columns with >X% gaps (-r <int>)

        // Case conversion
        bool uppercase;                  // Convert to uppercase (-uc)
        bool lowercase;                  // Convert to lowercase (-lc)

        // Gap character control
        std::string gap_char;            // "default", "", "-", etc.

        // Secondary structure
        bool remove_secondary_structure; // Remove ss_* and sa_* sequences (-noss)

        // Constructor with defaults
        Options()
            : residues_per_line(100)
            , max_name_length(1000)
            , name_field_width(32)
            , match_mode(MatchMode::NONE)
            , gap_threshold_percent(0)
            , remove_inserts(false)
            , remove_gapped_threshold(0)
            , uppercase(false)
            , lowercase(false)
            , gap_char("default")
            , remove_secondary_structure(false)
        {}
    };

    /**
     * Write alignment to file.
     *
     * @param aln Alignment to write
     * @param path Output file path
     * @param format Format (auto-detect from extension if not specified)
     * @param opts Writer options
     */
    static void write_file(
        const MultipleAlignment& aln,
        const std::string& path,
        std::optional<AlignmentFormat> format = std::nullopt,
        const Options& opts = Options()
    );

    /**
     * Write alignment to stream.
     *
     * @param aln Alignment to write
     * @param out Output stream
     * @param format Format (must be specified)
     * @param opts Writer options
     */
    static void write_stream(
        const MultipleAlignment& aln,
        std::ostream& out,
        AlignmentFormat format,
        const Options& opts = Options()
    );

private:
    // Format-specific writers
    static void write_fasta(std::ostream& out, const MultipleAlignment& aln, const Options& opts);
    static void write_a2m(std::ostream& out, const MultipleAlignment& aln, const Options& opts);
    static void write_a3m(std::ostream& out, const MultipleAlignment& aln, const Options& opts);
    static void write_stockholm(std::ostream& out, const MultipleAlignment& aln, const Options& opts);
    static void write_psi(std::ostream& out, const MultipleAlignment& aln, const Options& opts);
    static void write_clustal(std::ostream& out, const MultipleAlignment& aln, const Options& opts);
};

/**
 * Alignment transformation utilities.
 */
class AlignmentTransform {
public:
    /**
     * Fill A3M gaps to make all sequences same length.
     *
     * Converts A3M (variable-length) to A2M (fixed-length) by inserting '.' characters.
     */
    static MultipleAlignment fill_a3m_gaps(const MultipleAlignment& aln);

    /**
     * Assign match states using first sequence.
     *
     * Columns with residue in first sequence = uppercase (match)
     * Columns with gap in first sequence = lowercase (insert)
     */
    static MultipleAlignment assign_match_states_first(const MultipleAlignment& aln);

    /**
     * Assign match states using gap percentage rule.
     *
     * Columns with <threshold% gaps = uppercase (match)
     * Columns with >=threshold% gaps = lowercase (insert)
     */
    static MultipleAlignment assign_match_states_gap_rule(
        const MultipleAlignment& aln,
        int threshold_percent
    );

    /**
     * Remove columns with too many gaps.
     *
     * Removes entire columns where gap percentage >= threshold.
     * This is a destructive transformation that changes alignment structure.
     */
    static MultipleAlignment remove_gapped_columns(
        const MultipleAlignment& aln,
        int threshold_percent
    );

    /**
     * Remove insert states (lowercase residues).
     *
     * Deletes all lowercase characters from sequences.
     */
    static MultipleAlignment remove_inserts(const MultipleAlignment& aln);

    /**
     * Convert all residues to uppercase.
     */
    static MultipleAlignment to_uppercase(const MultipleAlignment& aln);

    /**
     * Convert all residues to lowercase.
     */
    static MultipleAlignment to_lowercase(const MultipleAlignment& aln);

    /**
     * Remove secondary structure sequences (ss_*, sa_*).
     */
    static MultipleAlignment remove_secondary_structure(const MultipleAlignment& aln);

    /**
     * Apply all transformations based on writer options.
     *
     * Order of operations:
     * 1. Fill A3M gaps (if needed for format)
     * 2. Match state assignment (if requested)
     * 3. Remove gapped columns (if requested)
     * 4. Remove inserts (if requested)
     * 5. Case conversion (if requested)
     * 6. Remove secondary structure (if requested)
     */
    static MultipleAlignment apply_options(
        MultipleAlignment aln,
        const AlignmentFormatWriter::Options& opts,
        AlignmentFormat target_format
    );
};

/**
 * Alignment statistics.
 */
struct AlignmentStats {
    double identity;        // Sequence identity (0-1)
    double similarity;      // Sequence similarity (0-1)
    double coverage;        // Alignment coverage (0-1)
    int gaps;               // Total number of gaps
    double gap_percentage;  // Gap percentage (0-1)
    int length;             // Total alignment length
};

/**
 * Compute alignment statistics for pairwise alignment.
 */
AlignmentStats compute_alignment_stats(
    const std::string& seq1,
    const std::string& seq2
);

}  // namespace io
}  // namespace pfalign
