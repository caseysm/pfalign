/**
 * Alignment format conversion implementation.
 */

#include "alignment_formats.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <cmath>

namespace pfalign {
namespace io {

// ============================================================================
// MultipleAlignment methods
// ============================================================================

bool MultipleAlignment::validate_lengths() const {
    if (sequences.empty()) return true;

    size_t len = sequences[0].length();
    for (size_t i = 1; i < sequences.size(); ++i) {
        if (sequences[i].length() != len) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Format utilities
// ============================================================================

AlignmentFormat parse_format(const std::string& format_str) {
    std::string fmt = format_str;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);

    if (fmt == "fas" || fmt == "fasta" || fmt == "fa" || fmt == "afa" || fmt == "afas" || fmt == "afasta") {
        return AlignmentFormat::FASTA;
    } else if (fmt == "a2m") {
        return AlignmentFormat::A2M;
    } else if (fmt == "a3m") {
        return AlignmentFormat::A3M;
    } else if (fmt == "sto" || fmt == "stockholm") {
        return AlignmentFormat::STOCKHOLM;
    } else if (fmt == "psi") {
        return AlignmentFormat::PSI;
    } else if (fmt == "clu" || fmt == "aln" || fmt == "clustal") {
        return AlignmentFormat::CLUSTAL;
    } else {
        throw std::runtime_error("Unknown alignment format: " + format_str);
    }
}

std::string format_to_string(AlignmentFormat format) {
    switch (format) {
        case AlignmentFormat::FASTA: return "fasta";
        case AlignmentFormat::A2M: return "a2m";
        case AlignmentFormat::A3M: return "a3m";
        case AlignmentFormat::STOCKHOLM: return "stockholm";
        case AlignmentFormat::PSI: return "psi";
        case AlignmentFormat::CLUSTAL: return "clustal";
        default: return "unknown";
    }
}

// ============================================================================
// AlignmentFormatParser
// ============================================================================

AlignmentFormat AlignmentFormatParser::detect_format(const std::string& path) {
    // Find file extension
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos) {
        return AlignmentFormat::FASTA;  // Default
    }

    std::string ext = path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    try {
        return parse_format(ext);
    } catch (...) {
        return AlignmentFormat::FASTA;  // Default
    }
}

MultipleAlignment AlignmentFormatParser::parse_file(
    const std::string& path,
    std::optional<AlignmentFormat> format
) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    AlignmentFormat fmt = format.value_or(detect_format(path));
    return parse_stream(in, fmt);
}

MultipleAlignment AlignmentFormatParser::parse_stream(
    std::istream& in,
    AlignmentFormat format
) {
    switch (format) {
        case AlignmentFormat::FASTA:
            return parse_fasta(in);
        case AlignmentFormat::A2M:
            return parse_a2m(in);
        case AlignmentFormat::A3M:
            return parse_a3m(in);
        case AlignmentFormat::STOCKHOLM:
            return parse_stockholm(in);
        case AlignmentFormat::PSI:
            return parse_psi(in);
        case AlignmentFormat::CLUSTAL:
            return parse_clustal(in);
        default:
            throw std::runtime_error("Unsupported format");
    }
}

void AlignmentFormatParser::clean_sequence(std::string& seq) {
    // Remove whitespace and non-alphanumeric characters except .- and ~
    seq.erase(
        std::remove_if(seq.begin(), seq.end(), [](char c) {
            return !std::isalnum(c) && c != '.' && c != '-' && c != '~';
        }),
        seq.end()
    );

    // Replace ~ with -
    std::replace(seq.begin(), seq.end(), '~', '-');
}

MultipleAlignment AlignmentFormatParser::parse_fasta(std::istream& in) {
    MultipleAlignment aln;
    std::string line;
    std::string current_name;
    std::string current_seq;
    bool first_seq = true;

    while (std::getline(in, line)) {
        // Remove carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip empty lines at beginning
        if (line.empty() && first_seq && current_name.empty()) {
            continue;
        }

        // Comment line (save as title if first)
        if (line[0] == '#') {
            if (aln.title.empty() && first_seq) {
                aln.title = line;
            }
            continue;
        }

        // Name line
        if (line[0] == '>') {
            // Save previous sequence if exists
            if (!current_name.empty()) {
                clean_sequence(current_seq);
                aln.names.push_back(current_name);
                aln.sequences.push_back(current_seq);
                first_seq = false;
            }

            // Parse new name (everything after '>')
            current_name = line.substr(1);
            current_seq.clear();

            // Check for special sequences
            if (current_name.find("ss_dssp") == 0 && !aln.ss_dssp.has_value()) {
                // Will be stored as ss_dssp later
            } else if (current_name.find("ss_pred") == 0 && !aln.ss_pred.has_value()) {
                // Will be stored as ss_pred later
            } else if (current_name.find("ss_conf") == 0 && !aln.ss_conf.has_value()) {
                // Will be stored as ss_conf later
            } else if (current_name.find("sa_dssp") == 0 && !aln.sa_dssp.has_value()) {
                // Will be stored as sa_dssp later
            }
        } else {
            // Sequence line
            current_seq += line;
        }
    }

    // Save last sequence
    if (!current_name.empty()) {
        clean_sequence(current_seq);

        // Check if this is a secondary structure sequence
        if (current_name.find("ss_dssp") == 0) {
            aln.ss_dssp = current_seq;
        } else if (current_name.find("ss_pred") == 0) {
            aln.ss_pred = current_seq;
        } else if (current_name.find("ss_conf") == 0) {
            aln.ss_conf = current_seq;
        } else if (current_name.find("sa_dssp") == 0) {
            aln.sa_dssp = current_seq;
        } else {
            aln.names.push_back(current_name);
            aln.sequences.push_back(current_seq);
        }
    }

    if (aln.num_sequences() == 0) {
        throw std::runtime_error("No sequences found in FASTA file");
    }

    return aln;
}

MultipleAlignment AlignmentFormatParser::parse_a2m(std::istream& in) {
    // A2M is same as FASTA, just preserves case
    return parse_fasta(in);
}

MultipleAlignment AlignmentFormatParser::parse_a3m(std::istream& in) {
    // A3M is same as FASTA/A2M, just may have variable-length sequences
    return parse_fasta(in);
}

MultipleAlignment AlignmentFormatParser::parse_stockholm(std::istream& in) {
    MultipleAlignment aln;
    std::string line;
    std::unordered_map<std::string, size_t> name_to_index;
    bool first_block = true;

    while (std::getline(in, line)) {
        // Remove carriage return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip STOCKHOLM header
        if (line.find("# STOCKHOLM") == 0) continue;

        // Skip comment lines
        if (line.empty() || line[0] == '#') {
            first_block = false;
            continue;
        }

        // End marker
        if (line.find("//") == 0) break;

        // Parse sequence line: NAME SEQUENCE
        std::istringstream iss(line);
        std::string name, seq;
        if (!(iss >> name >> seq)) continue;

        // Skip secondary structure annotations (will be handled specially)
        if (name.find("ss_") == 0 || name.find("sa_") == 0) {
            if (name == "ss_dssp" && !aln.ss_dssp.has_value()) {
                aln.ss_dssp = seq;
            } else if (name == "ss_pred" && !aln.ss_pred.has_value()) {
                aln.ss_pred = seq;
            } else if (name == "sa_dssp" && !aln.sa_dssp.has_value()) {
                aln.sa_dssp = seq;
            }
            continue;
        }

        // Add or append to sequence
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) {
            // New sequence
            if (!first_block && !aln.sequences.empty()) {
                throw std::runtime_error("New sequence in non-first block: " + name);
            }
            name_to_index[name] = aln.num_sequences();
            aln.names.push_back(name);
            aln.sequences.push_back(seq);
            first_block = true;
        } else {
            // Append to existing sequence
            if (first_block) {
                throw std::runtime_error("Duplicate sequence in first block: " + name);
            }
            aln.sequences[it->second] += seq;
        }
    }

    if (aln.num_sequences() == 0) {
        throw std::runtime_error("No sequences found in Stockholm file");
    }

    // Clean sequences
    for (auto& seq : aln.sequences) {
        clean_sequence(seq);
    }

    return aln;
}

MultipleAlignment AlignmentFormatParser::parse_psi(std::istream& in) {
    // PSI format is like Stockholm but without header/footer
    MultipleAlignment aln;
    std::string line;
    std::vector<std::string> current_names;
    bool first_block = true;

    while (std::getline(in, line)) {
        // Remove carriage return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Empty line marks new block
        if (line.empty()) {
            first_block = false;
            continue;
        }

        // Parse sequence line: NAME SEQUENCE
        std::istringstream iss(line);
        std::string name, seq;
        if (!(iss >> name >> seq)) continue;

        // Skip secondary structure
        if (name.find("ss_") == 0 || name.find("sa_") == 0 || name.find("aa_") == 0) {
            continue;
        }

        if (first_block) {
            aln.names.push_back(name);
            aln.sequences.push_back(seq);
        } else {
            // Append to existing sequences
            size_t idx = aln.sequences.size() - current_names.size() +
                         (std::find(current_names.begin(), current_names.end(), name) - current_names.begin());
            if (idx < aln.sequences.size()) {
                aln.sequences[idx] += seq;
            }
        }
    }

    if (aln.num_sequences() == 0) {
        throw std::runtime_error("No sequences found in PSI file");
    }

    // Clean sequences
    for (auto& seq : aln.sequences) {
        clean_sequence(seq);
    }

    return aln;
}

MultipleAlignment AlignmentFormatParser::parse_clustal(std::istream& in) {
    MultipleAlignment aln;
    std::string line;
    bool in_alignment = false;
    int block = 1;
    std::vector<std::string> block_names;

    while (std::getline(in, line)) {
        // Remove carriage return
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip CLUSTAL header
        if (line.find("CLUSTAL") != std::string::npos) {
            in_alignment = true;
            continue;
        }

        // Skip empty lines and conservation lines
        if (line.empty() || line.find_first_not_of(" .*:") == std::string::npos) {
            if (in_alignment && !block_names.empty()) {
                block++;
                block_names.clear();
            }
            continue;
        }

        if (!in_alignment) continue;

        // Parse sequence line: NAME SEQUENCE [optional number]
        std::istringstream iss(line);
        std::string name, seq;
        if (!(iss >> name >> seq)) continue;

        // Skip secondary structure
        if (name.find("ss_") == 0 || name.find("sa_") == 0 || name.find("aa_") == 0) {
            continue;
        }

        if (block == 1) {
            aln.names.push_back(name);
            aln.sequences.push_back(seq);
            block_names.push_back(name);
        } else {
            // Find matching sequence and append
            for (size_t i = 0; i < aln.names.size(); ++i) {
                if (aln.names[i] == name) {
                    aln.sequences[i] += seq;
                    block_names.push_back(name);
                    break;
                }
            }
        }
    }

    if (aln.num_sequences() == 0) {
        throw std::runtime_error("No sequences found in Clustal file");
    }

    // Clean sequences
    for (auto& seq : aln.sequences) {
        clean_sequence(seq);
    }

    return aln;
}

// ============================================================================
// AlignmentFormatWriter
// ============================================================================

void AlignmentFormatWriter::write_file(
    const MultipleAlignment& aln,
    const std::string& path,
    std::optional<AlignmentFormat> format,
    const Options& opts
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write to file: " + path);
    }

    AlignmentFormat fmt = format.value_or(AlignmentFormatParser::detect_format(path));
    write_stream(aln, out, fmt, opts);
}

void AlignmentFormatWriter::write_stream(
    const MultipleAlignment& aln,
    std::ostream& out,
    AlignmentFormat format,
    const Options& opts
) {
    // Apply transformations
    MultipleAlignment transformed = AlignmentTransform::apply_options(aln, opts, format);

    // Write in specified format
    switch (format) {
        case AlignmentFormat::FASTA:
            write_fasta(out, transformed, opts);
            break;
        case AlignmentFormat::A2M:
            write_a2m(out, transformed, opts);
            break;
        case AlignmentFormat::A3M:
            write_a3m(out, transformed, opts);
            break;
        case AlignmentFormat::STOCKHOLM:
            write_stockholm(out, transformed, opts);
            break;
        case AlignmentFormat::PSI:
            write_psi(out, transformed, opts);
            break;
        case AlignmentFormat::CLUSTAL:
            write_clustal(out, transformed, opts);
            break;
    }
}

void AlignmentFormatWriter::write_fasta(
    std::ostream& out,
    const MultipleAlignment& aln,
    const Options& opts
) {
    // Write title if present
    if (!aln.title.empty() && !opts.remove_secondary_structure) {
        out << aln.title << "\n";
    }

    // Write sequences
    for (size_t i = 0; i < aln.num_sequences(); ++i) {
        // Truncate name if needed
        std::string name = aln.names[i].substr(0, opts.max_name_length);

        out << ">" << name << "\n";

        // Write sequence with line breaks
        std::string seq = aln.sequences[i];
        for (size_t pos = 0; pos < seq.length(); pos += opts.residues_per_line) {
            out << seq.substr(pos, opts.residues_per_line) << "\n";
        }
    }
}

void AlignmentFormatWriter::write_a2m(
    std::ostream& out,
    const MultipleAlignment& aln,
    const Options& opts
) {
    // A2M is FASTA with match/insert distinction preserved
    write_fasta(out, aln, opts);
}

void AlignmentFormatWriter::write_a3m(
    std::ostream& out,
    const MultipleAlignment& aln,
    const Options& opts
) {
    // A3M is A2M with insert-gaps (.) removed
    MultipleAlignment compressed = aln;

    // Remove '.' characters from all sequences
    for (auto& seq : compressed.sequences) {
        seq.erase(std::remove(seq.begin(), seq.end(), '.'), seq.end());
    }

    write_fasta(out, compressed, opts);
}

void AlignmentFormatWriter::write_stockholm(
    std::ostream& out,
    const MultipleAlignment& aln,
    const Options& opts
) {
    out << "# STOCKHOLM 1.0\n\n";

    // Write sequences (one line per sequence)
    for (size_t i = 0; i < aln.num_sequences(); ++i) {
        std::string name = aln.names[i].substr(0, opts.max_name_length);

        // Left-align name in field
        out << name;
        for (size_t j = name.length(); j < static_cast<size_t>(opts.name_field_width); ++j) {
            out << " ";
        }
        out << " " << aln.sequences[i] << "\n";
    }

    out << "//\n";
}

void AlignmentFormatWriter::write_psi(
    std::ostream& out,
    const MultipleAlignment& aln,
    const Options& opts
) {
    // PSI format is like Stockholm without header/footer
    for (size_t i = 0; i < aln.num_sequences(); ++i) {
        std::string name = aln.names[i].substr(0, opts.max_name_length);

        // Left-align name in field
        out << name;
        for (size_t j = name.length(); j < static_cast<size_t>(opts.name_field_width); ++j) {
            out << " ";
        }
        out << " " << aln.sequences[i] << "\n";
    }
}

void AlignmentFormatWriter::write_clustal(
    std::ostream& out,
    const MultipleAlignment& aln,
    const Options& opts
) {
    out << "CLUSTAL\n\n\n";

    // Write sequences in blocks
    size_t seq_len = aln.alignment_length();
    for (size_t pos = 0; pos < seq_len; pos += opts.residues_per_line) {
        for (size_t i = 0; i < aln.num_sequences(); ++i) {
            std::string name = aln.names[i].substr(0, std::min(size_t(opts.name_field_width), aln.names[i].length()));

            // Left-align name
            out << name;
            for (size_t j = name.length(); j < static_cast<size_t>(opts.name_field_width); ++j) {
                out << " ";
            }

            // Write sequence chunk
            size_t chunk_size = std::min(size_t(opts.residues_per_line), seq_len - pos);
            out << " " << aln.sequences[i].substr(pos, chunk_size) << "\n";
        }
        out << "\n";
    }
}

}  // namespace io
}  // namespace pfalign
