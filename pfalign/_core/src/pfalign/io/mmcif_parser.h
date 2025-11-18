/**
 * mmCIF Parser - Minimal C++17 parser for mmCIF format
 *
 * Extracts backbone atoms (N, CA, C, O) from mmCIF files.
 * Based on gemmi but simplified for our specific use case.
 *
 * mmCIF (Macromolecular Crystallographic Information File) is the modern
 * PDB format used by RCSB PDB for structure files.
 *
 * Usage:
 *   mmCIFParser parser;
 *   auto protein = parser.parse_file("1crn.cif");
 *   auto coords = protein.get_backbone_coords(0);  // chain 0
 */

#pragma once

#include "protein_structure.h"
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cctype>

namespace pfalign {
namespace io {

/**
 * mmCIF file parser.
 *
 * Parses the _atom_site loop to extract backbone coordinates.
 */
class mmCIFParser {
public:
    /**
     * Parse mmCIF file and return Protein structure.
     *
     * @param filename Path to mmCIF file
     * @param backbone_only If true, only keep N, CA, C, O atoms
     * @return Parsed protein structure
     */
    Protein parse_file(const std::string& filename, bool backbone_only = true) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open mmCIF file: " + filename);
        }

        Protein protein;
        std::string line;
        bool in_atom_site_loop = false;
        std::vector<std::string> column_names;
        std::unordered_map<std::string, int> column_map;

        // Required columns
        int col_group_PDB = -1;
        int col_atom_id = -1;
        int col_comp_id = -1;
        int col_asym_id = -1;
        int col_seq_id = -1;
        int col_x = -1;
        int col_y = -1;
        int col_z = -1;
        int col_ins_code = -1;  // pdbx_PDB_ins_code (optional)
        int col_auth_seq_id = -1;

        // Parsing state
        char current_chain = '\0';
        int current_resi = -9999;
        char current_icode = ' ';
        int chain_idx = -1;
        int residue_idx = -1;
        std::unordered_map<char, int> chain_seq_fallback;

        while (std::getline(file, line)) {
            // Trim whitespace
            line = trim(line);
            if (line.empty() || line[0] == '#')
                continue;

            // Check for _atom_site loop
            if (line == "loop_") {
                in_atom_site_loop = false;
                column_names.clear();
                column_map.clear();
                continue;
            }

            // Collect column names in loop
            if (line.find("_atom_site.") == 0) {
                std::string col_name = line.substr(11);  // Remove "_atom_site."
                column_names.push_back(col_name);
                column_map[col_name] = column_names.size() - 1;

                // Map important columns
                if (col_name == "group_PDB")
                    col_group_PDB = column_names.size() - 1;
                else if (col_name == "label_atom_id")
                    col_atom_id = column_names.size() - 1;
                else if (col_name == "label_comp_id")
                    col_comp_id = column_names.size() - 1;
                else if (col_name == "label_asym_id")
                    col_asym_id = column_names.size() - 1;
                else if (col_name == "label_seq_id")
                    col_seq_id = column_names.size() - 1;
                else if (col_name == "Cartn_x")
                    col_x = column_names.size() - 1;
                else if (col_name == "Cartn_y")
                    col_y = column_names.size() - 1;
                else if (col_name == "Cartn_z")
                    col_z = column_names.size() - 1;
                else if (col_name == "pdbx_PDB_ins_code")
                    col_ins_code = column_names.size() - 1;
                else if (col_name == "auth_seq_id")
                    col_auth_seq_id = column_names.size() - 1;

                in_atom_site_loop = true;
                continue;
            }

            // Parse atom site data rows
            if (in_atom_site_loop && !column_names.empty()) {
                // Check if this line starts a new loop or data block
                if (line[0] == '_' || line == "loop_" || line.find("data_") == 0) {
                    in_atom_site_loop = false;
                    continue;
                }

                // Split line into tokens
                auto tokens = tokenize_mmcif_line(line);
                if (tokens.size() != column_names.size()) {
                    continue;  // Skip malformed lines
                }

                // Check required columns exist
                if (col_atom_id < 0 || col_comp_id < 0 || col_asym_id < 0 || col_seq_id < 0 ||
                    col_x < 0 || col_y < 0 || col_z < 0) {
                    continue;
                }

                // Extract data
                std::string group_PDB = (col_group_PDB >= 0) ? tokens[col_group_PDB] : "ATOM";
                std::string atom_name = tokens[col_atom_id];
                std::string resn = tokens[col_comp_id];
                char chain_id = tokens[col_asym_id][0];

                // Parse seq_id (might be '?' or '.')
                std::string seq_token = tokens[col_seq_id];
                if ((seq_token == "?" || seq_token == ".") && col_auth_seq_id >= 0) {
                    seq_token = tokens[col_auth_seq_id];
                }

                int resi = 0;
                bool have_resi = false;
                if (!(seq_token == "?" || seq_token == ".")) {
                    try {
                        resi = std::stoi(seq_token);
                        have_resi = true;
                    } catch (...) {
                        have_resi = false;
                    }
                }
                if (!have_resi) {
                    int next = ++chain_seq_fallback[chain_id];
                    resi = next;
                }

                char icode = ' ';
                if (col_ins_code >= 0 && tokens[col_ins_code] != "?" &&
                    tokens[col_ins_code] != ".") {
                    icode = tokens[col_ins_code][0];
                }

                float x = std::stof(tokens[col_x]);
                float y = std::stof(tokens[col_y]);
                float z = std::stof(tokens[col_z]);

                // Filter ATOM records only (skip HETATM unless needed)
                if (group_PDB != "ATOM")
                    continue;

                // Filter backbone atoms if requested
                if (backbone_only) {
                    if (atom_name != "N" && atom_name != "CA" && atom_name != "C" &&
                        atom_name != "O") {
                        continue;
                    }
                }

                // New chain?
                if (chain_id != current_chain) {
                    protein.chains.emplace_back(chain_id);
                    chain_idx++;
                    current_chain = chain_id;
                    current_resi = -9999;
                    residue_idx = -1;
                }

                // New residue?
                if (resi != current_resi || icode != current_icode) {
                    protein.chains[chain_idx].residues.emplace_back(resi, icode, resn);
                    residue_idx++;
                    current_resi = resi;
                    current_icode = icode;
                }

                // Add atom to current residue
                protein.chains[chain_idx].residues[residue_idx].atoms.emplace_back(atom_name, x, y,
                                                                                   z);
            }
        }

        return protein;
    }

private:
    // Trim whitespace
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    // Tokenize mmCIF line (handles quoted strings and special values)
    static std::vector<std::string> tokenize_mmcif_line(const std::string& line) {
        std::vector<std::string> tokens;
        std::string current_token;
        bool in_quotes = false;
        bool in_single_quotes = false;

        for (size_t i = 0; i < line.size(); i++) {
            char c = line[i];

            if (in_quotes) {
                if (c == '"') {
                    in_quotes = false;
                    tokens.push_back(current_token);
                    current_token.clear();
                } else {
                    current_token += c;
                }
            } else if (in_single_quotes) {
                if (c == '\'') {
                    in_single_quotes = false;
                    tokens.push_back(current_token);
                    current_token.clear();
                } else {
                    current_token += c;
                }
            } else {
                if (c == '"') {
                    in_quotes = true;
                } else if (c == '\'') {
                    in_single_quotes = true;
                } else if (std::isspace(c)) {
                    if (!current_token.empty()) {
                        tokens.push_back(current_token);
                        current_token.clear();
                    }
                } else {
                    current_token += c;
                }
            }
        }

        // Add last token if any
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }

        return tokens;
    }
};

}  // namespace io
}  // namespace pfalign
