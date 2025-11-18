/**
 * PDB Parser - Minimal C++17 parser for protein backbone atoms
 *
 * Extracts backbone atoms (N, CA, C, O) from PDB files.
 * Based on kad-ecoli/PDBParser but simplified for our use case.
 *
 * Usage:
 *   PDBParser parser;
 *   auto protein = parser.parse_file("1crn.pdb");
 *   auto coords = protein.get_backbone_coords(0);  // chain 0
 */

#pragma once

#include "protein_structure.h"
#include <fstream>
#include <string>
#include <string_view>

namespace pfalign {
namespace io {

/**
 * PDB file parser.
 */
class PDBParser {
public:
    /**
     * Parse PDB file and return Protein structure.
     *
     * @param filename Path to PDB file
     * @param backbone_only If true, only keep N, CA, C, O atoms
     * @return Parsed protein structure
     */
    Protein parse_file(const std::string& filename, bool backbone_only = true) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open PDB file: " + filename);
        }

        Protein protein;
        std::string line;

        // Current parsing state
        char current_chain = '\0';
        int current_resi = -9999;
        char current_icode = ' ';
        std::string current_resn;

        int chain_idx = -1;
        int residue_idx = -1;

        while (std::getline(file, line)) {
            // Skip short lines
            if (line.size() < 54)
                continue;

            // Only process ATOM records (standard amino acids)
            // Skip HETATM (waters, ligands, ions, etc.)
            std::string_view record = std::string_view(line).substr(0, 6);
            if (record != "ATOM  ")
                continue;

            // Skip alternate locations (keep only A or blank)
            char altLoc = line[16];
            if (altLoc != ' ' && altLoc != 'A')
                continue;

            // Parse atom info
            std::string atom_name = trim(line.substr(12, 4));
            std::string resn = trim(line.substr(17, 3));
            char chain_id = line[21];
            int resi = std::stoi(trim(line.substr(22, 4)));
            char icode = line[26];

            float x = std::stof(line.substr(30, 8));
            float y = std::stof(line.substr(38, 8));
            float z = std::stof(line.substr(46, 8));

            // Filter backbone atoms if requested
            if (backbone_only) {
                if (atom_name != "N" && atom_name != "CA" && atom_name != "C" && atom_name != "O") {
                    continue;
                }
            }

            // New chain?
            if (chain_id != current_chain) {
                protein.chains.emplace_back(chain_id);
                chain_idx++;
                current_chain = chain_id;
                current_resi = -9999;  // Reset residue tracking
                residue_idx = -1;
            }

            // New residue?
            if (resi != current_resi || icode != current_icode) {
                protein.chains[chain_idx].residues.emplace_back(resi, icode, resn);
                residue_idx++;
                current_resi = resi;
                current_icode = icode;
                current_resn = resn;
            }

            // Add atom to current residue
            protein.chains[chain_idx].residues[residue_idx].atoms.emplace_back(atom_name, x, y, z);
        }

        return protein;
    }

private:
    // Helper to trim whitespace
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }
};

}  // namespace io
}  // namespace pfalign
