/**
 * Common protein structure definitions for PDB and mmCIF parsers.
 *
 * Defines the data structures used to represent parsed protein structures.
 */

#pragma once

#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <optional>
#include <stdexcept>

#include "amino_acids.h"

namespace pfalign {
namespace io {

/**
 * Single atom with name and coordinates.
 */
struct Atom {
    std::string name;             // Atom name (e.g., "CA", "N", "C", "O")
    std::array<float, 3> coords;  // [x, y, z]

    Atom(std::string_view atom_name, float x, float y, float z) : name(atom_name), coords{x, y, z} {
    }
};

/**
 * Single residue with residue info and atoms.
 */
struct Residue {
    int resi;                 // Residue sequence number
    char icode;               // Insertion code
    std::string resn;         // Residue name (3-letter code)
    std::vector<Atom> atoms;  // List of atoms

    Residue(int res_num, char ins_code, std::string_view res_name)
        : resi(res_num), icode(ins_code), resn(res_name) {
    }

    // Get specific atom by name
    std::optional<std::array<float, 3>> get_atom(std::string_view atom_name) const {
        for (const auto& atom : atoms) {
            if (atom.name == atom_name) {
                return atom.coords;
            }
        }
        return std::nullopt;
    }
};

/**
 * Single chain with chain ID and residues.
 */
struct Chain {
    char chain_id;                  // Chain identifier
    std::vector<Residue> residues;  // List of residues

    explicit Chain(char id) : chain_id(id) {
    }

    // Get number of residues
    size_t size() const {
        return residues.size();
    }
};

/**
 * Protein structure (single model).
 */
class Protein {
public:
    std::vector<Chain> chains;

    // Get chain by index
    const Chain& get_chain(size_t idx) const {
        if (idx >= chains.size()) {
            throw std::out_of_range("Chain index out of range");
        }
        return chains[idx];
    }

    /**
     * Get backbone coordinates for a specific chain.
     * Returns [L, 4, 3] array: L residues, 4 atoms (N, CA, C, O), 3 coords (x,y,z).
     *
     * Missing atoms (including CA) are left as 0.0. Python callers can filter
     * out residues with missing CA atoms by checking for all-zero coordinates.
     */
    std::vector<float> get_backbone_coords(size_t chain_idx) const {
        const auto& chain = get_chain(chain_idx);
        size_t L = chain.size();
        std::vector<float> coords(L * 4 * 3, 0.0f);

        for (size_t i = 0; i < L; i++) {
            const auto& residue = chain.residues[i];

            // Extract N, CA, C, O atoms
            const char* atom_names[] = {"N", "CA", "C", "O"};
            for (size_t atom_idx = 0; atom_idx < 4; atom_idx++) {
                auto atom_coords = residue.get_atom(atom_names[atom_idx]);
                if (atom_coords) {
                    for (size_t dim = 0; dim < 3; dim++) {
                        coords[(i * 4 + atom_idx) * 3 + dim] = (*atom_coords)[dim];
                    }
                }
                // If atom missing (including CA), coords remain 0.0
            }
        }

        return coords;
    }

    /**
     * Get amino acid sequence for a specific chain.
     * Returns one-letter amino acid codes.
     *
     * Unknown residues are mapped to 'X'.
     */
    std::string get_sequence(size_t chain_idx) const {
        const auto& chain = get_chain(chain_idx);
        std::string sequence;
        sequence.reserve(chain.size());

        for (const auto& residue : chain.residues) {
            sequence.push_back(pfalign::io::three_to_one(residue.resn));
        }

        return sequence;
    }
    // Get total number of chains
    size_t num_chains() const {
        return chains.size();
    }

    int find_chain_index(char chain_id) const {
        for (size_t idx = 0; idx < chains.size(); ++idx) {
            if (chains[idx].chain_id == chain_id) {
                return static_cast<int>(idx);
            }
        }
        return -1;
    }
};

}  // namespace io
}  // namespace pfalign
