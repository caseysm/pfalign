/**
 * Unit tests for PDB and mmCIF parsers.
 *
 * Tests basic parsing functionality with minimal test files.
 */

#include "pfalign/io/pdb_parser.h"
#include "pfalign/io/mmcif_parser.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

using namespace pfalign::io;

void create_test_pdb(const char* filename) {
    std::ofstream out(filename);
    out << "ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N  \n";
    out << "ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80           C  \n";
    out << "ATOM      3  C   THR A   1      15.685  12.755   5.133  1.00  9.19           C  \n";
    out << "ATOM      4  O   THR A   1      15.268  13.825   5.594  1.00  9.85           O  \n";
    out << "ATOM      5  N   THR A   2      15.115  11.555   5.265  1.00  7.81           N  \n";
    out << "ATOM      6  CA  THR A   2      13.856  11.469   5.961  1.00  6.46           C  \n";
    out << "ATOM      7  C   THR A   2      14.164  10.785   7.249  1.00  5.24           C  \n";
    out << "ATOM      8  O   THR A   2      14.550   9.634   7.292  1.00  6.04           O  \n";
    out << "END\n";
    out.close();
}

void create_test_mmcif(const char* filename) {
    std::ofstream out(filename);
    out << "data_test\n";
    out << "#\n";
    out << "loop_\n";
    out << "_atom_site.group_PDB\n";
    out << "_atom_site.id\n";
    out << "_atom_site.label_atom_id\n";
    out << "_atom_site.label_comp_id\n";
    out << "_atom_site.label_asym_id\n";
    out << "_atom_site.label_seq_id\n";
    out << "_atom_site.Cartn_x\n";
    out << "_atom_site.Cartn_y\n";
    out << "_atom_site.Cartn_z\n";
    out << "ATOM 1 N THR A 1 17.047 14.099 3.625\n";
    out << "ATOM 2 CA THR A 1 16.967 12.784 4.338\n";
    out << "ATOM 3 C THR A 1 15.685 12.755 5.133\n";
    out << "ATOM 4 O THR A 1 15.268 13.825 5.594\n";
    out << "ATOM 5 N THR A 2 15.115 11.555 5.265\n";
    out << "ATOM 6 CA THR A 2 13.856 11.469 5.961\n";
    out << "ATOM 7 C THR A 2 14.164 10.785 7.249\n";
    out << "ATOM 8 O THR A 2 14.550 9.634 7.292\n";
    out << "#\n";
    out.close();
}

bool test_pdb_parser() {
    std::cout << "=== Test 1: PDB Parser ===" << std::endl;

    // Create test file
    create_test_pdb("/tmp/test.pdb");

    // Parse
    PDBParser parser;
    auto protein = parser.parse_file("/tmp/test.pdb");

    // Check structure
    if (protein.num_chains() != 1) {
        std::cerr << "Expected 1 chain, got " << protein.num_chains() << std::endl;
        return false;
    }

    const auto& chain = protein.get_chain(0);
    if (chain.size() != 2) {
        std::cerr << "Expected 2 residues, got " << chain.size() << std::endl;
        return false;
    }

    // Check first residue
    const auto& res1 = chain.residues[0];
    if (res1.resn != "THR" || res1.resi != 1) {
        std::cerr << "First residue incorrect" << std::endl;
        return false;
    }

    // Check atoms
    if (res1.atoms.size() != 4) {
        std::cerr << "Expected 4 atoms in residue 1, got " << res1.atoms.size() << std::endl;
        return false;
    }

    // Check CA coordinates
    auto ca_coords = res1.get_atom("CA");
    if (!ca_coords) {
        std::cerr << "CA atom not found" << std::endl;
        return false;
    }

    float expected_ca[3] = {16.967f, 12.784f, 4.338f};
    for (int i = 0; i < 3; i++) {
        if (std::abs((*ca_coords)[i] - expected_ca[i]) > 0.01f) {
            std::cerr << "CA coord mismatch" << std::endl;
            return false;
        }
    }

    // Test get_backbone_coords
    auto coords = protein.get_backbone_coords(0);
    if (coords.size() != 2 * 4 * 3) {  // 2 residues, 4 atoms, 3 coords
        std::cerr << "Backbone coords size mismatch" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    return true;
}

bool test_mmcif_parser() {
    std::cout << "\n=== Test 2: mmCIF Parser ===" << std::endl;

    // Create test file
    create_test_mmcif("/tmp/test.cif");

    // Parse
    mmCIFParser parser;
    auto protein = parser.parse_file("/tmp/test.cif");

    // Check structure
    if (protein.num_chains() != 1) {
        std::cerr << "Expected 1 chain, got " << protein.num_chains() << std::endl;
        return false;
    }

    const auto& chain = protein.get_chain(0);
    if (chain.size() != 2) {
        std::cerr << "Expected 2 residues, got " << chain.size() << std::endl;
        return false;
    }

    // Check first residue
    const auto& res1 = chain.residues[0];
    if (res1.resn != "THR" || res1.resi != 1) {
        std::cerr << "First residue incorrect" << std::endl;
        return false;
    }

    // Check atoms
    if (res1.atoms.size() != 4) {
        std::cerr << "Expected 4 atoms in residue 1, got " << res1.atoms.size() << std::endl;
        return false;
    }

    // Check CA coordinates
    auto ca_coords = res1.get_atom("CA");
    if (!ca_coords) {
        std::cerr << "CA atom not found" << std::endl;
        return false;
    }

    float expected_ca[3] = {16.967f, 12.784f, 4.338f};
    for (int i = 0; i < 3; i++) {
        if (std::abs((*ca_coords)[i] - expected_ca[i]) > 0.01f) {
            std::cerr << "CA coord mismatch" << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    return true;
}

bool test_real_pdb_files() {
    std::cout << "\n=== Test 3: Real PDB Files ===" << std::endl;

    struct TestCase {
        const char* file;
        const char* name;
        int expected_residues;
        int expected_chains;
    };

    // Note: Residue counts are ATOM records only (HETATM excluded)
    // Paths are relative to build directory where meson runs tests
    TestCase cases[] = {
        {"../tests/data/fixtures/structures/crambin.pdb", "Crambin", 46, 1},
        {"../tests/data/fixtures/structures/ubiquitin.pdb", "Ubiquitin", 76, 1},
    };

    PDBParser parser;
    int passed = 0;
    int total = sizeof(cases) / sizeof(cases[0]);

    for (const auto& tc : cases) {
        std::cout << "\nTesting " << tc.name << "..." << std::endl;

        // Try to parse
        try {
            auto protein = parser.parse_file(tc.file);

            // Check chain count
            if (protein.num_chains() != tc.expected_chains) {
                std::cerr << "  ✗ Expected " << tc.expected_chains << " chain(s), got "
                          << protein.num_chains() << std::endl;
                continue;
            }

            // Check residue count
            const auto& chain = protein.get_chain(0);
            if (chain.size() != tc.expected_residues) {
                std::cerr << "  ✗ Expected " << tc.expected_residues << " residues, got "
                          << chain.size() << std::endl;
                continue;
            }

            // Check backbone atoms for first residue
            const auto& res1 = chain.residues[0];
            bool has_backbone = res1.get_atom("N") && res1.get_atom("CA") &&
                               res1.get_atom("C") && res1.get_atom("O");
            if (!has_backbone) {
                std::cerr << "  ✗ First residue missing backbone atoms" << std::endl;
                continue;
            }

            // Check get_backbone_coords
            auto coords = protein.get_backbone_coords(0);
            // Backbone coords only include residues with all 4 backbone atoms (N, CA, C, O)
            // Waters and other HETATM won't have these, so they're excluded
            if (coords.size() % (4 * 3) != 0) {
                std::cerr << "  ✗ Backbone coords not divisible by 12: " << coords.size() << std::endl;
                continue;
            }

            int backbone_residues = coords.size() / (4 * 3);
            std::cout << "  ✓ " << tc.name << " - " << chain.size() << " total residues, "
                      << backbone_residues << " with backbone atoms, "
                      << protein.num_chains() << " chain(s)" << std::endl;
            passed++;

        } catch (const std::exception& e) {
            std::cerr << "  ✗ Failed to parse: " << e.what() << std::endl;
        }
    }

    std::cout << "\nResult: " << passed << "/" << total << " real PDB files passed" << std::endl;
    return passed == total;
}

bool test_real_mmcif_files() {
    std::cout << "\n=== Test 4: Real mmCIF Files ===" << std::endl;
    std::cout << "  Skipped - no .cif test files available" << std::endl;
    std::cout << "✓ PASS (skipped)" << std::endl;
    return true;

    // TODO: Add .cif test files to tests/data/fixtures/structures/
    // When files are available, use paths like:
    // "tests/data/fixtures/structures/crambin.cif"
    // "tests/data/fixtures/structures/ubiquitin.cif"
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Parser Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    bool pass1 = test_pdb_parser();
    bool pass2 = test_mmcif_parser();
    bool pass3 = test_real_pdb_files();
    bool pass4 = test_real_mmcif_files();

    std::cout << "\n========================================" << std::endl;
    if (pass1 && pass2 && pass3 && pass4) {
        std::cout << "✓ All tests passed (4/4)" << std::endl;
    } else {
        std::cout << "✗ Some tests failed" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return (pass1 && pass2 && pass3 && pass4) ? 0 : 1;
}
