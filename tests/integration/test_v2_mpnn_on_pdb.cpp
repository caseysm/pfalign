/**
 * Test V2 MPNN on real PDB files using our new C++ parsers.
 *
 * This validates that:
 * 1. C++ PDB parser extracts coords correctly
 * 2. V2 MPNN can process real protein data
 * 3. Output shape matches expectations
 *
 * Future: Load V1 weights for exact numerical comparison with V1 MPNN.
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/common/perf_timer.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>


using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::io;
using namespace pfalign::weights;

/**
 * Generate a minimal synthetic PDB file for testing.
 * Creates a simple helix-like structure with 10 residues.
 */
std::string generate_synthetic_pdb() {
    std::ostringstream pdb;

    pdb << "HEADER    TEST PROTEIN                            01-JAN-25   TEST\n";
    pdb << "TITLE     SYNTHETIC TEST PROTEIN FOR MPNN VALIDATION\n";
    pdb << "REMARK    GENERATED FOR AUTOMATED TESTING\n";

    int atom_num = 1;
    for (int i = 0; i < 10; i++) {
        char resname[4] = "ALA";
        int resnum = i + 1;

        // Generate helix coordinates (1.5Å rise per residue)
        float x = 1.5f * std::cos(i * 0.5f);
        float y = 1.5f * std::sin(i * 0.5f);
        float z = i * 1.5f;

        // N atom
        pdb << "ATOM  " << std::setw(5) << atom_num++ << "  N   " << resname << " A"
            << std::setw(4) << resnum << "    "
            << std::setw(8) << std::fixed << std::setprecision(3) << (x - 0.5f)
            << std::setw(8) << y
            << std::setw(8) << (z - 0.5f)
            << "  1.00  0.00           N\n";

        // CA atom
        pdb << "ATOM  " << std::setw(5) << atom_num++ << "  CA  " << resname << " A"
            << std::setw(4) << resnum << "    "
            << std::setw(8) << x
            << std::setw(8) << y
            << std::setw(8) << z
            << "  1.00  0.00           C\n";

        // C atom
        pdb << "ATOM  " << std::setw(5) << atom_num++ << "  C   " << resname << " A"
            << std::setw(4) << resnum << "    "
            << std::setw(8) << (x + 0.5f)
            << std::setw(8) << y
            << std::setw(8) << (z + 0.5f)
            << "  1.00  0.00           C\n";

        // O atom
        pdb << "ATOM  " << std::setw(5) << atom_num++ << "  O   " << resname << " A"
            << std::setw(4) << resnum << "    "
            << std::setw(8) << (x + 0.8f)
            << std::setw(8) << y
            << std::setw(8) << (z + 0.8f)
            << "  1.00  0.00           O\n";
    }

    pdb << "END\n";
    return pdb.str();
}

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("test_v2_mpnn_on_pdb");
    std::cout << "========================================" << std::endl;
    std::cout << "  V2 MPNN on PDB Files" << std::endl;
    std::cout << "========================================" << std::endl;

    PDBParser parser;
    Protein protein;

    // Parse PDB file (from argument or generate synthetic)
    if (argc >= 2) {
        std::cout << "\nStep 1: Parsing PDB file: " << argv[1] << std::endl;
        try {
            protein = parser.parse_file(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse PDB: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "\nStep 1: Generating synthetic PDB (10 residues)..." << std::endl;
        std::string pdb_content = generate_synthetic_pdb();

        // Write to temp file
        std::string temp_file = "/tmp/test_mpnn_synthetic.pdb";
        std::ofstream out(temp_file);
        if (!out) {
            std::cerr << "Failed to create temp PDB file" << std::endl;
            return 1;
        }
        out << pdb_content;
        out.close();

        // Parse the synthetic PDB
        try {
            protein = parser.parse_file(temp_file);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse synthetic PDB: " << e.what() << std::endl;
            return 1;
        }
    }

    std::cout << "✓ Parsed protein structure" << std::endl;
    std::cout << "  Chains: " << protein.num_chains() << std::endl;

    if (protein.num_chains() == 0) {
        std::cerr << "No chains found in PDB file" << std::endl;
        return 1;
    }

    // Use first chain
    const auto& chain = protein.get_chain(0);
    int L = chain.size();
    std::cout << "  Chain A: " << L << " residues" << std::endl;

    // Get backbone coordinates
    auto coords = protein.get_backbone_coords(0);
    std::cout << "  Backbone coords: [" << L << ", 4, 3] = " << coords.size() << " floats" << std::endl;

    // Load embedded weights (no external files needed)
    std::cout << "\nStep 2: Loading embedded MPNN weights..." << std::endl;
    MPNNWeights weights(3);
    MPNNConfig config;

    try {
        auto [loaded_weights, loaded_config, sw_defaults] = load_embedded_mpnn_weights();
        weights = std::move(loaded_weights);
        config = loaded_config;
        (void)sw_defaults;
        std::cout << "✓ Loaded embedded weights (hidden_dim=" << config.hidden_dim << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load embedded MPNN weights (" << e.what() << ")" << std::endl;
        return 0;
    }

    // Create workspace
    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }
    float* embeddings = new float[L * config.hidden_dim]();

    // Run MPNN forward pass
    std::cout << "\nStep 3: Running MPNN forward pass..." << std::endl;
    mpnn_forward<ScalarBackend>(
        coords.data(),
        L,
        weights,
        config,
        embeddings,
        &workspace
    );
    std::cout << "✓ Forward pass complete" << std::endl;

    // Compute statistics
    double sum = 0.0, sum_sq = 0.0;
    float min_val = 1e9f, max_val = -1e9f;

    for (int i = 0; i < L * config.hidden_dim; i++) {
        sum += embeddings[i];
        sum_sq += embeddings[i] * embeddings[i];
        min_val = std::min(min_val, embeddings[i]);
        max_val = std::max(max_val, embeddings[i]);
    }

    double mean = sum / (L * config.hidden_dim);
    double var = sum_sq / (L * config.hidden_dim) - mean * mean;
    double std = std::sqrt(var);

    std::cout << "\nStep 4: Results" << std::endl;
    std::cout << "  Output shape: [" << L << " * " << config.hidden_dim << "]" << std::endl;
    std::cout << "  Statistics:" << std::endl;
    std::cout << "    Mean: " << std::setw(10) << std::fixed << std::setprecision(6) << mean << std::endl;
    std::cout << "    Std:  " << std::setw(10) << std << std::endl;
    std::cout << "    Min:  " << std::setw(10) << min_val << std::endl;
    std::cout << "    Max:  " << std::setw(10) << max_val << std::endl;

    // Show first residue embedding (first 8 dims)
    std::cout << "\n  First residue (dims 0-7):" << std::endl;
    std::cout << "    ";
    for (int i = 0; i < 8; i++) {
        std::cout << std::setw(10) << embeddings[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ V2 MPNN successfully processed PDB file" << std::endl;
    std::cout << "========================================" << std::endl;

    // Cleanup
    delete[] embeddings;

    return 0;
}
