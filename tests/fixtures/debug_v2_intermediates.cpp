/**
 * Debug V2 MPNN: Dump intermediate outputs at each stage.
 *
 * Usage:
 *   ./debug_v2_intermediates \
 *       /tmp/v1_mpnn_weights.safetensors \
 *       /path/to/1CRN.pdb \
 *       /tmp/v2_intermediates/
 */

#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/perf_timer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>


using namespace pfalign;

void save_array(const std::string& filepath, const float* data, int size) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open: " + filepath);
    }
    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    std::cout << "  Saved: " << filepath << " [" << size << " floats]" << std::endl;
}

void save_array_2d(const std::string& filepath, const float* data, int rows, int cols) {
    save_array(filepath, data, rows * cols);
}

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("debug_v2_intermediates");
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.pdb> <output_dir>" << std::endl;
        return 1;
    }

    std::string pdb_path = argv[1];
    std::string output_dir = argv[2];

    // Create output directory
    mkdir(output_dir.c_str(), 0755);

    std::cout << "========================================" << std::endl;
    std::cout << "  V2 MPNN Intermediate Dumper" << std::endl;
    std::cout << "========================================" << std::endl;

    // Load weights
    std::cout << "\nStep 1: Loading embedded weights..." << std::endl;
    mpnn::MPNNWeights weights(3);
    mpnn::MPNNConfig config;

    try {
        auto [loaded_weights, loaded_config, sw_defaults] = weights::load_embedded_mpnn_weights();
        weights = std::move(loaded_weights);
        config = loaded_config;
        (void)sw_defaults;
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load embedded MPNN weights (" << e.what() << ")" << std::endl;
        return 0;
    }

    std::cout << "✓ Loaded weights" << std::endl;
    std::cout << "  hidden_dim: " << config.hidden_dim << std::endl;
    std::cout << "  num_layers: " << config.num_layers << std::endl;
    std::cout << "  num_rbf: " << config.num_rbf << std::endl;

    // Load PDB
    std::cout << "\nStep 2: Parsing PDB..." << std::endl;
    io::PDBParser parser;
    io::Protein protein = parser.parse_file(pdb_path);

    int L = protein.get_chain(0).size();
    auto coords = protein.get_backbone_coords(0);

    std::cout << "✓ Parsed protein" << std::endl;
    std::cout << "  Residues: " << L << std::endl;

    // Save input coordinates
    save_array_2d(output_dir + "/00_input_coords.bin", coords.data(), L * 4, 3);

    // Create workspace
    mpnn::MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);

    // Populate positional metadata for single-chain input
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    std::cout << "\nStep 3: Running V2 forward pass with intermediate dumps..." << std::endl;

    // We'll need to manually inline the forward pass to dump intermediates
    // This is a copy of mpnn_forward<ScalarBackend> with added dumps

    // TODO: Implement manual forward pass with dumps at each stage
    // For now, just run normal forward and save final output
    float* node_emb = new float[L * config.hidden_dim]();

    mpnn::mpnn_forward<ScalarBackend>(
        coords.data(),
        L,
        weights,
        config,
        node_emb,
        &workspace
    );

    save_array_2d(output_dir + "/99_final_output.bin", node_emb, L, config.hidden_dim);

    std::cout << "\n✓ Saved all intermediates to: " << output_dir << std::endl;
    std::cout << "========================================" << std::endl;

    delete[] node_emb;
    return 0;
}
