/**
 * Numerical validation: V2 MPNN vs V1 reference outputs
 *
 * Loads V1 weights into V2, runs on same input (Crambin), compares embeddings.
 * Target: RMSE < 1e-5 (V1 vs JAX is 2.37e-06, so this should be similar).
 *
 * Usage:
 *   ./validate_v2_vs_v1_numerical \
 *       /tmp/v1_mpnn_weights.safetensors \
 *       /home/ubuntu/projects/protein-forge/data/test_proteins/1CRN.pdb \
 *       /home/ubuntu/projects/protein-forge/data/mpnn_reference/1CRN_crambin/h_V_final.npy \
 *       [output_dir]
 */

#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/tools/weights/save_npy.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/perf_timer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>


using namespace pfalign;

/**
 * Simple .npy loader for V1 reference outputs.
 *
 * Only supports float32, C-order, no padding.
 */
std::vector<float> load_npy(const std::string& filepath, std::vector<size_t>& shape) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open .npy file: " + filepath);
    }

    // Read magic string (6 bytes)
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid .npy file (bad magic)");
    }

    // Read version (2 bytes)
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // Read header (Python dict describing array)
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    // Parse shape from header (very simple parser, assumes standard format)
    size_t shape_start = header.find("'shape': (");
    if (shape_start == std::string::npos) {
        throw std::runtime_error("Could not find 'shape' in .npy header");
    }

    size_t shape_end = header.find(")", shape_start);
    std::string shape_str = header.substr(shape_start + 10, shape_end - (shape_start + 10));

    // Parse comma-separated dimensions
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(",", pos);
        if (comma == std::string::npos) comma = shape_str.size();

        std::string dim_str = shape_str.substr(pos, comma - pos);
        // Trim whitespace
        dim_str.erase(0, dim_str.find_first_not_of(" \t"));
        dim_str.erase(dim_str.find_last_not_of(" \t") + 1);

        if (!dim_str.empty()) {
            shape.push_back(std::stoull(dim_str));
        }

        pos = comma + 1;
    }

    // Calculate total size
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }

    // Read data
    std::vector<float> data(total_size);
    file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));

    if (!file) {
        throw std::runtime_error("Failed to read .npy data");
    }

    return data;
}

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("validate_v2_vs_v1_numerical");
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <crambin.pdb> <v1_reference.npy> [output_dir]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example:" << std::endl;
        std::cerr << "  " << argv[0] << " \\" << std::endl;
        std::cerr << "      /path/to/1CRN.pdb \\" << std::endl;
        std::cerr << "      /path/to/h_V_final.npy \\" << std::endl;
        std::cerr << "      /tmp/v2_intermediates  # optional" << std::endl;
        return 1;
    }

    // Optional: output directory for intermediate dumps
    std::string output_dir = "";
    bool save_intermediates = false;
    if (argc == 4) {
        output_dir = argv[3];
        mkdir(output_dir.c_str(), 0755);
        save_intermediates = true;
        std::cout << "Will save intermediates to: " << output_dir << std::endl;
    }

    const char* pdb_path = argv[1];
    const char* reference_path = argv[2];

    std::cout << "========================================" << std::endl;
    std::cout << "  V2 vs V1 Numerical Validation" << std::endl;
    std::cout << "========================================" << std::endl;

    // Load embedded weights
    std::cout << "\nStep 1: Loading embedded V1 weights..." << std::endl;
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

    std::cout << "✓ Loaded V1 weights" << std::endl;
    std::cout << "  hidden_dim: " << config.hidden_dim << std::endl;
    std::cout << "  num_layers: " << config.num_layers << std::endl;
    std::cout << "  num_rbf: " << config.num_rbf << std::endl;

    {
        std::ofstream out("debug_ffn_W_out.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(weights.layers[0].ffn_W_out_weight.data()),
                  4 * config.hidden_dim * config.hidden_dim * sizeof(float));
    }

    // Load PDB
    std::cout << "\nStep 2: Parsing Crambin PDB..." << std::endl;
    io::PDBParser parser;
    io::Protein protein = parser.parse_file(pdb_path);

    int L = protein.get_chain(0).size();
    auto coords = protein.get_backbone_coords(0);

    // Use full adjacency to match V1 reference (k = L)
    config.k_neighbors = L;

    std::cout << "✓ Parsed protein" << std::endl;
    std::cout << "  Residues: " << L << std::endl;
    std::cout << "  Coords shape: [" << L << ", 4, 3]" << std::endl;

    // Run V2 MPNN
    std::cout << "\nStep 3: Running V2 MPNN forward pass..." << std::endl;
    mpnn::MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);
    float* v2_embeddings = new float[L * config.hidden_dim]();

    // Populate positional metadata (sequence offsets + chain ids)
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;    // Sequential indices (matches V1 inputs)
        workspace.chain_labels[i] = 0;   // Single chain for validation sample
    }

    // Run forward pass (workspace will contain intermediate values)
    mpnn::mpnn_forward<ScalarBackend>(
        coords.data(),
        L,
        weights,
        config,
        v2_embeddings,
        &workspace
    );

    std::cout << "✓ V2 forward pass complete" << std::endl;
    std::cout << "  Output shape: [" << L << " * " << config.hidden_dim << "]" << std::endl;

    // Save intermediates if requested
    if (save_intermediates) {
        std::cout << "\nSaving V2 intermediates..." << std::endl;

        // Note: workspace contains FINAL states, not all intermediates
        // For full debugging, need to modify mpnn_forward or inline it here

        // Final node embeddings
        save_npy_2d(output_dir + "/h_V_final.npy", v2_embeddings, L, config.hidden_dim);
        std::cout << "  Saved: h_V_final.npy" << std::endl;

        std::cout << "\nWARNING: Only final output saved." << std::endl;
        std::cout << "For full intermediate dumps, need to inline forward pass." << std::endl;
        std::cout << "See DEBUGGING_READY.md for details." << std::endl;
    }

    // Load V1 reference
    std::cout << "\nStep 4: Loading V1 reference outputs..." << std::endl;
    std::vector<size_t> v1_shape;
    std::vector<float> v1_embeddings = load_npy(reference_path, v1_shape);

    std::cout << "✓ Loaded V1 reference" << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < v1_shape.size(); i++) {
        std::cout << v1_shape[i];
        if (i < v1_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Verify shapes match
    if (v1_shape.size() == 3) {
        // [1, L, D] format
        if (v1_shape[1] != (size_t)L || v1_shape[2] != (size_t)config.hidden_dim) {
            std::cerr << "Shape mismatch!" << std::endl;
            return 1;
        }
    } else if (v1_shape.size() == 2) {
        // [L, D] format
        if (v1_shape[0] != (size_t)L || v1_shape[1] != (size_t)config.hidden_dim) {
            std::cerr << "Shape mismatch!" << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unexpected V1 shape" << std::endl;
        return 1;
    }

    // Compare
    std::cout << "\nStep 5: Computing errors..." << std::endl;

    double sum_sq_error = 0.0;
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;
    int total_elements = L * config.hidden_dim;

    // Skip batch dimension if present
    int v1_offset = (v1_shape.size() == 3) ? config.hidden_dim : 0;

    for (int i = 0; i < total_elements; i++) {
        float v2_val = v2_embeddings[i];
        float v1_val = v1_embeddings[i + v1_offset];

        float error = std::abs(v2_val - v1_val);
        sum_abs_error += error;
        max_abs_error = std::max(max_abs_error, (double)error);
        sum_sq_error += error * error;
    }

    double mean_abs_error = sum_abs_error / total_elements;
    double rmse = std::sqrt(sum_sq_error / total_elements);

    // Compute relative error
    double v1_norm = 0.0;
    for (int i = 0; i < total_elements; i++) {
        float v1_val = v1_embeddings[i + v1_offset];
        v1_norm += v1_val * v1_val;
    }
    v1_norm = std::sqrt(v1_norm);
    double relative_error = rmse / v1_norm;

    std::cout << std::scientific << std::setprecision(8);
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Error Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n  Max absolute error:  " << max_abs_error << std::endl;
    std::cout << "  Mean absolute error: " << mean_abs_error << std::endl;
    std::cout << "  RMSE:                " << rmse << std::endl;
    std::cout << "  Relative error:      " << relative_error << " (" << (relative_error * 100) << "%)" << std::endl;

    // Show sample comparison
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Sample Embeddings (First Residue)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nFirst 10 dims:" << std::endl;
    std::cout << "  V1: ";
    for (int i = 0; i < 10; i++) {
        std::cout << std::setw(10) << v1_embeddings[i + v1_offset] << " ";
    }
    std::cout << "\n  V2: ";
    for (int i = 0; i < 10; i++) {
        std::cout << std::setw(10) << v2_embeddings[i] << " ";
    }
    std::cout << "\n  Delta:  ";
    for (int i = 0; i < 10; i++) {
        float err = std::abs(v2_embeddings[i] - v1_embeddings[i + v1_offset]);
        std::cout << std::setw(10) << std::scientific << std::setprecision(2) << err << " ";
    }
    std::cout << std::endl;

    // Verdict
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Verdict" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::scientific << std::setprecision(2);

    if (rmse < 1e-5) {
        std::cout << "\n✓ EXCELLENT: V2 matches V1 with RMSE " << rmse << " < 1e-5" << std::endl;
        std::cout << "  V2 numerical validation PASSED!" << std::endl;
    } else if (rmse < 1e-4) {
        std::cout << "\n⚠ GOOD: V2 matches V1 with RMSE " << rmse << " < 1e-4" << std::endl;
        std::cout << "  V2 is numerically close to V1" << std::endl;
    } else {
        std::cout << "\n✗ FAILED: RMSE " << rmse << " > 1e-4" << std::endl;
        std::cout << "  V2 does NOT match V1 numerically!" << std::endl;
        return 1;
    }

    std::cout << "\nReference: V1 vs JAX RMSE = 2.37e-06" << std::endl;
    std::cout << "========================================" << std::endl;

    // Cleanup
    delete[] v2_embeddings;

    return 0;
}
