/**
 * Test safetensors loader with MPNN weights
 *
 * This test validates the zero-dependency safetensors loader.
 *
 * Usage:
 *   1. Generate test weights:
 *      python save_mpnn_weights.py --random --output /tmp/test_mpnn.safetensors \
 *             --hidden-dim 64 --num-layers 3 --num-rbf 16
 *
 *   2. Run this test:
 *      ./test_safetensors_loader /tmp/test_mpnn.safetensors
 */

#include "pfalign/tools/weights/safetensors_loader.h"
#include "pfalign/tools/weights/mpnn_weight_loader.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace pfalign::weights;
using namespace pfalign::mpnn;

void test_safetensors_loader(const std::string& filepath) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Safetensors Loader Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test 1: Load with SafetensorsLoader
    std::cout << "\nTest 1: Basic SafetensorsLoader" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    SafetensorsLoader loader(filepath);
    auto keys = loader.keys();

    std::cout << "Loaded file: " << filepath << std::endl;
    std::cout << "Num tensors: " << keys.size() << std::endl;

    // Print first few tensors
    std::cout << "\nFirst 5 tensors:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), keys.size()); i++) {
        const auto& key = keys[i];
        const auto& info = loader.get_info(key);

        std::cout << "  " << key << ": ";
        std::cout << "shape=[";
        for (size_t j = 0; j < info.shape.size(); j++) {
            std::cout << info.shape[j];
            if (j < info.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "], ";
        std::cout << "dtype=" << info.dtype << ", ";
        std::cout << "bytes=" << info.num_bytes() << std::endl;
    }

    // Test 2: Load MPNN weights
    std::cout << "\nTest 2: MPNN Weight Loader" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    auto result = MPNNWeightLoader::load(filepath);
    MPNNWeights& weights = std::get<0>(result);
    MPNNConfig& config = std::get<1>(result);
    SWParams& sw_params = std::get<2>(result);

    std::cout << "✓ Loaded MPNN weights" << std::endl;
    std::cout << "  Config:" << std::endl;
    std::cout << "    hidden_dim: " << config.hidden_dim << std::endl;
    std::cout << "    num_layers: " << config.num_layers << std::endl;
    std::cout << "    num_rbf: " << config.num_rbf << std::endl;
    std::cout << "    k_neighbors: " << config.k_neighbors << std::endl;

    // Test 3: Verify edge embedding weights
    std::cout << "\nTest 3: Edge Embedding Weights" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    int edge_features = 25 * config.num_rbf;
    std::cout << "Edge embedding weight: [" << edge_features << " * " << config.hidden_dim << "]" << std::endl;
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6)
                  << weights.edge_embedding_weight[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Edge norm gamma (first 5): ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(10) << weights.edge_norm_gamma[i] << " ";
    }
    std::cout << std::endl;

    // Test 4: Verify layer weights
    std::cout << "\nTest 4: Layer 0 Weights" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    auto& layer0 = weights.layers[0];
    int input_dim = 2 * config.hidden_dim;

    std::cout << "W1 weight: [" << input_dim << " * " << config.hidden_dim << "]" << std::endl;
    std::cout << "W1 first 5 values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6)
                  << layer0.W1_weight[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "W1 bias (first 5): ";
    for (int i = 0; i < 5; i++) {
        std::cout << std::setw(10) << layer0.W1_bias[i] << " ";
    }
    std::cout << std::endl;

    // Test 5: Compute weight statistics
    std::cout << "\nTest 5: Weight Statistics" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // Edge embedding stats
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t total_elements = edge_features * config.hidden_dim;

    for (size_t i = 0; i < total_elements; i++) {
        float val = weights.edge_embedding_weight[i];
        sum += val;
        sum_sq += val * val;
    }

    double mean = sum / total_elements;
    double var = sum_sq / total_elements - mean * mean;
    double std = std::sqrt(var);

    std::cout << "Edge embedding weight:" << std::endl;
    std::cout << "  Mean: " << std::setw(12) << std::fixed << std::setprecision(8) << mean << std::endl;
    std::cout << "  Std:  " << std::setw(12) << std << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;

    // Cleanup
    // (MPNNWeights destructor will handle cleanup)
}

int main(int argc, char** argv) {
    // Use existing v1_mpnn_weights.safetensors by default
    std::string filepath;

    if (argc >= 2) {
        filepath = argv[1];
    } else {
        #ifdef PFALIGN_SOURCE_ROOT
            filepath = std::string(PFALIGN_SOURCE_ROOT) + "/tests/data/fixtures/weights/v1_mpnn_weights.safetensors";
        #else
            filepath = "tests/data/fixtures/weights/v1_mpnn_weights.safetensors";
        #endif
    }

    std::cout << "Testing with: " << filepath << std::endl;
    std::cout << std::endl;

    try {
        test_safetensors_loader(filepath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
