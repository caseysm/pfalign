/**
 * Unit tests for MPNN encoder.
 *
 * Tests scalar implementation for correctness.
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <limits>
#include <vector>

using pfalign::ScalarBackend;
using pfalign::mpnn::mpnn_forward;
using pfalign::mpnn::MPNNConfig;
using pfalign::mpnn::MPNNWeights;
using pfalign::mpnn::MPNNWorkspace;

// Global embedded weights (loaded once)
static auto load_global_weights() {
    static bool loaded = false;
    static MPNNWeights weights(3);
    static MPNNConfig config;
    if (!loaded) {
        auto [w, c, sw] = pfalign::weights::load_embedded_mpnn_weights();
        weights = std::move(w);
        config = c;
        loaded = true;
    }
    return std::make_pair(std::ref(weights), std::ref(config));
}

constexpr float TOLERANCE = 1e-4f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * Helper: Generate random backbone coordinates.
 */
void generate_protein_coords(float* coords, int L, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < L; i++) {
        // For each residue, generate 4 atoms (N, Ca, C, O)
        for (int atom = 0; atom < 4; atom++) {
            coords[(i * 4 + atom) * 3 + 0] = dist(rng);  // x
            coords[(i * 4 + atom) * 3 + 1] = dist(rng);  // y
            coords[(i * 4 + atom) * 3 + 2] = dist(rng);  // z
        }
    }
}
bool test_small_protein() {
    std::cout << "=== Test 1: Small Protein (10 residues) ===" << std::endl;

    std::mt19937 rng(42);
    const int L = 10;

    auto [weights, config] = load_global_weights();
    
    
    

    

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim, config.num_rbf);

    float* coords = new float[L * 4 * 3];
    generate_protein_coords(coords, L, rng);

    float* node_emb = new float[L * config.hidden_dim];

    // Run MPNN forward
    mpnn_forward<ScalarBackend>(coords, L, weights, config, node_emb, &workspace);

    // Check output shape and non-zero values
    bool has_nonzero = false;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        if (std::abs(node_emb[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }

    std::cout << "Output has non-zero values: " << (has_nonzero ? "YES" : "NO") << std::endl;

    // Check output is finite
    bool all_finite = true;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        if (!std::isfinite(node_emb[i])) {
            all_finite = false;
            break;
        }
    }

    std::cout << "Output is finite: " << (all_finite ? "YES" : "NO") << std::endl;

    delete[] coords;
    delete[] node_emb;

    if (!has_nonzero || !all_finite) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: Protein-scale (100 residues).
 */
bool test_protein_scale() {
    std::cout << "=== Test 2: Protein-Scale (100 residues) ===" << std::endl;

    std::mt19937 rng(42);
    const int L = 100;

    auto [weights, config] = load_global_weights();  // Defaults: 128 hidden, 3 layers, 30 neighbors

    

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim, config.num_rbf);

    float* coords = new float[L * 4 * 3];
    generate_protein_coords(coords, L, rng);

    float* node_emb = new float[L * config.hidden_dim];

    // Run MPNN forward
    mpnn_forward<ScalarBackend>(coords, L, weights, config, node_emb, &workspace);

    // Check output statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        sum += node_emb[i];
        sum_sq += node_emb[i] * node_emb[i];
    }

    float mean = sum / (L * config.hidden_dim);
    float variance = sum_sq / (L * config.hidden_dim) - mean * mean;
    float std_dev = std::sqrt(variance);

    std::cout << "Output mean: " << mean << std::endl;
    std::cout << "Output std: " << std_dev << std::endl;

    // Check that output has reasonable statistics
    bool reasonable = std::abs(mean) < 5.0f && std_dev > 0.01f && std_dev < 10.0f;

    delete[] coords;
    delete[] node_emb;

    if (!reasonable) {
        std::cout << "✗ FAIL: Output statistics unreasonable" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Deterministic output.
 */
bool test_deterministic() {
    std::cout << "=== Test 3: Deterministic Output ===" << std::endl;

    std::mt19937 rng(123);
    const int L = 20;

    auto [weights, config] = load_global_weights();
    
    
    

    

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim, config.num_rbf);

    float* coords = new float[L * 4 * 3];
    generate_protein_coords(coords, L, rng);

    float* node_emb1 = new float[L * config.hidden_dim];
    float* node_emb2 = new float[L * config.hidden_dim];

    // Run twice with same input
    mpnn_forward<ScalarBackend>(coords, L, weights, config, node_emb1, &workspace);
    mpnn_forward<ScalarBackend>(coords, L, weights, config, node_emb2, &workspace);

    // Check outputs are identical
    bool identical = true;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        if (!close(node_emb1[i], node_emb2[i], 1e-6f)) {
            identical = false;
            std::cout << "Mismatch at index " << i << ": "
                      << node_emb1[i] << " vs " << node_emb2[i] << std::endl;
            break;
        }
    }

    std::cout << "Outputs identical: " << (identical ? "YES" : "NO") << std::endl;

    delete[] coords;
    delete[] node_emb1;
    delete[] node_emb2;

    if (!identical) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 4: Different coordinates produce different outputs.
 */
bool test_coords_sensitivity() {
    std::cout << "=== Test 4: Coordinate Sensitivity ===" << std::endl;

    std::mt19937 rng(456);
    const int L = 15;

    auto [weights, config] = load_global_weights();
    
    
    

    

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim, config.num_rbf);

    float* coords1 = new float[L * 4 * 3];
    float* coords2 = new float[L * 4 * 3];
    generate_protein_coords(coords1, L, rng);
    generate_protein_coords(coords2, L, rng);

    float* node_emb1 = new float[L * config.hidden_dim];
    float* node_emb2 = new float[L * config.hidden_dim];

    // Run with different coordinates
    mpnn_forward<ScalarBackend>(coords1, L, weights, config, node_emb1, &workspace);
    mpnn_forward<ScalarBackend>(coords2, L, weights, config, node_emb2, &workspace);

    // Check outputs are different
    bool different = false;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        if (!close(node_emb1[i], node_emb2[i], 1e-3f)) {
            different = true;
            break;
        }
    }

    std::cout << "Different coords → different outputs: " << (different ? "YES" : "NO") << std::endl;

    delete[] coords1;
    delete[] coords2;
    delete[] node_emb1;
    delete[] node_emb2;

    if (!different) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

bool test_input_validation() {
    std::cout << "=== Test 5: Input Validation ===" << std::endl;

    int L = 4;
    auto [weights, config] = load_global_weights();

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim, config.num_rbf);

    std::vector<float> coords(L * 4 * 3, 1.0f);
    std::vector<float> node_emb(L * config.hidden_dim, 0.0f);

    // Test basic execution with valid inputs
    bool threw = false;
    try {
        mpnn_forward<ScalarBackend>(coords.data(), L, weights, config, node_emb.data(), &workspace);

        // Check output is finite
        bool all_finite = true;
        for (int i = 0; i < L * config.hidden_dim; i++) {
            if (!std::isfinite(node_emb[i])) {
                all_finite = false;
                break;
            }
        }

        if (!all_finite) {
            std::cout << "Output contains non-finite values" << std::endl;
            threw = true;
        }
    } catch (const std::exception& e) {
        std::cout << "Unexpected exception: " << e.what() << std::endl;
        threw = true;
    }

    if (threw) {
        std::cout << "✗ FAIL" << std::endl;
        std::cout << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  MPNN Encoder Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 5;

    if (test_small_protein()) passed++;
    if (test_protein_scale()) passed++;
    if (test_deterministic()) passed++;
    if (test_coords_sensitivity()) passed++;
    if (test_input_validation()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
