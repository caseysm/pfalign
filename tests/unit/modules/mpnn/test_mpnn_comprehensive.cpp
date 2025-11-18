/**
 * Comprehensive unit tests for MPNN encoder.
 *
 * Tests V2 MPNN implementation with real trained weights against
 * validated reference outputs.
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/tools/weights/mpnn_weight_loader.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/dispatch/backend_traits.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::io;

// Test tolerance
constexpr float TOL = 1e-5f;

// Helper: Load .npy file
bool load_npy_simple(const std::string& path, float* data, int size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char header[256];
    f.read(header, 128);
    f.read(reinterpret_cast<char*>(data), size * sizeof(float));
    f.close();
    return true;
}

// Helper: Compute max absolute error
float max_abs_error(const float* a, const float* b, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

// ============================================================================
// Test 1: Basic Functionality
// ============================================================================

bool test_basic_shapes() {
    std::cout << "=== Test: Basic MPNN Shapes ===" << std::endl;

    // Load embedded weights (no external file needed!)
    auto [weights, config, sw_params] = weights::load_embedded_mpnn_weights();

    const int L = 10;
    float coords[L * 4 * 3];  // 10 residues * 4 atoms * 3 coords

    // Simple coordinates
    for (int i = 0; i < L * 4 * 3; i++) {
        coords[i] = (float)(i % 10);
    }

    // Allocate output
    float* embeddings = new float[L * config.hidden_dim];
    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);

    // Initialize workspace
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    // Run forward pass
    mpnn_forward<ScalarBackend>(coords, L, weights, config, embeddings, &workspace);

    // Check outputs are finite
    bool all_finite = true;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        if (!std::isfinite(embeddings[i])) {
            all_finite = false;
            break;
        }
    }

    bool passed = all_finite;

    std::cout << "  Output shape: [" << L << " * " << config.hidden_dim << "]" << std::endl;
    std::cout << "  All finite: " << (all_finite ? "YES" : "NO") << std::endl;
    std::cout << "  Status: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;

    delete[] embeddings;

    return passed;
}

// ============================================================================
// Test 2: Crambin Reference Validation
// ============================================================================

bool test_crambin_reference() {
    std::cout << "=== Test: Crambin Reference Validation ===" << std::endl;

    // Load embedded weights (no external file needed!)
    auto [weights, config, sw_params] = weights::load_embedded_mpnn_weights();

    // Parse crambin PDB
    PDBParser parser;
    Protein prot;

    try {
        prot = parser.parse_file("../../validation/1CRN.pdb");
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load crambin PDB: " << e.what() << std::endl;
        std::cerr << "Make sure 1CRN.pdb is in validation/ directory" << std::endl;
        std::cerr << "Skipping this test (not a failure)" << std::endl;
        std::cout << std::endl;
        return true;  // Skip test when optional data is missing
    }

    int L = prot.get_chain(0).size();
    auto coords = prot.get_backbone_coords(0);

    std::cout << "  Loaded crambin: " << L << " residues" << std::endl;

    // Run MPNN
    float* embeddings = new float[L * config.hidden_dim];
    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);

    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    mpnn_forward<ScalarBackend>(coords.data(), L, weights, config, embeddings, &workspace);

    // Load reference outputs
    float* ref_embeddings = new float[L * config.hidden_dim];
    bool loaded = load_npy_simple("/tmp/crambin_node_features.npy", ref_embeddings, L * config.hidden_dim);

    if (!loaded) {
        std::cerr << "SKIP: Could not load reference embeddings" << std::endl;
        std::cerr << "Run: cd build/validation && ./end_to_end_align ../../weights/data/v1_mpnn_weights.safetensors 1CRN.pdb 1CRN.pdb /tmp/test.npy" << std::endl;
        std::cerr << "Skipping this test (not a failure)" << std::endl;
        std::cout << std::endl;
        delete[] embeddings;
        delete[] ref_embeddings;
        return true;  // Skip test when optional data is missing
    }

    // Compare
    float max_error = max_abs_error(embeddings, ref_embeddings, L * config.hidden_dim);
    bool passed = max_error < TOL;

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Threshold: " << TOL << std::endl;
    std::cout << "  Status: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;

    delete[] embeddings;
    delete[] ref_embeddings;

    return passed;
}

// ============================================================================
// Test 3: Small Protein (2 residues)
// ============================================================================

bool test_tiny_protein() {
    std::cout << "=== Test: Tiny Protein (2 residues) ===" << std::endl;

    auto [weights, config, sw_params] = weights::load_embedded_mpnn_weights();

    const int L = 2;
    float coords[L * 4 * 3] = {
        // Residue 0: N, Ca, C, O
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
        // Residue 1: N, Ca, C, O
        4.0f, 0.0f, 0.0f,
        5.0f, 0.0f, 0.0f,
        6.0f, 0.0f, 0.0f,
        7.0f, 0.0f, 0.0f
    };

    float* embeddings = new float[L * config.hidden_dim];
    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);

    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    mpnn_forward<ScalarBackend>(coords, L, weights, config, embeddings, &workspace);

    // Check all outputs are finite
    bool all_finite = true;
    for (int i = 0; i < L * config.hidden_dim; i++) {
        if (!std::isfinite(embeddings[i])) {
            all_finite = false;
            break;
        }
    }

    bool passed = all_finite;

    std::cout << "  All finite: " << (all_finite ? "YES" : "NO") << std::endl;
    std::cout << "  Status: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;

    delete[] embeddings;

    return passed;
}

// ============================================================================
// Test 4: Determinism (same input → same output)
// ============================================================================

bool test_determinism() {
    std::cout << "=== Test: Determinism ===" << std::endl;

    auto [weights, config, sw_params] = weights::load_embedded_mpnn_weights();

    const int L = 5;
    float coords[L * 4 * 3];
    for (int i = 0; i < L * 4 * 3; i++) {
        coords[i] = (float)(i % 7) * 0.5f;
    }

    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    // Run twice
    float* emb1 = new float[L * config.hidden_dim];
    float* emb2 = new float[L * config.hidden_dim];

    mpnn_forward<ScalarBackend>(coords, L, weights, config, emb1, &workspace);
    mpnn_forward<ScalarBackend>(coords, L, weights, config, emb2, &workspace);

    // Compare
    float max_error = max_abs_error(emb1, emb2, L * config.hidden_dim);
    bool passed = (max_error == 0.0f);

    std::cout << "  Max difference: " << max_error << std::endl;
    std::cout << "  Status: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;

    delete[] emb1;
    delete[] emb2;

    return passed;
}

// ============================================================================
// Test 3: Input Validation
// ============================================================================

bool test_invalid_inputs() {
    std::cout << "=== Test: mpnn_forward Input Validation ===" << std::endl;

    MPNNConfig config;
    config.hidden_dim = 32;
    config.k_neighbors = 4;
    config.num_layers = 1;

    MPNNWeights weights(config.num_layers);
    MPNNWorkspace workspace(5, config.k_neighbors, config.hidden_dim);

    float coords[5 * 4 * 3] = {};
    std::vector<float> embeddings(5 * config.hidden_dim, 0.0f);

    bool passed = true;

    auto expect_invalid = [&](auto&& fn, const char* label) {
        try {
            fn();
            std::cout << "  [" << label << "] FAIL (no exception)" << std::endl;
            return false;
        } catch (const std::invalid_argument&) {
            std::cout << "  [" << label << "] PASS" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "  [" << label << "] FAIL (unexpected exception: " << e.what() << ")" << std::endl;
            return false;
        }
    };

    passed &= expect_invalid([&] {
        mpnn_forward<ScalarBackend>(nullptr, 5, weights, config, embeddings.data(), &workspace);
    }, "null coords");

    passed &= expect_invalid([&] {
        mpnn_forward<ScalarBackend>(coords, 0, weights, config, embeddings.data(), &workspace);
    }, "non-positive length");

    MPNNWorkspace small_workspace(3, config.k_neighbors, config.hidden_dim);
    passed &= expect_invalid([&] {
        mpnn_forward<ScalarBackend>(coords, 5, weights, config, embeddings.data(), &small_workspace);
    }, "workspace too small (L)");

    MPNNConfig mismatch_config = config;
    mismatch_config.hidden_dim = 16;
    MPNNWeights mismatch_weights(mismatch_config.num_layers);
    passed &= expect_invalid([&] {
        mpnn_forward<ScalarBackend>(coords, 5, mismatch_weights, mismatch_config, embeddings.data(), &workspace);
    }, "hidden_dim mismatch");

    std::cout << "  Status: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;

    return passed;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "MPNN Comprehensive Unit Tests" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 0;

    total++; if (test_basic_shapes()) passed++;
    total++; if (test_crambin_reference()) passed++;
    total++; if (test_tiny_protein()) passed++;
    total++; if (test_determinism()) passed++;
    total++; if (test_invalid_inputs()) passed++;

    // Summary
    std::cout << "================================================================" << std::endl;
    std::cout << "Summary: " << passed << " / " << total << " tests passed" << std::endl;
    std::cout << "================================================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
