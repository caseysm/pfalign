/**
 * Unit tests for pairwise protein alignment pipeline.
 *
 * Tests the complete end-to-end pipeline:
 * Coords → MPNN → Similarity → Smith-Waterman
 */

#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <limits>

using pfalign::ScalarBackend;
using pfalign::pairwise::pairwise_align;
using pfalign::pairwise::pairwise_align_from_embeddings;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;
using pfalign::mpnn::MPNNWeights;
using pfalign::mpnn::MPNNConfig;

// Global embedded weights (loaded once)
static auto& get_global_weights() {
    static auto data = pfalign::weights::load_embedded_mpnn_weights();
    return data;
}

constexpr float TOL = 1e-4f;

bool close(float a, float b, float tol = TOL) {
    return std::abs(a - b) < tol;
}

/**
 * Helper: Generate synthetic protein coordinates.
 * Creates a simple helix-like structure for testing.
 */
void generate_test_coords(float* coords, int L, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Generate backbone atoms (N, Ca, C, O) - 4 atoms * 3 coords = 12 floats per residue
    for (int i = 0; i < L; i++) {
        // N atom
        coords[i * 12 + 0] = 1.5f * std::cos(i * 0.5f) + 0.1f * dist(gen);
        coords[i * 12 + 1] = 1.5f * std::sin(i * 0.5f) + 0.1f * dist(gen);
        coords[i * 12 + 2] = i * 1.5f + 0.1f * dist(gen);

        // Ca atom
        coords[i * 12 + 3] = coords[i * 12 + 0] + 0.5f;
        coords[i * 12 + 4] = coords[i * 12 + 1];
        coords[i * 12 + 5] = coords[i * 12 + 2] + 0.5f;

        // C atom
        coords[i * 12 + 6] = coords[i * 12 + 3] + 0.5f;
        coords[i * 12 + 7] = coords[i * 12 + 4] + 0.3f;
        coords[i * 12 + 8] = coords[i * 12 + 5];

        // O atom
        coords[i * 12 + 9] = coords[i * 12 + 6] + 0.3f;
        coords[i * 12 + 10] = coords[i * 12 + 7] + 0.2f;
        coords[i * 12 + 11] = coords[i * 12 + 8];
    }
}

/**
 * Test 1: Simple end-to-end alignment with JAX regular mode.
 */
bool test_simple_end_to_end() {
    std::cout << "=== Test 1: Simple End-to-End (JAX Regular) ===" << std::endl;

    int L1 = 10, L2 = 12;

    // Generate test data
    float* coords1 = new float[L1 * 4 * 3];
    float* coords2 = new float[L2 * 4 * 3];
    generate_test_coords(coords1, L1, 42);
    generate_test_coords(coords2, L2, 43);

    // Get embedded weights and config
    auto& [weights, mpnn_config, sw_params] = get_global_weights();

    // Configuration - use embedded weights config
    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap = -0.2f;
    config.sw_config.temperature = 1.0f;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;

    // Workspace
    PairwiseWorkspace workspace(L1, L2, config);

    // Run pipeline
    float partition;
    pairwise_align<ScalarBackend>(
        coords1, L1,
        coords2, L2,
        config,
        weights,
        &workspace,
        &partition
    );

    // Check result
    bool is_finite = std::isfinite(partition);
    bool is_reasonable = partition > -100.0f && partition < 100.0f;

    std::cout << "Partition: " << partition << std::endl;
    std::cout << "Is finite: " << (is_finite ? "YES" : "NO") << std::endl;
    std::cout << "Is reasonable: " << (is_reasonable ? "YES" : "NO") << std::endl;

    delete[] coords1;
    delete[] coords2;

    if (!is_finite || !is_reasonable) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: Compare JAX regular vs Direct regular modes.
 */
bool test_jax_vs_direct_modes() {
    std::cout << "=== Test 2: JAX vs Direct Modes ===" << std::endl;

    int L1 = 8, L2 = 10;

    float* coords1 = new float[L1 * 4 * 3];
    float* coords2 = new float[L2 * 4 * 3];
    generate_test_coords(coords1, L1, 100);
    generate_test_coords(coords2, L2, 101);

    // Get embedded weights and config
    auto& [weights, mpnn_config, sw_params] = get_global_weights();

    // Configuration
    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap = -0.3f;
    config.sw_config.temperature = 1.0f;

    // Test JAX mode
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    PairwiseWorkspace workspace_jax(L1, L2, config);
    float partition_jax;
    pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, config, weights, &workspace_jax, &partition_jax);

    // Test Direct mode
    config.sw_mode = PairwiseConfig::SWMode::DIRECT_REGULAR;
    PairwiseWorkspace workspace_direct(L1, L2, config);
    float partition_direct;
    pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, config, weights, &workspace_direct, &partition_direct);

    std::cout << "JAX partition:    " << partition_jax << std::endl;
    std::cout << "Direct partition: " << partition_direct << std::endl;

    // Both should be finite, but different
    bool both_finite = std::isfinite(partition_jax) && std::isfinite(partition_direct);
    bool different = std::abs(partition_jax - partition_direct) > 0.01f;

    delete[] coords1;
    delete[] coords2;

    if (!both_finite || !different) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Affine gap mode.
 */
bool test_affine_gaps() {
    std::cout << "=== Test 3: Affine Gap Mode ===" << std::endl;

    int L1 = 8, L2 = 8;

    float* coords1 = new float[L1 * 4 * 3];
    float* coords2 = new float[L2 * 4 * 3];
    generate_test_coords(coords1, L1, 200);
    generate_test_coords(coords2, L2, 201);

    // Get embedded weights and config
    auto& [weights, mpnn_config, sw_params] = get_global_weights();

    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap_open = -1.0f;
    config.sw_config.gap_extend = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.sw_mode = PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;

    PairwiseWorkspace workspace(L1, L2, config);

    float partition;
    pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, config, weights, &workspace, &partition);

    bool is_finite = std::isfinite(partition);
    bool is_reasonable = partition > -100.0f && partition < 100.0f;

    std::cout << "Partition (affine): " << partition << std::endl;

    delete[] coords1;
    delete[] coords2;

    if (!is_finite || !is_reasonable) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 4: From pre-computed embeddings.
 */
bool test_from_embeddings() {
    std::cout << "=== Test 4: From Pre-Computed Embeddings ===" << std::endl;

    // Get embedded config for hidden_dim
    auto& [weights, mpnn_config, sw_params] = get_global_weights();
    int hidden_dim = mpnn_config.hidden_dim;

    int L1 = 6, L2 = 8;

    // Generate synthetic embeddings
    std::mt19937 gen(300);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* embeddings1 = new float[L1 * hidden_dim];
    float* embeddings2 = new float[L2 * hidden_dim];

    for (int i = 0; i < L1 * hidden_dim; i++) embeddings1[i] = dist(gen);
    for (int i = 0; i < L2 * hidden_dim; i++) embeddings2[i] = dist(gen);

    // Configuration
    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap = -0.2f;
    config.sw_config.temperature = 1.0f;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;

    PairwiseWorkspace workspace(L1, L2, config);

    float partition;
    pairwise_align_from_embeddings<ScalarBackend>(
        embeddings1, L1,
        embeddings2, L2,
        hidden_dim,
        config,
        &workspace,
        &partition
    );

    bool is_finite = std::isfinite(partition);
    bool is_reasonable = partition > -100.0f && partition < 100.0f;

    std::cout << "Partition (from embeddings): " << partition << std::endl;

    delete[] embeddings1;
    delete[] embeddings2;

    if (!is_finite || !is_reasonable) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 5: Determinism.
 */
bool test_determinism() {
    std::cout << "=== Test 5: Determinism ===" << std::endl;

    int L1 = 10, L2 = 12;

    float* coords1 = new float[L1 * 4 * 3];
    float* coords2 = new float[L2 * 4 * 3];
    generate_test_coords(coords1, L1, 400);
    generate_test_coords(coords2, L2, 401);

    // Get embedded weights and config
    auto& [weights, mpnn_config, sw_params] = get_global_weights();

    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap = -0.2f;
    config.sw_config.temperature = 1.0f;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;

    PairwiseWorkspace workspace(L1, L2, config);

    // Run twice
    float partition1, partition2;
    pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, config, weights, &workspace, &partition1);
    pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, config, weights, &workspace, &partition2);

    bool deterministic = close(partition1, partition2, 1e-6f);

    std::cout << "Run 1: " << partition1 << std::endl;
    std::cout << "Run 2: " << partition2 << std::endl;
    std::cout << "Deterministic: " << (deterministic ? "YES" : "NO") << std::endl;

    delete[] coords1;
    delete[] coords2;

    if (!deterministic) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 6: Input validation.
 */
bool test_input_validation() {
    std::cout << "=== Test 6: Input Validation ===" << std::endl;

    int L1 = 5, L2 = 4;

    // Get embedded weights and config
    auto& [weights, mpnn_config, sw_params] = get_global_weights();

    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap = -0.2f;
    config.sw_config.temperature = 1.0f;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;

    float* coords1 = new float[L1 * 4 * 3];
    float* coords2 = new float[L2 * 4 * 3];
    generate_test_coords(coords1, L1, 510);
    generate_test_coords(coords2, L2, 511);

    PairwiseWorkspace workspace(L1, L2, config);

    float partition = 0.0f;
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
        pairwise_align<ScalarBackend>(nullptr, L1, coords2, L2, config, weights, &workspace, &partition);
    }, "null coords1");

    passed &= expect_invalid([&] {
        pairwise_align<ScalarBackend>(coords1, 0, coords2, L2, config, weights, &workspace, &partition);
    }, "non-positive L1");

    // Test hidden_dim mismatch
    PairwiseConfig mismatch_config = config;
    mismatch_config.mpnn_config.hidden_dim = 20;
    passed &= expect_invalid([&] {
        pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, mismatch_config, weights, &workspace, &partition);
    }, "hidden_dim mismatch");

    // Test layer count mismatch
    PairwiseConfig layer_mismatch_config = config;
    layer_mismatch_config.mpnn_config.num_layers = config.mpnn_config.num_layers + 1;
    passed &= expect_invalid([&] {
        pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, layer_mismatch_config, weights, &workspace, &partition);
    }, "weights layer mismatch");

    delete[] coords1;
    delete[] coords2;

    std::cout << "  Status: " << (passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::endl;

    return passed;
}

/**
 * Test 7: Protein-scale data.
 */
bool test_protein_scale() {
    std::cout << "=== Test 7: Protein-Scale Data ===" << std::endl;

    int L1 = 50, L2 = 60;

    float* coords1 = new float[L1 * 4 * 3];
    float* coords2 = new float[L2 * 4 * 3];
    generate_test_coords(coords1, L1, 500);
    generate_test_coords(coords2, L2, 501);

    // Get embedded weights and config
    auto& [weights, mpnn_config, sw_params] = get_global_weights();

    PairwiseConfig config;
    config.mpnn_config = mpnn_config;
    config.sw_config.gap = -0.2f;
    config.sw_config.temperature = 1.0f;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;

    PairwiseWorkspace workspace(L1, L2, config);

    float partition;
    pairwise_align<ScalarBackend>(coords1, L1, coords2, L2, config, weights, &workspace, &partition);

    bool is_finite = std::isfinite(partition);
    bool is_reasonable = partition > -1000.0f && partition < 1000.0f;

    std::cout << "L1=" << L1 << ", L2=" << L2 << ", D=" << mpnn_config.hidden_dim << std::endl;
    std::cout << "Partition: " << partition << std::endl;

    delete[] coords1;
    delete[] coords2;

    if (!is_finite || !is_reasonable) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Pairwise Alignment Pipeline Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 7;

    if (test_simple_end_to_end()) passed++;
    if (test_jax_vs_direct_modes()) passed++;
    if (test_affine_gaps()) passed++;
    if (test_from_embeddings()) passed++;
    if (test_determinism()) passed++;
    if (test_input_validation()) passed++;
    if (test_protein_scale()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
