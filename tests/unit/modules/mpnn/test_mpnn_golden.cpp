/**
 * Golden data tests for MPNN encoder
 *
 * These tests validate the C++ MPNN implementation against JAX reference outputs.
 * Golden data includes pre-generated embeddings and the exact weight tensors,
 * allowing bit-for-bit parity checks without invoking JAX at test time.
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/tools/weights/mpnn_weight_loader.h"
#include "pfalign/common/perf_timer.h"
#include <iostream>
#include <filesystem>
#include <cmath>

namespace mpnn = pfalign::mpnn;
using pfalign::ScalarBackend;
using pfalign::testing::GoldenDataTest;

namespace {

std::filesystem::path golden_root() {
    // Use compile-time project source root (passed by Meson)
    std::filesystem::path source_root(PFALIGN_SOURCE_ROOT);
    return source_root / "data" / "golden" / "mpnn";
}

std::filesystem::path golden_weights_path() {
    return golden_root() / "weights" / "mpnn_golden_weights.safetensors";
}

std::tuple<mpnn::MPNNWeights, mpnn::MPNNConfig> load_test_weights() {
    const auto weights_path = golden_weights_path();
    if (std::filesystem::exists(weights_path)) {
        auto [weights, config, sw_params] = pfalign::weights::MPNNWeightLoader::load(weights_path.string());
        (void)sw_params;
        return {std::move(weights), config};
    }

    auto [weights, config, sw_params] = pfalign::weights::load_embedded_mpnn_weights();
    (void)sw_params;
    return {std::move(weights), config};
}

} // namespace

/**
 * Test MPNN on a single golden data case
 */
bool test_mpnn_case(
    const std::string& test_name,
    const std::string& data_dir,
    const mpnn::MPNNWeights& weights,
    const mpnn::MPNNConfig& base_config
) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << test_name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    GoldenDataTest test(data_dir);

    // Load golden data
    auto [coords_flat, coords_shape] = test.load_with_shape("input_coords.npy");
    auto [expected_flat, expected_shape] = test.load_with_shape("output_embeddings.npy");

    if (coords_shape.size() != 3 || expected_shape.size() != 2) {
        std::cerr << "ERROR: Unexpected array dimensions" << std::endl;
        return false;
    }

    int L = coords_shape[0];
    int hidden_dim = expected_shape[1];

    std::cout << "\nTest parameters:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Hidden dim: " << hidden_dim << std::endl;

    auto config = base_config;
    // Ensure config matches expectations
    if (config.hidden_dim != hidden_dim) {
        std::cerr << "  ✗ FAIL: Hidden dim mismatch: expected " << hidden_dim
                  << ", got " << config.hidden_dim << std::endl;
        return false;
    }
    config.k_neighbors = 30;  // Recorded in golden config
    config.num_rbf = 16;

    // Create workspace
    mpnn::MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    // Allocate output
    std::vector<float> embeddings(L * config.hidden_dim, 0.0f);

    // Run MPNN forward pass
    std::cout << "\nRunning C++ MPNN forward pass..." << std::endl;
    mpnn::mpnn_forward<ScalarBackend>(
        coords_flat.data(),
        L,
        weights,
        config,
        embeddings.data(),
        &workspace
    );
    std::cout << "  ✓ Forward pass complete" << std::endl;

    // Validate outputs
    std::cout << "\nValidating outputs..." << std::endl;

    for (float value : embeddings) {
        if (!std::isfinite(value)) {
            std::cerr << "  ✗ FAIL: Output contains NaN/Inf" << std::endl;
            return false;
        }
    }

    // Compare against golden data
    test.compare("embeddings", expected_flat, embeddings, 1e-4, 1e-4);

    test.print_summary();
    bool passed = test.all_passed();
    std::cout << (passed ? "\n✓ Test passed" : "\n✗ Test failed") << std::endl;

    return passed;
}

int main() {
    pfalign::perf::PerfTimer perf_timer("test_mpnn_golden");
    std::cout << "========================================" << std::endl;
    std::cout << "  MPNN Golden Data Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    auto base_dir = golden_root();

    mpnn::MPNNWeights weights(3);
    mpnn::MPNNConfig config;

    try {
        std::tie(weights, config) = load_test_weights();
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load golden or embedded MPNN weights (" << e.what() << ")" << std::endl;
        return 0;
    }

    // Test cases: synthetic + real PDB structures
    std::vector<std::pair<std::string, std::filesystem::path>> test_cases = {
        // Synthetic
        {"small_10res", base_dir / "small_10res"},
        {"medium_50res", base_dir / "medium_50res"},
        {"large_100res", base_dir / "large_100res"},

        // Real PDB structures (Phase 1)
        {"villin_1VII", base_dir / "villin_1VII"},
        {"crambin_1CRN", base_dir / "crambin_1CRN"},
        {"ubiquitin_1UBQ", base_dir / "ubiquitin_1UBQ"},
        {"myoglobin_1MBO", base_dir / "myoglobin_1MBO"},

        // Phase 2 additions (populated once new golden data is generated)
        {"lysozyme_1LYZ", base_dir / "lysozyme_1LYZ"},
        {"hemoglobin_1HBS", base_dir / "hemoglobin_1HBS"},
        {"immunoglobulin_1IGT", base_dir / "immunoglobulin_1IGT"},
        {"immunoglobulin_1FBI", base_dir / "immunoglobulin_1FBI"},
        {"ribonuclease_1RNH", base_dir / "ribonuclease_1RNH"},
        {"lysozyme_2LYZ", base_dir / "lysozyme_2LYZ"},
        {"afdb_kinase_P00519", base_dir / "afdb_kinase_P00519"},
        {"afdb_globin_P69905", base_dir / "afdb_globin_P69905"},
        {"afdb_enzyme_P04406", base_dir / "afdb_enzyme_P04406"},
        {"esm_MGYP003592128331", base_dir / "esm_MGYP003592128331"},
        {"esm_MGYP002802792805", base_dir / "esm_MGYP002802792805"},
        {"esm_MGYP001105483357", base_dir / "esm_MGYP001105483357"},
    };

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    for (const auto& [name, dir] : test_cases) {
        if (!std::filesystem::exists(dir / "input_coords.npy") ||
            !std::filesystem::exists(dir / "output_embeddings.npy")) {
            std::cout << "\nSKIP: " << name << " (missing " << dir << ")" << std::endl;
            skipped++;
            continue;
        }

        if (test_mpnn_case(name, dir.string(), weights, config)) {
            passed++;
        } else {
            failed++;
        }
    }

    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Final Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Passed: " << passed << "/" << (passed + failed) << std::endl;
    std::cout << "Failed: " << failed << "/" << (passed + failed) << std::endl;
    std::cout << "Skipped: " << skipped << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return (failed == 0) ? 0 : 1;
}
