/**
 * Golden data tests for Smith-Waterman alignment.
 *
 * Validates C++ JAX-mode forward and backward passes against JAX reference outputs.
 *
 * Tests 2 modes * 3 sizes = 6 cases:
 * - Regular gap (JAX mode)
 * - Affine gap (JAX mode)
 * - Small (8*10), Medium (20*25), Large (50*60)
 *
 * NOTE: Direct modes use a different algorithm and are tested separately via parity tests.
 */

#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/common/perf_timer.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>

using namespace pfalign;
using namespace pfalign::smith_waterman;
using namespace pfalign::testing;

namespace {

std::filesystem::path golden_root() {
    // Use compile-time project source root (passed by Meson)
    std::filesystem::path source_root(PFALIGN_SOURCE_ROOT);
    return source_root / "data" / "golden" / "smith_waterman";
}

} // namespace

/**
 * Parse test mode from name.
 */
struct TestMode {
    bool affine;
    bool jax_mode;
    std::string size_name;
};

TestMode parse_mode(const std::string& name) {
    TestMode mode;
    mode.affine = (name.find("affine") != std::string::npos);
    mode.jax_mode = (name.find("jax") != std::string::npos);

    if (name.find("small") != std::string::npos) {
        mode.size_name = "small";
    } else if (name.find("medium") != std::string::npos) {
        mode.size_name = "medium";
    } else if (name.find("large") != std::string::npos) {
        mode.size_name = "large";
    }

    return mode;
}

/**
 * Get SW config for this test case.
 * Hard-coded constants matching golden data generation.
 */
SWConfig get_config(bool affine) {
    SWConfig config;
    config.affine = affine;
    config.temperature = 1.0f;

    if (affine) {
        // Affine mode: gap_open = -0.5, gap_extend = -0.1
        config.gap_open = -0.5f;
        config.gap_extend = -0.1f;
    } else {
        // Regular mode: gap = -0.1
        config.gap = -0.1f;
    }

    return config;
}

/**
 * Test a single SW golden data case.
 */
bool test_sw_case(const std::string& name, const std::string& data_dir) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    GoldenDataTest test(data_dir);

    // Load golden data
    auto [scores_flat, scores_shape] = test.load_with_shape("input_similarity.npy");
    auto expected_partition = test.load("output_partition.npy");
    auto [expected_posterior_flat, posterior_shape] = test.load_with_shape("output_posterior.npy");

    if (scores_shape.size() != 2 || posterior_shape.size() != 2) {
        std::cerr << "ERROR: Unexpected array dimensions" << std::endl;
        return false;
    }

    int L1 = scores_shape[0];
    int L2 = scores_shape[1];

    // Parse test mode
    TestMode mode = parse_mode(name);
    SWConfig config = get_config(mode.affine);

    std::cout << "\nTest parameters:" << std::endl;
    std::cout << "  L1 (seq 1 length): " << L1 << std::endl;
    std::cout << "  L2 (seq 2 length): " << L2 << std::endl;
    std::cout << "  Mode: " << (mode.affine ? "affine" : "regular") << std::endl;
    std::cout << "  Batch type: " << (mode.jax_mode ? "JAX" : "direct") << std::endl;
    if (mode.affine) {
        std::cout << "  Gap open: " << config.gap_open << std::endl;
        std::cout << "  Gap extend: " << config.gap_extend << std::endl;
    } else {
        std::cout << "  Gap: " << config.gap << std::endl;
    }
    std::cout << "  Temperature: " << config.temperature << std::endl;

    // Allocate DP workspace (size depends on mode)
    size_t alpha_size;
    if (mode.jax_mode) {
        // JAX mode: L1 * L2 (or 3 * L1 * L2 for affine)
        alpha_size = mode.affine ? (3 * L1 * L2) : (L1 * L2);
    } else {
        // Direct mode: (L1+1) * (L2+1) (or 3 * (L1+1) * (L2+1) for affine)
        alpha_size = mode.affine ? (3 * (L1+1) * (L2+1)) : ((L1+1) * (L2+1));
    }

    std::vector<float> alpha(alpha_size, 0.0f);
    float partition = 0.0f;

    // Run forward pass
    std::cout << "\nRunning forward pass..." << std::endl;

    if (mode.affine && mode.jax_mode) {
        // Use flexible variant to match JAX sw_affine(restrict_turns=True, penalize_turns=True)
        smith_waterman_jax_affine_flexible<ScalarBackend>(
            scores_flat.data(), L1, L2, config, alpha.data(), &partition
        );
    } else if (mode.affine && !mode.jax_mode) {
        smith_waterman_direct_affine<ScalarBackend>(
            scores_flat.data(), L1, L2, config, alpha.data(), &partition
        );
    } else if (!mode.affine && mode.jax_mode) {
        smith_waterman_jax_regular<ScalarBackend>(
            scores_flat.data(), L1, L2, config, alpha.data(), &partition
        );
    } else { // regular, direct
        smith_waterman_direct_regular<ScalarBackend>(
            scores_flat.data(), L1, L2, config, alpha.data(), &partition
        );
    }

    std::cout << "  ✓ Forward pass complete" << std::endl;
    std::cout << "  Partition: " << partition << std::endl;

    // Check partition for NaN/Inf
    if (std::isnan(partition) || std::isinf(partition)) {
        std::cerr << "  ✗ FAIL: Partition is NaN/Inf" << std::endl;
        return false;
    }

    // Run backward pass to compute posteriors
    std::cout << "\nRunning backward pass..." << std::endl;
    std::vector<float> posteriors(L1 * L2, 0.0f);

    // Create arena for backward pass (4 MB should be sufficient for these test sizes)
    pfalign::memory::GrowableArena temp_arena(4);

    // Backward pass signatures: (alpha, scores, L1, L2, config, partition, posteriors, arena)
    if (mode.affine && mode.jax_mode) {
        // Use flexible variant to match JAX sw_affine(restrict_turns=True, penalize_turns=True)
        smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
            alpha.data(), scores_flat.data(), L1, L2, config, partition, posteriors.data(), &temp_arena
        );
    } else if (mode.affine && !mode.jax_mode) {
        smith_waterman_direct_affine_backward<ScalarBackend>(
            alpha.data(), scores_flat.data(), L1, L2, config, partition, posteriors.data(), &temp_arena
        );
    } else if (!mode.affine && mode.jax_mode) {
        smith_waterman_jax_regular_backward<ScalarBackend>(
            alpha.data(), scores_flat.data(), L1, L2, config, partition, posteriors.data(), &temp_arena
        );
    } else { // regular, direct
        smith_waterman_direct_regular_backward<ScalarBackend>(
            alpha.data(), scores_flat.data(), L1, L2, config, partition, posteriors.data(), &temp_arena
        );
    }

    std::cout << "  ✓ Backward pass complete" << std::endl;

    // Validate outputs
    std::cout << "\nValidating outputs..." << std::endl;

    // Check posteriors for NaN/Inf
    bool has_nan = false;
    bool has_inf = false;
    for (float val : posteriors) {
        if (std::isnan(val)) has_nan = true;
        if (std::isinf(val)) has_inf = true;
    }

    if (has_nan) {
        std::cerr << "  ✗ FAIL: Posteriors contain NaN" << std::endl;
        return false;
    }
    if (has_inf) {
        std::cerr << "  ✗ FAIL: Posteriors contain Inf" << std::endl;
        return false;
    }
    std::cout << "  ✓ No NaN/Inf in posteriors" << std::endl;

    // Compare against golden data
    // Tolerance: 1e-4 (DP accumulates numerical error)
    test.compare_scalar("partition", expected_partition[0], partition, 1e-4, 1e-4);
    test.compare("posterior", expected_posterior_flat, posteriors, 1e-4, 1e-4);

    test.print_summary();

    if (test.all_passed()) {
        std::cout << "\n✓ Test passed" << std::endl;
        return true;
    } else {
        std::cout << "\n✗ Test failed" << std::endl;
        return false;
    }
}

int main() {
    pfalign::perf::PerfTimer perf_timer("test_sw_golden");
    std::cout << "========================================" << std::endl;
    std::cout << "  Smith-Waterman Golden Data Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    auto base_dir = golden_root();

    // Test cases: 2 modes * 3 sizes (JAX)
    std::vector<std::pair<std::string, std::filesystem::path>> test_cases = {
        {"regular_jax_small_8x10", base_dir / "regular_jax_small_8x10"},
        {"regular_jax_medium_20x25", base_dir / "regular_jax_medium_20x25"},
        {"regular_jax_large_50x60", base_dir / "regular_jax_large_50x60"},
        {"affine_jax_small_8x10", base_dir / "affine_jax_small_8x10"},
        {"affine_jax_medium_20x25", base_dir / "affine_jax_medium_20x25"},
        {"affine_jax_large_50x60", base_dir / "affine_jax_large_50x60"},
    };

    int passed = 0;
    int failed = 0;

    for (const auto& [name, dir] : test_cases) {
        if (test_sw_case(name, dir.string())) {
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
    std::cout << std::string(60, '=') << std::endl;

    return (failed == 0) ? 0 : 1;
}
