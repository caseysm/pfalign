/**
 * Golden data tests for similarity computation.
 *
 * Validates C++ implementation against JAX reference outputs.
 */

#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/common/perf_timer.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

using namespace pfalign;
using namespace pfalign::similarity;
using namespace pfalign::testing;

namespace {

std::filesystem::path golden_root() {
    // Use compile-time project source root (passed by Meson)
    std::filesystem::path source_root(PFALIGN_SOURCE_ROOT);
    return source_root / "data" / "golden" / "similarity";
}

} // namespace

/**
 * Test a single similarity golden data case.
 */
bool test_similarity_case(const std::string& name, const std::string& data_dir) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    GoldenDataTest test(data_dir);

    // Load golden data
    auto [emb1_flat, emb1_shape] = test.load_with_shape("input_emb1.npy");
    auto [emb2_flat, emb2_shape] = test.load_with_shape("input_emb2.npy");
    auto [expected_flat, expected_shape] = test.load_with_shape("output_similarity.npy");

    if (emb1_shape.size() != 2 || emb2_shape.size() != 2 || expected_shape.size() != 2) {
        std::cerr << "ERROR: Unexpected array dimensions" << std::endl;
        return false;
    }

    int L1 = emb1_shape[0];
    int D1 = emb1_shape[1];
    int L2 = emb2_shape[0];
    int D2 = emb2_shape[1];

    if (D1 != D2) {
        std::cerr << "ERROR: Embedding dimensions don't match: " << D1 << " vs " << D2 << std::endl;
        return false;
    }

    int D = D1;

    std::cout << "\nTest parameters:" << std::endl;
    std::cout << "  L1 (seq 1 length): " << L1 << std::endl;
    std::cout << "  L2 (seq 2 length): " << L2 << std::endl;
    std::cout << "  D (embedding dim): " << D << std::endl;

    // Allocate output
    std::vector<float> actual(L1 * L2, 0.0f);

    // Run C++ similarity computation
    std::cout << "\nRunning C++ similarity computation..." << std::endl;
    compute_similarity<ScalarBackend>(
        emb1_flat.data(),
        emb2_flat.data(),
        actual.data(),
        L1, L2, D
    );
    std::cout << "  ✓ Computation complete" << std::endl;

    // Validate outputs
    std::cout << "\nValidating outputs..." << std::endl;

    // Check for NaN/Inf
    bool has_nan = false;
    bool has_inf = false;
    for (float val : actual) {
        if (std::isnan(val)) has_nan = true;
        if (std::isinf(val)) has_inf = true;
    }

    if (has_nan) {
        std::cerr << "  ✗ FAIL: Output contains NaN" << std::endl;
        return false;
    }
    if (has_inf) {
        std::cerr << "  ✗ FAIL: Output contains Inf" << std::endl;
        return false;
    }
    std::cout << "  ✓ No NaN/Inf values" << std::endl;

    // Compare against golden data
    // Tolerance: 1e-5 (dot product is numerically stable)
    test.compare("similarity", expected_flat, actual, 1e-5, 1e-5);

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
    pfalign::perf::PerfTimer perf_timer("test_similarity_golden");
    std::cout << "========================================" << std::endl;
    std::cout << "  Similarity Golden Data Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    auto base_dir = golden_root();

    // Test cases
    std::vector<std::pair<std::string, std::filesystem::path>> test_cases = {
        {"dot_32d_10x12", base_dir / "dot_32d_10x12"},
        {"dot_64d_20x25", base_dir / "dot_64d_20x25"},
        {"dot_128d_50x60", base_dir / "dot_128d_50x60"},
        {"dot_128d_8x10", base_dir / "dot_128d_8x10"},
        {"dot_48d_7x11", base_dir / "dot_48d_7x11"},
        {"dot_64d_16x16", base_dir / "dot_64d_16x16"},
        {"dot_64d_32x8", base_dir / "dot_64d_32x8"},
        {"dot_80d_5x40", base_dir / "dot_80d_5x40"},
        {"dot_96d_24x24", base_dir / "dot_96d_24x24"},
        {"dot_96d_12x48", base_dir / "dot_96d_12x48"},
        {"dot_128d_20x10", base_dir / "dot_128d_20x10"},
        {"dot_128d_4x64", base_dir / "dot_128d_4x64"},
    };

    int passed = 0;
    int failed = 0;
    int skipped = 0;

    for (const auto& [name, dir] : test_cases) {
        if (!std::filesystem::exists(dir / "input_emb1.npy")) {
            std::cout << "\nSKIP: " << name << " (missing " << dir << ")" << std::endl;
            skipped++;
            continue;
        }

        if (test_similarity_case(name, dir.string())) {
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
