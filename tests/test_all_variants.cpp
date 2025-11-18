/**
 * Comprehensive temperature invariance test for ALL 6 Smith-Waterman variants.
 */

#include "../pfalign/_core/src/pfalign/primitives/smith_waterman/smith_waterman.h"
#include "../pfalign/_core/src/pfalign/dispatch/scalar_traits.h"
#include "../pfalign/_core/src/pfalign/common/growable_arena.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

using namespace pfalign::smith_waterman;
using pfalign::ScalarBackend;

struct TestResult {
    std::string name;
    float variation;
    bool passed;
};

TestResult test_variant(
    const std::string& name,
    const std::vector<float>& scores,
    int L1, int L2,
    bool use_affine,
    bool use_jax,
    bool use_flexible
) {
    std::vector<float> temperatures = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f};
    std::vector<float> post_sums;

    for (float temp : temperatures) {
        SWConfig config;
        config.temperature = temp;

        if (use_affine) {
            config.affine = true;
            config.gap_open = -11.0f;
            config.gap_extend = -1.0f;
        } else {
            config.gap = -11.0f;
        }

        float partition;
        pfalign::memory::GrowableArena temp_arena(10);

        if (use_jax) {
            // JAX variants use L1*L2 matrices (no +1 padding)
            if (use_affine) {
                std::vector<float> hij(L1 * L2 * 3);
                std::vector<float> posteriors(L1 * L2);

                if (use_flexible) {
                    // JAX Affine Flexible
                    smith_waterman_jax_affine_flexible<ScalarBackend>(
                        scores.data(), L1, L2, config, hij.data(), &partition);
                    smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
                        hij.data(), scores.data(), L1, L2, config, partition,
                        posteriors.data(), &temp_arena);
                } else {
                    // JAX Affine (standard)
                    smith_waterman_jax_affine<ScalarBackend>(
                        scores.data(), L1, L2, config, hij.data(), &partition);
                    smith_waterman_jax_affine_backward<ScalarBackend>(
                        hij.data(), scores.data(), L1, L2, config, partition,
                        posteriors.data(), &temp_arena);
                }

                float post_sum = 0.0f;
                for (float p : posteriors) post_sum += p;
                post_sums.push_back(post_sum);

            } else {
                // JAX Regular
                std::vector<float> hij(L1 * L2);
                std::vector<float> posteriors(L1 * L2);

                smith_waterman_jax_regular<ScalarBackend>(
                    scores.data(), L1, L2, config, hij.data(), &partition);
                smith_waterman_jax_regular_backward<ScalarBackend>(
                    hij.data(), scores.data(), L1, L2, config, partition,
                    posteriors.data(), &temp_arena);

                float post_sum = 0.0f;
                for (float p : posteriors) post_sum += p;
                post_sums.push_back(post_sum);
            }

        } else {
            // Direct variants use (L1+1)*(L2+1) matrices
            if (use_affine) {
                std::vector<float> alpha((L1 + 1) * (L2 + 1) * 3);
                std::vector<float> posteriors(L1 * L2);

                if (use_flexible) {
                    // Direct Affine Flexible
                    smith_waterman_direct_affine_flexible<ScalarBackend>(
                        scores.data(), L1, L2, config, alpha.data(), &partition);
                    smith_waterman_direct_affine_flexible_backward<ScalarBackend>(
                        alpha.data(), scores.data(), L1, L2, config, partition,
                        posteriors.data(), &temp_arena);
                } else {
                    // Direct Affine
                    smith_waterman_direct_affine<ScalarBackend>(
                        scores.data(), L1, L2, config, alpha.data(), &partition);
                    smith_waterman_direct_affine_backward<ScalarBackend>(
                        alpha.data(), scores.data(), L1, L2, config, partition,
                        posteriors.data(), &temp_arena);
                }

                float post_sum = 0.0f;
                for (float p : posteriors) post_sum += p;
                post_sums.push_back(post_sum);

            } else {
                // Direct Regular
                std::vector<float> alpha((L1 + 1) * (L2 + 1));
                std::vector<float> posteriors(L1 * L2);

                smith_waterman_direct_regular<ScalarBackend>(
                    scores.data(), L1, L2, config, alpha.data(), &partition);
                smith_waterman_direct_regular_backward<ScalarBackend>(
                    alpha.data(), scores.data(), L1, L2, config, partition,
                    posteriors.data(), &temp_arena);

                float post_sum = 0.0f;
                for (float p : posteriors) post_sum += p;
                post_sums.push_back(post_sum);
            }
        }
    }

    // Calculate variation
    float mean_sum = 0.0f;
    for (float ps : post_sums) mean_sum += ps;
    mean_sum /= static_cast<float>(post_sums.size());

    float max_dev = 0.0f;
    for (float ps : post_sums) {
        float dev = (ps > mean_sum) ? (ps - mean_sum) : (mean_sum - ps);
        max_dev = (dev > max_dev) ? dev : max_dev;
    }
    float rel_dev = (max_dev / mean_sum) * 100.0f;

    TestResult result;
    result.name = name;
    result.variation = rel_dev;
    result.passed = rel_dev < 5.0f;

    return result;
}

void test_all_variants(int L1, int L2, const std::string& size_desc) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Testing All Variants: " << size_desc << " (" << L1 << "*" << L2 << ")\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Generate similarity matrix
    std::vector<float> scores(L1 * L2);
    unsigned int seed = 42;
    for (int i = 0; i < L1 * L2; i++) {
        seed = seed * 1103515245 + 12345;
        float r = static_cast<float>((seed / 65536) % 32768) / 32768.0f;
        scores[i] = r * 4.0f - 2.0f + 5.0f;
    }

    std::vector<TestResult> results;

    // Test all 6 variants
    results.push_back(test_variant("Direct Regular", scores, L1, L2, false, false, false));
    results.push_back(test_variant("Direct Affine", scores, L1, L2, true, false, false));
    results.push_back(test_variant("Direct Affine Flexible", scores, L1, L2, true, false, true));
    results.push_back(test_variant("JAX Regular", scores, L1, L2, false, true, false));
    results.push_back(test_variant("JAX Affine", scores, L1, L2, true, true, false));
    results.push_back(test_variant("JAX Affine Flexible", scores, L1, L2, true, true, true));

    // Print results
    std::cout << std::left << std::setw(30) << "Variant"
              << std::setw(15) << "Variation"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(80, '-') << "\n";

    int passed = 0;
    for (const auto& result : results) {
        std::cout << std::setw(30) << result.name
                  << std::setw(15) << (std::to_string(result.variation) + "%")
                  << std::setw(10) << (result.passed ? "✓ PASS" : "✗ FAIL") << "\n";
        if (result.passed) passed++;
    }

    std::cout << std::string(80, '-') << "\n";
    std::cout << "Passed: " << passed << "/6\n";
}

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "COMPREHENSIVE TEMPERATURE INVARIANCE TEST\n";
    std::cout << "Testing all 6 Smith-Waterman variants\n";
    std::cout << "Target: <5% variation for T ∈ [0.5, 2.0]\n";
    std::cout << std::string(80, '=') << "\n";

    // Test different sizes
    test_all_variants(10, 10, "Small");
    test_all_variants(30, 30, "Medium");
    test_all_variants(50, 40, "Large");

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST COMPLETE\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
