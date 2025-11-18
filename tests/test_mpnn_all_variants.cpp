/**
 * Test ALL 6 Smith-Waterman variants with REAL MPNN similarity (unnormalized).
 *
 * This validates that Approach B works correctly with production-realistic
 * MPNN embeddings using the correct similarity metric (unnormalized dot product).
 */

#include "../pfalign/_core/src/pfalign/primitives/smith_waterman/smith_waterman.h"
#include "../pfalign/_core/src/pfalign/dispatch/scalar_traits.h"
#include "../pfalign/_core/src/pfalign/common/growable_arena.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

using namespace pfalign::smith_waterman;
using pfalign::ScalarBackend;

struct TestResult {
    std::string name;
    float variation;
    bool passed;
    float mean_posterior_sum;
};

// Load pre-computed MPNN similarity matrix
std::vector<float> load_mpnn_similarity(const std::string& filepath, int L1, int L2) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Could not open " << filepath << std::endl;
        std::cerr << "Make sure to run: python generate_mpnn_similarity.py" << std::endl;
        exit(1);
    }

    std::vector<float> sim(L1 * L2);
    file.read(reinterpret_cast<char*>(sim.data()), L1 * L2 * sizeof(float));

    if (!file) {
        std::cerr << "ERROR: Failed to read similarity matrix" << std::endl;
        exit(1);
    }

    return sim;
}

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
    result.mean_posterior_sum = mean_sum;

    return result;
}

int main() {
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "COMPREHENSIVE VALIDATION: ALL 6 VARIANTS * REAL MPNN SIMILARITY\n";
    std::cout << std::string(90, '=') << "\n\n";

    // Matrix dimensions (Myoglobin 1MBA vs 1MBO)
    const int L1 = 146;
    const int L2 = 153;

    std::cout << "Loading MPNN similarity matrix...\n";
    auto scores = load_mpnn_similarity("mpnn_similarity_1MBA_1MBO.bin", L1, L2);

    // Calculate statistics
    float min_val = scores[0], max_val = scores[0];
    float sum = 0.0f;
    for (float s : scores) {
        if (s < min_val) min_val = s;
        if (s > max_val) max_val = s;
        sum += s;
    }
    float mean_val = sum / scores.size();

    std::cout << "  Matrix: " << L1 << "*" << L2 << " (Myoglobin 1MBA vs 1MBO)\n";
    std::cout << "  Similarity: UNNORMALIZED dot product (MPNN embeddings)\n";
    std::cout << "  Range: [" << min_val << ", " << max_val << "]\n";
    std::cout << "  Mean: " << mean_val << "\n\n";

    std::cout << std::string(90, '=') << "\n";
    std::cout << "TESTING ALL 6 SMITH-WATERMAN VARIANTS\n";
    std::cout << "Temperature range: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]\n";
    std::cout << "Gap penalties: -11.0 (open/regular), -1.0 (extend)\n";
    std::cout << "Target: <5% variation (Approach B)\n";
    std::cout << std::string(90, '=') << "\n\n";

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
              << std::setw(15) << "Post. Sum"
              << std::setw(15) << "Expected"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(90, '-') << "\n";

    int passed = 0;
    for (const auto& result : results) {
        std::string var_str = std::to_string(result.variation).substr(0, 5) + "%";
        std::string sum_str = std::to_string(result.mean_posterior_sum).substr(0, 6);

        std::cout << std::setw(30) << result.name
                  << std::setw(15) << var_str
                  << std::setw(15) << sum_str
                  << std::setw(15) << ("~" + std::to_string(L1))
                  << std::setw(10) << (result.passed ? "✓ PASS" : "✗ FAIL") << "\n";
        if (result.passed) passed++;
    }

    std::cout << std::string(90, '-') << "\n";
    std::cout << "Overall: " << passed << "/6 tests passed\n";

    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(90, '=') << "\n";
    if (passed == 6) {
        std::cout << "✓ ALL 6 VARIANTS PASS with real MPNN similarity (unnormalized)\n";
        std::cout << "✓ Temperature invariance validated for production-realistic data\n";
        std::cout << "✓ Approach B implementation is correct and production-ready\n";
    } else {
        std::cout << "⚠ " << (6 - passed) << " variant(s) failed\n";
        std::cout << "⚠ Review implementation for failing variants\n";
    }
    std::cout << std::string(90, '=') << "\n\n";

    return (passed == 6) ? 0 : 1;
}
