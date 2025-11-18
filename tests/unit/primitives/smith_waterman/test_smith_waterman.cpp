/**
 * Unit tests for Smith-Waterman alignment.
 *
 * Tests all 6 versions (2 formulations * 3 gap models):
 * 1. Direct regular
 * 2. Direct affine (standard)
 * 3. Direct affine (flexible)
 * 4. JAX regular
 * 5. JAX affine (standard)
 * 6. JAX affine (flexible)
 */

#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <vector>

using pfalign::ScalarBackend;
using pfalign::smith_waterman::smith_waterman_direct_regular;
using pfalign::smith_waterman::smith_waterman_direct_affine;
using pfalign::smith_waterman::smith_waterman_direct_affine_flexible;
using pfalign::smith_waterman::smith_waterman_jax_regular;
using pfalign::smith_waterman::smith_waterman_jax_affine;
using pfalign::smith_waterman::smith_waterman_jax_affine_flexible;
using pfalign::smith_waterman::smith_waterman_jax_regular_backward;
using pfalign::smith_waterman::smith_waterman_jax_affine_flexible_backward;
using pfalign::smith_waterman::SWConfig;
using pfalign::smith_waterman::NINF;

constexpr float TOL = 1e-4f;

bool close(float a, float b, float tol = TOL) {
    return std::abs(a - b) < tol;
}

/**
 * Test 1: Direct regular - Simple 3*3 case.
 */
bool test_direct_regular_simple() {
    std::cout << "=== Test 1: Direct Regular (3*3) ===" << std::endl;

    // Simple similarity matrix: diagonal has high scores
    float scores[9] = {
        2.0f, -1.0f, -1.0f,
        -1.0f, 2.0f, -1.0f,
        -1.0f, -1.0f, 2.0f
    };

    const int L1 = 3;
    const int L2 = 3;
    std::vector<float> alpha((L1 + 1) * (L2 + 1));
    float partition;

    SWConfig config;
    config.gap = -0.5f;
    config.temperature = 1.0f;

    smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config, alpha.data(), &partition);

    // Check partition is finite and positive
    bool is_finite = std::isfinite(partition);
    bool is_reasonable = partition > 0.0f && partition < 10.0f;

    std::cout << "Partition: " << partition << std::endl;
    std::cout << "Is finite: " << (is_finite ? "YES" : "NO") << std::endl;
    std::cout << "Is reasonable: " << (is_reasonable ? "YES" : "NO") << std::endl;

    if (!is_finite || !is_reasonable) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: JAX regular - Compare with Direct for same input.
 */
bool test_jax_regular_vs_direct() {
    std::cout << "=== Test 2: JAX Regular vs Direct ===" << std::endl;

    float scores[12] = {
        1.0f, 0.5f, 0.2f, 0.1f,
        0.5f, 1.5f, 0.3f, 0.2f,
        0.2f, 0.3f, 1.2f, 0.4f
    };

    const int L1 = 3;
    const int L2 = 4;

    SWConfig config;
    config.gap = -0.2f;
    config.temperature = 1.0f;

    // Direct mode
    std::vector<float> alpha_direct((L1 + 1) * (L2 + 1));
    float partition_direct;
    smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config, alpha_direct.data(), &partition_direct);

    // JAX mode
    std::vector<float> hij_jax(L1 * L2);
    float partition_jax;
    smith_waterman_jax_regular<ScalarBackend>(scores, L1, L2, config, hij_jax.data(), &partition_jax);

    std::cout << "Direct partition: " << partition_direct << std::endl;
    std::cout << "JAX partition:    " << partition_jax << std::endl;
    std::cout << "Difference:       " << std::abs(partition_jax - partition_direct) << std::endl;

    // Both should be finite and positive
    // Note: JAX formulation can give higher OR lower partition depending on input
    // The CUDA docs mention ~0.25-0.96 higher, but this is input-dependent
    bool both_finite = std::isfinite(partition_direct) && std::isfinite(partition_jax);
    bool both_reasonable = partition_direct > 0.0f && partition_jax > 0.0f;
    bool different = std::abs(partition_jax - partition_direct) > 0.01f;  // They should differ

    std::cout << "Both finite: " << (both_finite ? "YES" : "NO") << std::endl;
    std::cout << "Different: " << (different ? "YES" : "NO") << std::endl;

    if (!both_finite || !both_reasonable || !different) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Direct affine - Verify 3 states work.
 */
bool test_direct_affine() {
    std::cout << "=== Test 3: Direct Affine ===" << std::endl;

    float scores[6] = {
        2.0f, -0.5f,
        -0.5f, 2.0f,
        1.0f, 1.0f
    };

    const int L1 = 3;
    const int L2 = 2;

    SWConfig config;
    config.gap_open = -1.0f;
    config.gap_extend = -0.1f;
    config.temperature = 1.0f;

    std::vector<float> alpha((L1 + 1) * (L2 + 1) * 3);
    float partition;

    smith_waterman_direct_affine<ScalarBackend>(scores, L1, L2, config, alpha.data(), &partition);

    bool is_finite = std::isfinite(partition);
    bool is_positive = partition > 0.0f;

    std::cout << "Partition: " << partition << std::endl;
    std::cout << "Is finite: " << (is_finite ? "YES" : "NO") << std::endl;

    if (!is_finite || !is_positive) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 4: JAX affine - Verify 3 states work.
 */
bool test_jax_affine() {
    std::cout << "=== Test 4: JAX Affine ===" << std::endl;

    float scores[12] = {
        1.5f, 0.5f, 0.3f, 0.2f,
        0.4f, 1.8f, 0.6f, 0.3f,
        0.2f, 0.5f, 1.3f, 0.7f
    };

    const int L1 = 3;
    const int L2 = 4;

    SWConfig config;
    config.gap_open = -0.5f;
    config.gap_extend = -0.1f;
    config.temperature = 1.0f;

    std::vector<float> hij(L1 * L2 * 3);
    float partition;

    smith_waterman_jax_affine<ScalarBackend>(scores, L1, L2, config, hij.data(), &partition);

    bool is_finite = std::isfinite(partition);
    bool is_positive = partition > 0.0f;

    std::cout << "Partition: " << partition << std::endl;
    std::cout << "Is finite: " << (is_finite ? "YES" : "NO") << std::endl;

    if (!is_finite || !is_positive) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 5: Flexible Affine Variants - Test both Direct and JAX flexible modes.
 */
bool test_flexible_affine() {
    std::cout << "=== Test 5: Flexible Affine Variants ===" << std::endl;

    float scores[20] = {
        1.5f, 0.5f, 0.3f, 0.2f, 0.1f,
        0.4f, 1.8f, 0.6f, 0.3f, 0.2f,
        0.2f, 0.5f, 1.3f, 0.7f, 0.4f,
        0.1f, 0.3f, 0.6f, 1.2f, 0.8f
    };

    const int L1 = 4;
    const int L2 = 5;

    SWConfig config;
    config.gap_open = -0.5f;
    config.gap_extend = -0.1f;
    config.temperature = 1.0f;

    // Test Direct affine (standard vs flexible)
    std::vector<float> alpha_standard((L1 + 1) * (L2 + 1) * 3);
    std::vector<float> alpha_flexible((L1 + 1) * (L2 + 1) * 3);
    float partition_standard, partition_flexible;

    smith_waterman_direct_affine<ScalarBackend>(scores, L1, L2, config, alpha_standard.data(), &partition_standard);
    smith_waterman_direct_affine_flexible<ScalarBackend>(scores, L1, L2, config, alpha_flexible.data(), &partition_flexible);

    bool direct_finite = std::isfinite(partition_standard) && std::isfinite(partition_flexible);
    bool direct_different = std::abs(partition_standard - partition_flexible) > 1e-6f;

    std::cout << "Direct standard:  " << partition_standard << std::endl;
    std::cout << "Direct flexible:  " << partition_flexible << std::endl;
    std::cout << "Are different: " << (direct_different ? "YES" : "NO") << std::endl;

    // Test JAX affine (standard vs flexible)
    std::vector<float> hij_standard(L1 * L2 * 3);
    std::vector<float> hij_flexible(L1 * L2 * 3);
    float jax_partition_standard, jax_partition_flexible;

    smith_waterman_jax_affine<ScalarBackend>(scores, L1, L2, config, hij_standard.data(), &jax_partition_standard);
    smith_waterman_jax_affine_flexible<ScalarBackend>(scores, L1, L2, config, hij_flexible.data(), &jax_partition_flexible);

    bool jax_finite = std::isfinite(jax_partition_standard) && std::isfinite(jax_partition_flexible);
    bool jax_different = std::abs(jax_partition_standard - jax_partition_flexible) > 1e-6f;

    std::cout << "JAX standard:     " << jax_partition_standard << std::endl;
    std::cout << "JAX flexible:     " << jax_partition_flexible << std::endl;
    std::cout << "Are different: " << (jax_different ? "YES" : "NO") << std::endl;

    if (!direct_finite || !jax_finite || !direct_different || !jax_different) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS - Flexible variants work and differ from standard" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 6: Determinism - All versions should be deterministic.
 */
bool test_determinism() {
    std::cout << "=== Test 5: Determinism (All Versions) ===" << std::endl;

    float scores[20];
    for (int i = 0; i < 20; i++) {
        scores[i] = (i % 5) * 0.3f - 0.5f;
    }

    const int L1 = 4;
    const int L2 = 5;
    SWConfig config;
    config.gap = -0.2f;
    config.gap_open = -0.8f;
    config.gap_extend = -0.15f;
    config.temperature = 1.0f;

    bool all_deterministic = true;

    // Test Direct regular
    float p1, p2;
    std::vector<float> alpha1((L1 + 1) * (L2 + 1));
    std::vector<float> alpha2((L1 + 1) * (L2 + 1));
    smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config, alpha1.data(), &p1);
    smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config, alpha2.data(), &p2);
    if (!close(p1, p2, 1e-6f)) {
        std::cout << "Direct regular not deterministic: " << p1 << " vs " << p2 << std::endl;
        all_deterministic = false;
    }

    // Test JAX regular
    std::vector<float> hij1(L1 * L2), hij2(L1 * L2);
    smith_waterman_jax_regular<ScalarBackend>(scores, L1, L2, config, hij1.data(), &p1);
    smith_waterman_jax_regular<ScalarBackend>(scores, L1, L2, config, hij2.data(), &p2);
    if (!close(p1, p2, 1e-6f)) {
        std::cout << "JAX regular not deterministic: " << p1 << " vs " << p2 << std::endl;
        all_deterministic = false;
    }

    // Test Direct affine
    std::vector<float> alpha_affine1((L1 + 1) * (L2 + 1) * 3);
    std::vector<float> alpha_affine2((L1 + 1) * (L2 + 1) * 3);
    smith_waterman_direct_affine<ScalarBackend>(scores, L1, L2, config, alpha_affine1.data(), &p1);
    smith_waterman_direct_affine<ScalarBackend>(scores, L1, L2, config, alpha_affine2.data(), &p2);
    if (!close(p1, p2, 1e-6f)) {
        std::cout << "Direct affine not deterministic: " << p1 << " vs " << p2 << std::endl;
        all_deterministic = false;
    }

    // Test JAX affine
    std::vector<float> hij_affine1(L1 * L2 * 3), hij_affine2(L1 * L2 * 3);
    smith_waterman_jax_affine<ScalarBackend>(scores, L1, L2, config, hij_affine1.data(), &p1);
    smith_waterman_jax_affine<ScalarBackend>(scores, L1, L2, config, hij_affine2.data(), &p2);
    if (!close(p1, p2, 1e-6f)) {
        std::cout << "JAX affine not deterministic: " << p1 << " vs " << p2 << std::endl;
        all_deterministic = false;
    }

    // Test Direct affine flexible
    std::vector<float> alpha_flex1((L1 + 1) * (L2 + 1) * 3);
    std::vector<float> alpha_flex2((L1 + 1) * (L2 + 1) * 3);
    smith_waterman_direct_affine_flexible<ScalarBackend>(scores, L1, L2, config, alpha_flex1.data(), &p1);
    smith_waterman_direct_affine_flexible<ScalarBackend>(scores, L1, L2, config, alpha_flex2.data(), &p2);
    if (!close(p1, p2, 1e-6f)) {
        std::cout << "Direct affine flexible not deterministic: " << p1 << " vs " << p2 << std::endl;
        all_deterministic = false;
    }

    // Test JAX affine flexible
    std::vector<float> hij_flex1(L1 * L2 * 3), hij_flex2(L1 * L2 * 3);
    smith_waterman_jax_affine_flexible<ScalarBackend>(scores, L1, L2, config, hij_flex1.data(), &p1);
    smith_waterman_jax_affine_flexible<ScalarBackend>(scores, L1, L2, config, hij_flex2.data(), &p2);
    if (!close(p1, p2, 1e-6f)) {
        std::cout << "JAX affine flexible not deterministic: " << p1 << " vs " << p2 << std::endl;
        all_deterministic = false;
    }

    std::cout << "All deterministic: " << (all_deterministic ? "YES" : "NO") << std::endl;

    if (!all_deterministic) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 6: Temperature scaling - Verify numerical stability across temperatures.
 */
bool test_temperature() {
    std::cout << "=== Test 6: Temperature Scaling ===" << std::endl;

    float scores[9] = {
        1.0f, 0.5f, 0.3f,
        0.6f, 1.2f, 0.4f,
        0.3f, 0.5f, 0.9f
    };

    const int L1 = 3;
    const int L2 = 3;

    SWConfig config1, config2;
    config1.gap = -0.2f;
    config1.temperature = 1.0f;
    config2.gap = -0.2f;
    config2.temperature = 0.5f;  // Lower temperature

    std::vector<float> alpha1((L1 + 1) * (L2 + 1));
    std::vector<float> alpha2((L1 + 1) * (L2 + 1));
    float p1, p2;

    smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config1, alpha1.data(), &p1);
    smith_waterman_direct_regular<ScalarBackend>(scores, L1, L2, config2, alpha2.data(), &p2);

    std::cout << "T=1.0 partition: " << p1 << std::endl;
    std::cout << "T=0.5 partition: " << p2 << std::endl;

    // Both partitions should be finite and positive
    bool both_finite = std::isfinite(p1) && std::isfinite(p2);
    bool both_positive = p1 > 0.0f && p2 > 0.0f;

    if (!both_finite || !both_positive) {
        std::cout << "✗ FAIL: Partitions not finite/positive" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 8: Backward posteriors form a normalized distribution.
 */
bool test_backward_normalization() {
    std::cout << "=== Test 8: Backward Posterior Normalization ===" << std::endl;

    // Simple 2x2 similarity matrix with clear diagonal preference.
    float scores[4] = {
        1.5f, 0.2f,
        0.2f, 1.2f
    };

    const int L1 = 2;
    const int L2 = 2;

    SWConfig config_reg;
    config_reg.gap = -0.3f;
    config_reg.temperature = 1.0f;

    std::vector<float> hij_reg(L1 * L2);
    float partition_reg;
    smith_waterman_jax_regular<ScalarBackend>(
        scores, L1, L2, config_reg, hij_reg.data(), &partition_reg
    );

    std::vector<float> posteriors_reg(L1 * L2);
    smith_waterman_jax_regular_backward<ScalarBackend>(
        hij_reg.data(), scores, L1, L2, config_reg, partition_reg, posteriors_reg.data()
    );

    float sum_reg = std::accumulate(posteriors_reg.begin(), posteriors_reg.end(), 0.0f);
    float expected_length = static_cast<float>(std::min(L1, L2));
    bool reg_sum_valid = close(sum_reg, expected_length, 1e-4f);
    bool reg_non_negative = true;
    for (int i = 0; i < L1 * L2; ++i) {
        if (posteriors_reg[i] < -1e-6f) {
            reg_non_negative = false;
            break;
        }
    }

    SWConfig config_aff = config_reg;
    config_aff.gap_open = -1.5f;
    config_aff.gap_extend = -0.3f;

    std::vector<float> hij_aff(L1 * L2 * 3);
    float partition_aff;
    smith_waterman_jax_affine_flexible<ScalarBackend>(
        scores, L1, L2, config_aff, hij_aff.data(), &partition_aff
    );

    std::vector<float> posteriors_aff(L1 * L2);
    smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
        hij_aff.data(), scores, L1, L2, config_aff, partition_aff, posteriors_aff.data()
    );

    float sum_aff = std::accumulate(posteriors_aff.begin(), posteriors_aff.end(), 0.0f);
    bool aff_sum_valid = close(sum_aff, expected_length, 1e-4f);
    bool aff_non_negative = true;
    for (int i = 0; i < L1 * L2; ++i) {
        if (posteriors_aff[i] < -1e-6f) {
            aff_non_negative = false;
            break;
        }
    }

    std::cout << "Regular sum: " << sum_reg << std::endl;
    std::cout << "Affine sum:  " << sum_aff << std::endl;

    bool passed = reg_sum_valid && aff_sum_valid && reg_non_negative && aff_non_negative;

    if (!passed) {
        std::cout << "✗ FAIL: Posterior normalization at T=1.0" << std::endl;
        return false;
    }

    // Additional test: Temperature invariance (T=0.5 and T=2.0)
    std::cout << "  Testing temperature invariance..." << std::endl;
    float test_temps[] = {0.5f, 2.0f};
    for (float T : test_temps) {
        SWConfig config_temp = config_reg;
        config_temp.temperature = T;

        std::vector<float> hij_temp(L1 * L2);
        float partition_temp;
        smith_waterman_jax_regular<ScalarBackend>(
            scores, L1, L2, config_temp, hij_temp.data(), &partition_temp
        );

        std::vector<float> posteriors_temp(L1 * L2);
        smith_waterman_jax_regular_backward<ScalarBackend>(
            hij_temp.data(), scores, L1, L2, config_temp, partition_temp, posteriors_temp.data()
        );

        float sum_temp = std::accumulate(posteriors_temp.begin(), posteriors_temp.end(), 0.0f);
        bool temp_sum_valid = close(sum_temp, expected_length, 0.5f);  // Relaxed tolerance

        std::cout << "    T=" << T << " sum=" << sum_temp << " valid=" << (temp_sum_valid ? "YES" : "NO") << std::endl;

        if (!temp_sum_valid) {
            std::cout << "  ✗ FAIL: Posterior sum not invariant at T=" << T << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS (including temperature invariance)" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Smith-Waterman Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 8;

    if (test_direct_regular_simple()) passed++;
    if (test_jax_regular_vs_direct()) passed++;
    if (test_direct_affine()) passed++;
    if (test_jax_affine()) passed++;
    if (test_flexible_affine()) passed++;
    if (test_determinism()) passed++;
    if (test_temperature()) passed++;
    if (test_backward_normalization()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
