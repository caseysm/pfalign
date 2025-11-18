/**
 * Unit tests for alignment module.
 */

#include "pfalign/modules/alignment/alignment.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

using pfalign::ScalarBackend;
using pfalign::alignment::compute_alignment;
using pfalign::alignment::compute_alignment_with_posteriors;
using pfalign::alignment::AlignmentConfig;
using pfalign::alignment::AlignmentMode;
using pfalign::alignment::get_dp_matrix_size;

constexpr float TOLERANCE = 1e-4f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * Test 1: Simple diagonal alignment (identical sequences).
 */
bool test_simple_diagonal() {
    std::cout << "=== Test 1: Simple Diagonal (JAX Regular) ===" << std::endl;

    const int L = 5;

    // Perfect diagonal similarity matrix (identity)
    float scores[L * L];
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            scores[i * L + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    AlignmentConfig config;
    config.mode = AlignmentMode::JAX_REGULAR;
    config.sw_config.gap = -1.0f;
    config.sw_config.temperature = 1.0f;

    size_t dp_size = get_dp_matrix_size(L, L, config.mode);
    float* dp_matrix = new float[dp_size];
    float partition = 0.0f;

    compute_alignment<ScalarBackend>(scores, L, L, config, dp_matrix, &partition);

    std::cout << "Partition function: " << partition << std::endl;

    // For perfect diagonal, partition should be positive
    bool passed = partition > 0.0f && std::isfinite(partition);

    delete[] dp_matrix;

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

/**
 * Test 2: Test all 6 alignment modes.
 */
bool test_all_modes() {
    std::cout << "=== Test 2: All Alignment Modes ===" << std::endl;

    const int L1 = 8;
    const int L2 = 10;

    // Random similarity scores
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* scores = new float[L1 * L2];
    for (int i = 0; i < L1 * L2; i++) {
        scores[i] = dist(rng);
    }

    AlignmentMode modes[] = {
        AlignmentMode::JAX_REGULAR,
        AlignmentMode::JAX_AFFINE,
        AlignmentMode::JAX_AFFINE_FLEXIBLE,
        AlignmentMode::DIRECT_REGULAR,
        AlignmentMode::DIRECT_AFFINE,
        AlignmentMode::DIRECT_AFFINE_FLEXIBLE
    };

    const char* mode_names[] = {
        "JAX_REGULAR",
        "JAX_AFFINE",
        "JAX_AFFINE_FLEXIBLE",
        "DIRECT_REGULAR",
        "DIRECT_AFFINE",
        "DIRECT_AFFINE_FLEXIBLE"
    };

    bool all_passed = true;

    for (int m = 0; m < 6; m++) {
        AlignmentConfig config;
        config.mode = modes[m];
        config.sw_config.gap = -0.5f;
        config.sw_config.gap_open = -2.0f;
        config.sw_config.gap_extend = -0.5f;
        config.sw_config.temperature = 1.0f;

        size_t dp_size = get_dp_matrix_size(L1, L2, config.mode);
        float* dp_matrix = new float[dp_size];
        float partition = 0.0f;

        compute_alignment<ScalarBackend>(scores, L1, L2, config, dp_matrix, &partition);

        bool finite = std::isfinite(partition);

        std::cout << "  " << mode_names[m] << ": partition=" << partition
                  << " (finite=" << (finite ? "yes" : "no") << ")" << std::endl;

        if (!finite) {
            all_passed = false;
        }

        delete[] dp_matrix;
    }

    delete[] scores;

    if (all_passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return all_passed;
}

/**
 * Test 3: Backward pass (posteriors).
 */
bool test_posteriors() {
    std::cout << "=== Test 3: Posteriors (JAX Affine Flexible) ===" << std::endl;

    const int L = 12;

    // Random similarity scores
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* scores = new float[L * L];
    for (int i = 0; i < L * L; i++) {
        scores[i] = dist(rng);
    }

    AlignmentConfig config;
    config.mode = AlignmentMode::JAX_AFFINE_FLEXIBLE;
    config.sw_config.gap = -0.5f;
    config.sw_config.gap_open = -2.0f;
    config.sw_config.gap_extend = -0.5f;
    config.sw_config.temperature = 1.0f;

    size_t dp_size = get_dp_matrix_size(L, L, config.mode);
    float* dp_matrix = new float[dp_size];
    float* posteriors = new float[L * L];
    float partition = 0.0f;

    // Create arena for backward pass
    pfalign::memory::GrowableArena temp_arena(4);  // 4 MB

    compute_alignment_with_posteriors<ScalarBackend>(
        scores, L, L, config, dp_matrix, posteriors, &partition, &temp_arena
    );

    // Check posteriors properties
    bool all_finite = true;
    float sum_posteriors = 0.0f;
    float min_post = posteriors[0];
    float max_post = posteriors[0];

    for (int i = 0; i < L * L; i++) {
        if (!std::isfinite(posteriors[i])) {
            all_finite = false;
            break;
        }
        sum_posteriors += posteriors[i];
        min_post = std::min(min_post, posteriors[i]);
        max_post = std::max(max_post, posteriors[i]);
    }

    std::cout << "Partition: " << partition << std::endl;
    std::cout << "Posteriors sum: " << sum_posteriors << std::endl;
    std::cout << "Posteriors range: [" << min_post << ", " << max_post << "]" << std::endl;
    std::cout << "All finite: " << (all_finite ? "YES" : "NO") << std::endl;

    // Posteriors should be probabilities (non-negative, sum close to sequence length)
    bool passed = all_finite && sum_posteriors > 0.0f && min_post >= 0.0f && max_post <= 1.0f;

    delete[] scores;
    delete[] dp_matrix;
    delete[] posteriors;

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

/**
 * Test 4: Protein-scale alignment (100*150).
 */
bool test_protein_scale() {
    std::cout << "=== Test 4: Protein-Scale (100*150) ===" << std::endl;

    const int L1 = 100;
    const int L2 = 150;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* scores = new float[L1 * L2];
    for (int i = 0; i < L1 * L2; i++) {
        scores[i] = dist(rng);
    }

    AlignmentConfig config;
    config.mode = AlignmentMode::JAX_AFFINE_FLEXIBLE;
    config.sw_config.gap = -0.5f;
    config.sw_config.gap_open = -2.0f;
    config.sw_config.gap_extend = -0.5f;
    config.sw_config.temperature = 1.0f;

    size_t dp_size = get_dp_matrix_size(L1, L2, config.mode);
    float* dp_matrix = new float[dp_size];
    float* posteriors = new float[L1 * L2];
    float partition = 0.0f;

    // Create arena for backward pass (allocate enough for large proteins)
    pfalign::memory::GrowableArena temp_arena(64);  // 64 MB

    compute_alignment_with_posteriors<ScalarBackend>(
        scores, L1, L2, config, dp_matrix, posteriors, &partition, &temp_arena
    );

    bool all_finite = true;
    for (int i = 0; i < L1 * L2; i++) {
        if (!std::isfinite(posteriors[i])) {
            all_finite = false;
            break;
        }
    }

    std::cout << "Partition: " << partition << std::endl;
    std::cout << "All finite: " << (all_finite ? "YES" : "NO") << std::endl;

    bool passed = all_finite && std::isfinite(partition);

    delete[] scores;
    delete[] dp_matrix;
    delete[] posteriors;

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

/**
 * Test 5: Deterministic output.
 */
bool test_deterministic() {
    std::cout << "=== Test 5: Deterministic Output ===" << std::endl;

    const int L1 = 20;
    const int L2 = 30;

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* scores = new float[L1 * L2];
    for (int i = 0; i < L1 * L2; i++) {
        scores[i] = dist(rng);
    }

    AlignmentConfig config;
    config.mode = AlignmentMode::JAX_REGULAR;
    config.sw_config.gap = -0.5f;
    config.sw_config.temperature = 1.0f;

    size_t dp_size = get_dp_matrix_size(L1, L2, config.mode);
    float* dp1 = new float[dp_size];
    float* dp2 = new float[dp_size];
    float partition1 = 0.0f;
    float partition2 = 0.0f;

    // Compute twice
    compute_alignment<ScalarBackend>(scores, L1, L2, config, dp1, &partition1);
    compute_alignment<ScalarBackend>(scores, L1, L2, config, dp2, &partition2);

    // Check partition functions match
    bool partition_match = close(partition1, partition2, 1e-6f);

    // Check DP matrices match
    bool dp_match = true;
    for (size_t i = 0; i < dp_size; i++) {
        if (!close(dp1[i], dp2[i], 1e-6f)) {
            dp_match = false;
            break;
        }
    }

    std::cout << "Partition 1: " << partition1 << std::endl;
    std::cout << "Partition 2: " << partition2 << std::endl;
    std::cout << "Partitions match: " << (partition_match ? "YES" : "NO") << std::endl;
    std::cout << "DP matrices match: " << (dp_match ? "YES" : "NO") << std::endl;

    bool passed = partition_match && dp_match;

    delete[] scores;
    delete[] dp1;
    delete[] dp2;

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Alignment Module Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 5;

    if (test_simple_diagonal()) passed++;
    if (test_all_modes()) passed++;
    if (test_posteriors()) passed++;
    if (test_protein_scale()) passed++;
    if (test_deterministic()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
