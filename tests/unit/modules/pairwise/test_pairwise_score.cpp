/**
 * Unit tests for pairwise alignment score computation.
 *
 * Tests the scoring formula: score = sum(cosine_similarity ⊙ posteriors)
 * where cosine_similarity[i,j] = dot_product[i,j] / (||e1[i]|| * ||e2[j]||)
 *
 * Verifies that score ∈ [0, 1] with score=1 for identical proteins.
 */

#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <iostream>
#include <cmath>
#include <random>

using pfalign::ScalarBackend;
using pfalign::pairwise::pairwise_align_with_score;
using pfalign::pairwise::pairwise_align_from_embeddings_with_score;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;
using pfalign::memory::Arena;
using pfalign::memory::GrowableArena;

constexpr float TOLERANCE = 1e-5f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

//==============================================================================
// Test 1: Score from Synthetic Embeddings (Identical Proteins)
//==============================================================================

bool test_identical_proteins() {
    std::cout << "=== Test 1: Identical Proteins (Score ~= 1) ===" << std::endl;

    const int L = 10;
    const int hidden_dim = 128;

    // Create identical embeddings
    std::vector<float> embeddings(L * hidden_dim);
    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings[i] = static_cast<float>(i % 10) * 0.1f;
    }

    // Setup config and workspace
    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(1);  // 1 MB

    // Compute score
    float partition, score;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,  // Same embeddings
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score: " << score << std::endl;

    // Identical proteins should score close to 1.0
    bool passed = close(score, 1.0f, 0.01f);  // Within 1% of perfect score

    if (passed) {
        std::cout << "  ✓ PASS (score ~= 1.0 for identical proteins)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (expected ~=1.0, got " << score << ")" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 2: Score from Random Embeddings (Unrelated Proteins)
//==============================================================================

bool test_random_proteins() {
    std::cout << "=== Test 2: Random Proteins (Score ∈ [0, 1]) ===" << std::endl;

    const int L1 = 15;
    const int L2 = 12;
    const int hidden_dim = 128;

    // Create random embeddings
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> embeddings1(L1 * hidden_dim);
    std::vector<float> embeddings2(L2 * hidden_dim);

    for (int i = 0; i < L1 * hidden_dim; i++) {
        embeddings1[i] = dist(rng);
    }
    for (int i = 0; i < L2 * hidden_dim; i++) {
        embeddings2[i] = dist(rng);
    }

    // Setup config and workspace
    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;
    config.sw_config.gap_open = -1.0f;
    config.sw_config.gap_extend = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L1, L2, config);
    GrowableArena arena(1);

    // Compute score
    float partition, score;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1.data(), L1,
        embeddings2.data(), L2,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score: " << score << std::endl;

    // Score must be in [0, 1] by Cauchy-Schwarz
    bool passed = (score >= 0.0f && score <= 1.0f);

    if (passed) {
        std::cout << "  ✓ PASS (score ∈ [0, 1])" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (score = " << score << " not in [0, 1])" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 3: Score Consistency (Multiple Runs)
//==============================================================================

bool test_score_consistency() {
    std::cout << "=== Test 3: Score Consistency ===" << std::endl;

    const int L = 8;
    const int hidden_dim = 64;

    // Create fixed embeddings
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings1[i] = std::sin(static_cast<float>(i) * 0.1f);
        embeddings2[i] = std::cos(static_cast<float>(i) * 0.1f);
    }

    // Setup config and workspace
    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);

    // Run multiple times
    float scores[3];
    for (int run = 0; run < 3; run++) {
        GrowableArena arena(1);
        float partition;

        pairwise_align_from_embeddings_with_score<ScalarBackend>(
            embeddings1.data(), L,
            embeddings2.data(), L,
            hidden_dim,
            config,
            &workspace,
            &partition,
            &scores[run],
            &arena
        );
    }

    std::cout << "  Run 1: " << scores[0] << std::endl;
    std::cout << "  Run 2: " << scores[1] << std::endl;
    std::cout << "  Run 3: " << scores[2] << std::endl;

    // All runs should give identical results
    bool passed = close(scores[0], scores[1]) && close(scores[1], scores[2]);

    if (passed) {
        std::cout << "  ✓ PASS (consistent across runs)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (inconsistent scores)" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 4: Different Alignment Modes
//==============================================================================

bool test_different_modes() {
    std::cout << "=== Test 4: Different Alignment Modes ===" << std::endl;

    const int L = 10;
    const int hidden_dim = 64;

    // Create embeddings
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings1[i] = static_cast<float>(i % 7) * 0.2f;
        embeddings2[i] = static_cast<float>((i + 3) % 7) * 0.2f;
    }

    // Test all 6 modes
    PairwiseConfig::SWMode modes[] = {
        PairwiseConfig::SWMode::JAX_REGULAR,
        PairwiseConfig::SWMode::JAX_AFFINE_STANDARD,
        PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE,
        PairwiseConfig::SWMode::DIRECT_REGULAR,
        PairwiseConfig::SWMode::DIRECT_AFFINE,
        PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE
    };

    const char* mode_names[] = {
        "JAX_REGULAR",
        "JAX_AFFINE_STANDARD",
        "JAX_AFFINE_FLEXIBLE",
        "DIRECT_REGULAR",
        "DIRECT_AFFINE",
        "DIRECT_AFFINE_FLEXIBLE"
    };

    bool all_passed = true;

    for (int m = 0; m < 6; m++) {
        PairwiseConfig config;
        config.sw_mode = modes[m];
        config.mpnn_config.hidden_dim = hidden_dim;

        PairwiseWorkspace workspace(L, L, config);
        GrowableArena arena(1);

        float partition, score;
        pairwise_align_from_embeddings_with_score<ScalarBackend>(
            embeddings1.data(), L,
            embeddings2.data(), L,
            hidden_dim,
            config,
            &workspace,
            &partition,
            &score,
            &arena
        );

        std::cout << "  " << mode_names[m] << ": score=" << score << std::endl;

        bool mode_passed = (score >= 0.0f && score <= 1.0f);
        if (!mode_passed) {
            std::cout << "    ✗ FAIL (score out of bounds)" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "  ✓ PASS (all modes produce valid scores)" << std::endl;
    }

    std::cout << std::endl;
    return all_passed;
}

//==============================================================================
// Test 5: Large Proteins (Stress Test)
//==============================================================================

bool test_large_proteins() {
    std::cout << "=== Test 5: Large Proteins (50*50) ===" << std::endl;

    const int L1 = 50;
    const int L2 = 50;
    const int hidden_dim = 128;

    // Create random embeddings
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> embeddings1(L1 * hidden_dim);
    std::vector<float> embeddings2(L2 * hidden_dim);

    for (int i = 0; i < L1 * hidden_dim; i++) {
        embeddings1[i] = dist(rng);
    }
    for (int i = 0; i < L2 * hidden_dim; i++) {
        embeddings2[i] = dist(rng);
    }

    // Setup config and workspace
    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L1, L2, config);
    GrowableArena arena(16);  // 16 MB

    // Compute score
    float partition, score;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1.data(), L1,
        embeddings2.data(), L2,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score: " << score << std::endl;

    bool passed = (score >= 0.0f && score <= 1.0f);

    if (passed) {
        std::cout << "  ✓ PASS (large proteins handled correctly)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (score = " << score << ")" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 6: Low Temperature (Sharp Posteriors)
//==============================================================================

bool test_low_temperature() {
    std::cout << "=== Test 6: Low Temperature (T=0.001) ===" << std::endl;

    const int L = 10;
    const int hidden_dim = 64;

    // Create embeddings with some structure
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings1[i] = std::sin(static_cast<float>(i) * 0.1f);
        embeddings2[i] = std::cos(static_cast<float>(i) * 0.15f);
    }

    // Setup config with very low temperature
    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.temperature = 0.001f;  // Very low temperature
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(1);

    // Compute score
    float partition, score;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena
    );

    std::cout << "  Partition (T=0.001): " << partition << std::endl;
    std::cout << "  Score: " << score << std::endl;

    // At low temperature, posteriors should be sharply peaked
    // Score should still be in [0, 1]
    bool passed = (score >= 0.0f && score <= 1.0f);

    if (passed) {
        std::cout << "  ✓ PASS (low temperature score valid)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (score = " << score << ")" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 7: Temperature Comparison (High vs Low)
//==============================================================================

bool test_temperature_comparison() {
    std::cout << "=== Test 7: Temperature Comparison ===" << std::endl;

    const int L = 8;
    const int hidden_dim = 64;

    // Create embeddings
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings1[i] = static_cast<float>(i % 10) * 0.1f;
        embeddings2[i] = static_cast<float>((i + 2) % 10) * 0.1f;
    }

    // Test at high temperature
    PairwiseConfig config_high;
    config_high.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config_high.sw_config.temperature = 10.0f;  // High temperature
    config_high.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace_high(L, L, config_high);
    GrowableArena arena_high(1);

    float partition_high, score_high;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config_high,
        &workspace_high,
        &partition_high,
        &score_high,
        &arena_high
    );

    // Test at low temperature
    PairwiseConfig config_low;
    config_low.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config_low.sw_config.temperature = 0.01f;  // Low temperature
    config_low.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace_low(L, L, config_low);
    GrowableArena arena_low(1);

    float partition_low, score_low;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config_low,
        &workspace_low,
        &partition_low,
        &score_low,
        &arena_low
    );

    std::cout << "  High T (10.0):  partition=" << partition_high << ", score=" << score_high << std::endl;
    std::cout << "  Low T (0.01):   partition=" << partition_low << ", score=" << score_low << std::endl;

    // Both should be valid scores
    bool passed = (score_high >= 0.0f && score_high <= 1.0f);
    passed &= (score_low >= 0.0f && score_low <= 1.0f);

    // At higher temperature, posteriors are more spread out
    // This doesn't necessarily mean score_high < score_low, but both should be valid
    std::cout << "  Score ratio (high/low): " << (score_high / score_low) << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS (both temperatures produce valid scores)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (high=" << score_high << ", low=" << score_low << ")" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 8: Zero Similarity Edge Case
//==============================================================================

bool test_zero_similarity() {
    std::cout << "=== Test 8: Zero Similarity Edge Case ===" << std::endl;

    const int L = 5;
    const int hidden_dim = 32;

    // Create zero embeddings (will produce zero similarity)
    std::vector<float> embeddings1(L * hidden_dim, 0.0f);
    std::vector<float> embeddings2(L * hidden_dim, 0.0f);

    // Setup config and workspace
    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(1);

    // Compute score
    float partition, score;
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena
    );

    std::cout << "  Score: " << score << " (expected: 0.0)" << std::endl;

    bool passed = close(score, 0.0f);

    if (passed) {
        std::cout << "  ✓ PASS (zero similarity → zero score)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (score = " << score << ", expected 0.0)" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Pairwise Score Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 8;

    if (test_identical_proteins()) passed++;
    if (test_random_proteins()) passed++;
    if (test_score_consistency()) passed++;
    if (test_different_modes()) passed++;
    if (test_large_proteins()) passed++;
    if (test_low_temperature()) passed++;
    if (test_temperature_comparison()) passed++;
    if (test_zero_similarity()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
