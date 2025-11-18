/**
 * Dual Score Comparison Test
 *
 * Demonstrates the difference between cosine-based and magnitude-aware scoring.
 *
 * Key insights:
 * - Cosine score: Pure directional similarity (magnitude-invariant)
 * - Magnitude score: Retains magnitude signal in final score
 * - Both use dot product posteriors (magnitude influences which positions align)
 *
 * This test helps determine empirically which metric better captures
 * protein alignment quality for your use case.
 */

#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>

using pfalign::ScalarBackend;
using pfalign::pairwise::pairwise_align_from_embeddings_with_dual_score;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;
using pfalign::memory::Arena;
using pfalign::memory::GrowableArena;

void test_identical_proteins() {
    std::cout << "=== Test 1: Identical Proteins ===" << std::endl;
    std::cout << "Expected: Both scores ~= 1.0" << std::endl << std::endl;

    const int L = 10;
    const int hidden_dim = 128;

    // Create identical embeddings with uniform magnitude
    std::vector<float> embeddings(L * hidden_dim);
    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings[i] = static_cast<float>(i % 10) * 0.1f;
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(2);  // 2 MB

    float partition, score_cos, score_mag;
    pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score_cos,
        &score_mag,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine):    " << std::fixed << std::setprecision(6) << score_cos << std::endl;
    std::cout << "  Score (magnitude): " << std::fixed << std::setprecision(6) << score_mag << std::endl;
    std::cout << "  Difference:        " << std::fixed << std::setprecision(6) << std::abs(score_cos - score_mag) << std::endl;
    std::cout << std::endl;
}

void test_variable_magnitude() {
    std::cout << "=== Test 2: Variable Magnitude Embeddings ===" << std::endl;
    std::cout << "Testing identical direction with varying magnitude" << std::endl << std::endl;

    const int L = 5;
    const int hidden_dim = 128;

    // Create embeddings with same direction but varying magnitudes
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L; i++) {
        float magnitude1 = 1.0f + i * 0.5f;  // Increasing magnitude: 1.0, 1.5, 2.0, 2.5, 3.0
        float magnitude2 = 1.0f + i * 0.5f;

        for (int k = 0; k < hidden_dim; k++) {
            float base_value = std::sin(static_cast<float>(k) * 0.1f);
            embeddings1[i * hidden_dim + k] = base_value * magnitude1;
            embeddings2[i * hidden_dim + k] = base_value * magnitude2;
        }
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(2);

    float partition, score_cos, score_mag;
    pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score_cos,
        &score_mag,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine):    " << std::fixed << std::setprecision(6) << score_cos << std::endl;
    std::cout << "  Score (magnitude): " << std::fixed << std::setprecision(6) << score_mag << std::endl;
    std::cout << "  Difference:        " << std::fixed << std::setprecision(6) << std::abs(score_cos - score_mag) << std::endl;
    std::cout << "  Note: Cosine should be ~1.0 (perfect direction), magnitude may differ" << std::endl;
    std::cout << std::endl;
}

void test_high_vs_low_confidence() {
    std::cout << "=== Test 3: High vs Low Confidence Positions ===" << std::endl;
    std::cout << "Simulating alignment with variable position confidence" << std::endl << std::endl;

    const int L = 10;
    const int hidden_dim = 128;

    // Create embeddings where some positions have high magnitude (high confidence)
    // and others have low magnitude (low confidence)
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L; i++) {
        // Positions 0-4: low confidence (magnitude 0.1)
        // Positions 5-9: high confidence (magnitude 2.0)
        float magnitude = (i < L/2) ? 0.1f : 2.0f;

        for (int k = 0; k < hidden_dim; k++) {
            float value = std::cos(static_cast<float>(i + k) * 0.1f);
            embeddings1[i * hidden_dim + k] = value * magnitude;
            embeddings2[i * hidden_dim + k] = value * magnitude;  // Identical
        }
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(2);

    float partition, score_cos, score_mag;
    pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score_cos,
        &score_mag,
        &arena
    );

    std::cout << "  Embedding structure: " << std::endl;
    std::cout << "    Positions 0-4: low magnitude (0.1) - low confidence" << std::endl;
    std::cout << "    Positions 5-9: high magnitude (2.0) - high confidence" << std::endl;
    std::cout << std::endl;
    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine):    " << std::fixed << std::setprecision(6) << score_cos << std::endl;
    std::cout << "  Score (magnitude): " << std::fixed << std::setprecision(6) << score_mag << std::endl;
    std::cout << "  Difference:        " << std::fixed << std::setprecision(6) << std::abs(score_cos - score_mag) << std::endl;
    std::cout << "  Interpretation: Both should be ~1.0 (perfect alignment)" << std::endl;
    std::cout << "    - Cosine: Measures directional similarity only" << std::endl;
    std::cout << "    - Magnitude: Weighted average, influenced by confidence" << std::endl;
    std::cout << std::endl;
}

void test_misaligned_with_magnitude() {
    std::cout << "=== Test 4: Misaligned Proteins with Variable Magnitude ===" << std::endl;
    std::cout << "Different proteins, but with magnitude patterns" << std::endl << std::endl;

    const int L = 10;
    const int hidden_dim = 128;

    // Create different embeddings with varying magnitudes
    std::vector<float> embeddings1(L * hidden_dim);
    std::vector<float> embeddings2(L * hidden_dim);

    for (int i = 0; i < L; i++) {
        float magnitude1 = 0.5f + i * 0.2f;
        float magnitude2 = 0.5f + (L - 1 - i) * 0.2f;  // Reverse magnitude pattern

        for (int k = 0; k < hidden_dim; k++) {
            embeddings1[i * hidden_dim + k] = std::sin(static_cast<float>(i + k) * 0.15f) * magnitude1;
            embeddings2[i * hidden_dim + k] = std::cos(static_cast<float>(i + k) * 0.12f) * magnitude2;
        }
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(2);

    float partition, score_cos, score_mag;
    pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
        embeddings1.data(), L,
        embeddings2.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score_cos,
        &score_mag,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine):    " << std::fixed << std::setprecision(6) << score_cos << std::endl;
    std::cout << "  Score (magnitude): " << std::fixed << std::setprecision(6) << score_mag << std::endl;
    std::cout << "  Difference:        " << std::fixed << std::setprecision(6) << std::abs(score_cos - score_mag) << std::endl;
    std::cout << "  Ratio (cos/mag):   " << std::fixed << std::setprecision(3)
              << (score_mag > 1e-6f ? score_cos / score_mag : 0.0f) << std::endl;
    std::cout << std::endl;
}

void test_invalid_inputs() {
    std::cout << "=== Test 5: Input Validation ===" << std::endl;

    const int L = 4;
    const int hidden_dim = 32;

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(1);

    std::vector<float> embeddings1(L * hidden_dim, 0.0f);
    std::vector<float> embeddings2(L * hidden_dim, 0.0f);

    embeddings1[0] = std::numeric_limits<float>::quiet_NaN();

    float partition, score_cos, score_mag;
    bool threw = false;
    try {
        pairwise_align_from_embeddings_with_dual_score<ScalarBackend>(
            embeddings1.data(), L,
            embeddings2.data(), L,
            hidden_dim,
            config,
            &workspace,
            &partition,
            &score_cos,
            &score_mag,
            &arena
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    std::cout << "  Throws on NaN embeddings: " << (threw ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Dual Score Comparison Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "These tests compare two scoring metrics:" << std::endl;
    std::cout << "1. Cosine score: Sigma P[i,j] * (dot[i,j] / (||e1[i]|| * ||e2[j]||))" << std::endl;
    std::cout << "   → Pure directional similarity (magnitude-invariant)" << std::endl;
    std::cout << std::endl;
    std::cout << "2. Magnitude score: Sigma P[i,j] * dot[i,j] / Sigma P[i,j] * ||e1[i]|| * ||e2[j]||" << std::endl;
    std::cout << "   → Magnitude-aware weighted average" << std::endl;
    std::cout << std::endl;
    std::cout << "Both use dot product posteriors (magnitude influences alignment)." << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_identical_proteins();
    test_variable_magnitude();
    test_high_vs_low_confidence();
    test_misaligned_with_magnitude();
    test_invalid_inputs();

    std::cout << "========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "- Small differences suggest magnitude has limited effect" << std::endl;
    std::cout << "- Large differences suggest magnitude significantly affects scoring" << std::endl;
    std::cout << "- Use these results to choose the appropriate metric for your use case" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
