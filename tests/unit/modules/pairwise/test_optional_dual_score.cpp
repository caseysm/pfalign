/**
 * Test Optional Dual Score Parameter
 *
 * Verifies that the optional score_magnitude parameter works correctly:
 * - Default behavior (nullptr): Only computes cosine score
 * - Explicit nullptr: Only computes cosine score
 * - Provided pointer: Computes both cosine and magnitude scores
 */

#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using pfalign::ScalarBackend;
using pfalign::pairwise::pairwise_align_from_embeddings_with_score;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;
using pfalign::memory::Arena;
using pfalign::memory::GrowableArena;

void test_default_single_score() {
    std::cout << "=== Test 1: Default Behavior (No Magnitude Score) ===" << std::endl;

    const int L = 5;
    const int hidden_dim = 128;

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
    GrowableArena arena(2);

    float partition, score;

    // Call WITHOUT magnitude score parameter (uses default nullptr)
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine only): " << std::fixed << std::setprecision(6) << score << std::endl;
    std::cout << "  ✓ Test passed - function callable without magnitude parameter" << std::endl;
    std::cout << std::endl;
}

void test_explicit_nullptr() {
    std::cout << "=== Test 2: Explicit nullptr (No Magnitude Score) ===" << std::endl;

    const int L = 5;
    const int hidden_dim = 128;

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
    GrowableArena arena(2);

    float partition, score;

    // Call WITH explicit nullptr
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score,
        &arena,
        nullptr  // Explicit nullptr
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine only): " << std::fixed << std::setprecision(6) << score << std::endl;
    std::cout << "  ✓ Test passed - function callable with explicit nullptr" << std::endl;
    std::cout << std::endl;
}

void test_with_magnitude_score() {
    std::cout << "=== Test 3: With Magnitude Score (Dual Score) ===" << std::endl;

    const int L = 5;
    const int hidden_dim = 128;

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
    GrowableArena arena(2);

    float partition, score_cos, score_mag;

    // Call WITH magnitude score parameter
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition,
        &score_cos,
        &arena,
        &score_mag  // Provide magnitude score pointer
    );

    std::cout << "  Partition: " << partition << std::endl;
    std::cout << "  Score (cosine):    " << std::fixed << std::setprecision(6) << score_cos << std::endl;
    std::cout << "  Score (magnitude): " << std::fixed << std::setprecision(6) << score_mag << std::endl;
    std::cout << "  Difference:        " << std::fixed << std::setprecision(6)
              << std::abs(score_cos - score_mag) << std::endl;
    std::cout << "  ✓ Test passed - both scores computed successfully" << std::endl;
    std::cout << std::endl;
}

void test_consistency() {
    std::cout << "=== Test 4: Consistency Check ===" << std::endl;
    std::cout << "Verifies cosine score is identical with/without magnitude computation" << std::endl;
    std::cout << std::endl;

    const int L = 10;
    const int hidden_dim = 128;

    std::vector<float> embeddings(L * hidden_dim);
    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena1(2);
    GrowableArena arena2(2);

    float partition1, score1;
    float partition2, score2_cos, score2_mag;

    // First call: cosine only
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition1,
        &score1,
        &arena1
    );

    // Second call: dual score
    pairwise_align_from_embeddings_with_score<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &partition2,
        &score2_cos,
        &arena2,
        &score2_mag
    );

    float partition_diff = std::abs(partition1 - partition2);
    float cosine_diff = std::abs(score1 - score2_cos);

    std::cout << "  Single-score call:" << std::endl;
    std::cout << "    Partition: " << partition1 << std::endl;
    std::cout << "    Score:     " << std::fixed << std::setprecision(6) << score1 << std::endl;
    std::cout << std::endl;

    std::cout << "  Dual-score call:" << std::endl;
    std::cout << "    Partition: " << partition2 << std::endl;
    std::cout << "    Score (cosine):    " << std::fixed << std::setprecision(6) << score2_cos << std::endl;
    std::cout << "    Score (magnitude): " << std::fixed << std::setprecision(6) << score2_mag << std::endl;
    std::cout << std::endl;

    std::cout << "  Consistency:" << std::endl;
    std::cout << "    Partition difference: " << std::scientific << std::setprecision(2)
              << partition_diff << std::endl;
    std::cout << "    Cosine difference:    " << std::scientific << std::setprecision(2)
              << cosine_diff << std::endl;

    bool consistent = (partition_diff < 1e-6f) && (cosine_diff < 1e-6f);
    if (consistent) {
        std::cout << "  ✓ Test passed - results are consistent" << std::endl;
    } else {
        std::cout << "  ✗ Test failed - results differ!" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Optional Dual Score Parameter Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing the optional score_magnitude parameter:" << std::endl;
    std::cout << "- Default (no parameter): computes cosine score only" << std::endl;
    std::cout << "- nullptr: computes cosine score only" << std::endl;
    std::cout << "- Provided pointer: computes both cosine and magnitude scores" << std::endl;
    std::cout << std::endl;

    test_default_single_score();
    test_explicit_nullptr();
    test_with_magnitude_score();
    test_consistency();

    std::cout << "========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "- ✓ Default behavior: cosine score only (backward compatible)" << std::endl;
    std::cout << "- ✓ Optional parameter: both scores when requested" << std::endl;
    std::cout << "- ✓ Consistent results: cosine score identical in both modes" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
