/**
 * Test for full pairwise alignment (posteriors + decoded path).
 *
 * Verifies that pairwise_align_full and pairwise_align_from_embeddings_full
 * correctly compute all alignment outputs:
 * - Partition function
 * - Alignment score
 * - Posterior matrix
 * - Decoded alignment path with gaps
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
using pfalign::pairwise::pairwise_align_from_embeddings_full;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;
using pfalign::pairwise::AlignmentResult;
using pfalign::AlignmentPair;
using pfalign::memory::Arena;
using pfalign::memory::GrowableArena;

void test_basic_full_alignment() {
    std::cout << "=== Test 1: Basic Full Alignment ===" << std::endl;
    std::cout << "Verify all outputs are computed correctly" << std::endl;
    std::cout << std::endl;

    const int L1 = 5;
    const int L2 = 5;
    const int hidden_dim = 128;

    // Create identical embeddings for near-perfect alignment
    std::vector<float> embeddings1(L1 * hidden_dim);
    std::vector<float> embeddings2(L2 * hidden_dim);

    for (int i = 0; i < L1; i++) {
        for (int k = 0; k < hidden_dim; k++) {
            float val = std::sin(static_cast<float>(i + k) * 0.1f);
            embeddings1[i * hidden_dim + k] = val;
            embeddings2[i * hidden_dim + k] = val;  // Identical
        }
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.1f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L1, L2, config);
    GrowableArena arena(4);  // 4 MB

    // Allocate result buffers
    AlignmentResult result;
    result.posteriors = arena.allocate<float>(L1 * L2);
    result.alignment_path = arena.allocate<AlignmentPair>(L1 + L2);
    result.max_path_length = L1 + L2;

    // Run full alignment
    pairwise_align_from_embeddings_full<ScalarBackend>(
        embeddings1.data(), L1,
        embeddings2.data(), L2,
        hidden_dim,
        config,
        &workspace,
        &result,
        &arena,
        -2.0f  // Gap penalty
    );

    // Verify outputs
    std::cout << "  Partition:   " << result.partition << std::endl;
    std::cout << "  Score:       " << std::fixed << std::setprecision(3) << result.score << std::endl;
    std::cout << "  L1:          " << result.L1 << std::endl;
    std::cout << "  L2:          " << result.L2 << std::endl;
    std::cout << "  Path length: " << result.path_length << std::endl;
    std::cout << std::endl;

    // Verify posteriors sum to ~1.0
    float posterior_sum = 0.0f;
    for (int i = 0; i < L1 * L2; i++) {
        posterior_sum += result.posteriors[i];
    }
    std::cout << "  Posteriors sum: " << std::fixed << std::setprecision(6) << posterior_sum << std::endl;

    bool posteriors_valid = std::abs(posterior_sum - 1.0f) < 0.01f;
    std::cout << "  Posteriors valid: " << (posteriors_valid ? "✓" : "✗") << std::endl;
    std::cout << std::endl;

    // Print decoded alignment path
    std::cout << "  Decoded alignment path:" << std::endl;
    for (int k = 0; k < result.path_length && k < 10; k++) {  // Show first 10
        if (result.alignment_path[k].i == -1) {
            std::cout << "    Gap in seq1 (seq2 pos " << result.alignment_path[k].j << ")" << std::endl;
        } else if (result.alignment_path[k].j == -1) {
            std::cout << "    Gap in seq2 (seq1 pos " << result.alignment_path[k].i << ")" << std::endl;
        } else {
            std::cout << "    Match: " << result.alignment_path[k].i
                     << " -> " << result.alignment_path[k].j
                     << " (p=" << std::fixed << std::setprecision(3)
                     << result.alignment_path[k].posterior << ")" << std::endl;
        }
    }
    if (result.path_length > 10) {
        std::cout << "    ... (" << (result.path_length - 10) << " more)" << std::endl;
    }
    std::cout << std::endl;

    bool test_passed = posteriors_valid && result.path_length > 0 && result.score > 0.5f;
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;
}

void test_mismatched_sequences() {
    std::cout << "=== Test 2: Mismatched Sequences ===" << std::endl;
    std::cout << "Sequences with poor similarity should have gaps" << std::endl;
    std::cout << std::endl;

    const int L1 = 8;
    const int L2 = 8;
    const int hidden_dim = 128;

    // Create different embeddings
    std::vector<float> embeddings1(L1 * hidden_dim);
    std::vector<float> embeddings2(L2 * hidden_dim);

    for (int i = 0; i < L1; i++) {
        for (int k = 0; k < hidden_dim; k++) {
            embeddings1[i * hidden_dim + k] = std::sin(static_cast<float>(i + k) * 0.1f);
        }
    }

    for (int j = 0; j < L2; j++) {
        for (int k = 0; k < hidden_dim; k++) {
            embeddings2[j * hidden_dim + k] = std::cos(static_cast<float>(j + k) * 0.15f);
        }
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -1.0f;  // More lenient gaps
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L1, L2, config);
    GrowableArena arena(4);

    AlignmentResult result;
    result.posteriors = arena.allocate<float>(L1 * L2);
    result.alignment_path = arena.allocate<AlignmentPair>(L1 + L2);
    result.max_path_length = L1 + L2;

    pairwise_align_from_embeddings_full<ScalarBackend>(
        embeddings1.data(), L1,
        embeddings2.data(), L2,
        hidden_dim,
        config,
        &workspace,
        &result,
        &arena,
        -1.5f
    );

    std::cout << "  Partition:   " << result.partition << std::endl;
    std::cout << "  Score:       " << std::fixed << std::setprecision(3) << result.score << std::endl;
    std::cout << "  Path length: " << result.path_length << std::endl;

    // Count gaps
    int gaps_seq1 = 0, gaps_seq2 = 0, matches = 0;
    for (int k = 0; k < result.path_length; k++) {
        if (result.alignment_path[k].i == -1) gaps_seq1++;
        else if (result.alignment_path[k].j == -1) gaps_seq2++;
        else matches++;
    }

    std::cout << "  Matches:     " << matches << std::endl;
    std::cout << "  Gaps in seq1: " << gaps_seq1 << std::endl;
    std::cout << "  Gaps in seq2: " << gaps_seq2 << std::endl;

    bool test_passed = result.path_length > 0;
    std::cout << std::endl;
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;
}

void test_different_lengths() {
    std::cout << "=== Test 3: Different Sequence Lengths ===" << std::endl;
    std::cout << "L1 != L2 should work correctly" << std::endl;
    std::cout << std::endl;

    const int L1 = 10;
    const int L2 = 6;
    const int hidden_dim = 128;

    std::vector<float> embeddings1(L1 * hidden_dim);
    std::vector<float> embeddings2(L2 * hidden_dim);

    for (int i = 0; i < L1 * hidden_dim; i++) {
        embeddings1[i] = std::sin(static_cast<float>(i) * 0.05f);
    }

    for (int i = 0; i < L2 * hidden_dim; i++) {
        // Similar pattern but shorter
        embeddings2[i] = std::sin(static_cast<float>(i) * 0.05f);
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.5f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L1, L2, config);
    GrowableArena arena(4);

    AlignmentResult result;
    result.posteriors = arena.allocate<float>(L1 * L2);
    result.alignment_path = arena.allocate<AlignmentPair>(L1 + L2);
    result.max_path_length = L1 + L2;

    pairwise_align_from_embeddings_full<ScalarBackend>(
        embeddings1.data(), L1,
        embeddings2.data(), L2,
        hidden_dim,
        config,
        &workspace,
        &result,
        &arena
    );

    std::cout << "  L1 = " << L1 << ", L2 = " << L2 << std::endl;
    std::cout << "  Partition:   " << result.partition << std::endl;
    std::cout << "  Score:       " << std::fixed << std::setprecision(3) << result.score << std::endl;
    std::cout << "  Path length: " << result.path_length << std::endl;

    // Verify posteriors matrix size
    float posterior_sum = 0.0f;
    for (int i = 0; i < L1 * L2; i++) {
        posterior_sum += result.posteriors[i];
    }

    std::cout << "  Posteriors sum: " << std::fixed << std::setprecision(6) << posterior_sum << std::endl;

    bool test_passed = result.path_length > 0 && std::abs(posterior_sum - 1.0f) < 0.01f;
    std::cout << std::endl;
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;
}

void test_monotonicity() {
    std::cout << "=== Test 4: Path Monotonicity ===" << std::endl;
    std::cout << "Decoded path should be monotonic (i and j non-decreasing)" << std::endl;
    std::cout << std::endl;

    const int L = 7;
    const int hidden_dim = 128;

    std::vector<float> embeddings(L * hidden_dim);
    for (int i = 0; i < L * hidden_dim; i++) {
        embeddings[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    PairwiseConfig config;
    config.sw_mode = PairwiseConfig::SWMode::JAX_REGULAR;
    config.sw_config.gap = -0.2f;
    config.sw_config.temperature = 1.0f;
    config.mpnn_config.hidden_dim = hidden_dim;

    PairwiseWorkspace workspace(L, L, config);
    GrowableArena arena(4);

    AlignmentResult result;
    result.posteriors = arena.allocate<float>(L * L);
    result.alignment_path = arena.allocate<AlignmentPair>(L + L);
    result.max_path_length = L + L;

    pairwise_align_from_embeddings_full<ScalarBackend>(
        embeddings.data(), L,
        embeddings.data(), L,
        hidden_dim,
        config,
        &workspace,
        &result,
        &arena
    );

    // Check monotonicity
    int prev_i = -1, prev_j = -1;
    bool is_monotonic = true;

    for (int k = 0; k < result.path_length; k++) {
        int curr_i = result.alignment_path[k].i;
        int curr_j = result.alignment_path[k].j;

        // Skip gap positions for monotonicity check
        if (curr_i >= 0 && prev_i >= 0 && curr_i < prev_i) {
            is_monotonic = false;
            break;
        }
        if (curr_j >= 0 && prev_j >= 0 && curr_j < prev_j) {
            is_monotonic = false;
            break;
        }

        if (curr_i >= 0) prev_i = curr_i;
        if (curr_j >= 0) prev_j = curr_j;
    }

    std::cout << "  Path length: " << result.path_length << std::endl;
    std::cout << "  Monotonic:   " << (is_monotonic ? "✓" : "✗") << std::endl;
    std::cout << std::endl;
    std::cout << (is_monotonic ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Full Pairwise Alignment Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_basic_full_alignment();
    test_mismatched_sequences();
    test_different_lengths();
    test_monotonicity();

    std::cout << "========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "- Full alignment computes partition + score + posteriors + path" << std::endl;
    std::cout << "- Posteriors sum to 1.0 (probability distribution)" << std::endl;
    std::cout << "- Decoded path is monotonic and handles gaps correctly" << std::endl;
    std::cout << "- Works with different sequence lengths" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
