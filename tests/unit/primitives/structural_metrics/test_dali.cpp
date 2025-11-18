/**
 * Unit tests for DALI scoring.
 */

#include "pfalign/primitives/structural_metrics/dali_impl.h"
#include "pfalign/primitives/structural_metrics/distance_matrix.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace pfalign;

const float EPS = 1e-2f;  // DALI uses looser tolerance

bool test_dali_z_score_formula() {
    std::cout << "\n=== Test: DALI Z-Score Formula ===\n";

    // Test known values from paper
    // For L1=L2=100: n12 = 100, expect mean around 78
    float Z1 = structural_metrics::dali_Z_from_score_and_lengths(100.0f, 100, 100);
    std::cout << "  Z(score=100, L1=100, L2=100) = " << Z1 << "\n";

    // For L1=L2=200: n12 = 200, expect mean around 176
    float Z2 = structural_metrics::dali_Z_from_score_and_lengths(200.0f, 200, 200);
    std::cout << "  Z(score=200, L1=200, L2=200) = " << Z2 << "\n";

    // Z-scores should be reasonable (not NaN, not infinite)
    if (std::isfinite(Z1) && std::isfinite(Z2)) {
        std::cout << "  ✓ Z-scores are finite\n";
        return true;
    } else {
        std::cout << "  ✗ Z-scores are not finite\n";
        return false;
    }
}

bool test_dali_perfect_match() {
    std::cout << "\n=== Test: DALI Perfect Match ===\n";

    // Two identical structures (use L=50 for meaningful Z-scores with calibrated formula)
    int L = 50;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i);
        ca_coords[i*3 + 1] = static_cast<float>(i * 1.5);
        ca_coords[i*3 + 2] = static_cast<float>(i * 0.8);
    }

    // Compute distance matrix
    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Perfect alignment
    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    // Compute DALI
    structural_metrics::DALIParams params;
    auto result = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params
    );

    std::cout << "  DALI score: " << result.score << "\n";
    std::cout << "  DALI Z-score: " << result.Z << "\n";

    // Perfect self-match should have very high Z-score (Z-score formula is calibrated for L~50-400)
    // For L=50, perfect match should give Z > 5 (high confidence)
    if (result.Z > 5.0f) {
        std::cout << "  ✓ High Z-score for perfect match (Z > 5)\n";
        return true;
    } else {
        std::cout << "  ✗ Expected Z > 5 for perfect match (got " << result.Z << ")\n";
        return false;
    }
}

bool test_dali_similar_structures() {
    std::cout << "\n=== Test: DALI Similar Structures ===\n";

    // Two similar structures (use L=50 for meaningful Z-scores with calibrated formula)
    int L = 50;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    for (int i = 0; i < L; i++) {
        // Structure 1: helix
        float t = i * 100.0f * M_PI / 180.0f;
        ca1[i*3 + 0] = 2.3f * std::cos(t);
        ca1[i*3 + 1] = 2.3f * std::sin(t);
        ca1[i*3 + 2] = 1.5f * i;

        // Structure 2: similar helix with small distortion
        ca2[i*3 + 0] = ca1[i*3 + 0] + 0.3f * std::sin(i);
        ca2[i*3 + 1] = ca1[i*3 + 1] + 0.3f * std::cos(i);
        ca2[i*3 + 2] = ca1[i*3 + 2] + 0.2f;
    }

    // Compute distance matrices
    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca1.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca2.data(), L, dist2.data());

    // Perfect alignment
    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    // Compute DALI
    structural_metrics::DALIParams params;
    auto result = structural_metrics::dali_score<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, L, L, params
    );

    std::cout << "  DALI score: " << result.score << "\n";
    std::cout << "  DALI Z-score: " << result.Z << "\n";

    // Similar structures with small distortions should have good Z-score
    // For L=50, expect Z > 3 (moderate to high confidence)
    if (result.Z > 3.0f) {
        std::cout << "  ✓ Good Z-score for similar structures (Z > 3)\n";
        return true;
    } else {
        std::cout << "  ✗ Expected Z > 3 for similar structures (got " << result.Z << ")\n";
        return false;
    }
}

bool test_dali_different_structures() {
    std::cout << "\n=== Test: DALI Different Structures ===\n";

    // Two different structures
    int L = 10;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    // Structure 1: linear
    for (int i = 0; i < L; i++) {
        ca1[i*3 + 0] = static_cast<float>(i * 2);
        ca1[i*3 + 1] = 0.0f;
        ca1[i*3 + 2] = 0.0f;
    }

    // Structure 2: compact sphere
    for (int i = 0; i < L; i++) {
        float theta = 2.0f * M_PI * i / L;
        float phi = M_PI * i / (2 * L);
        ca2[i*3 + 0] = 3.0f * std::sin(phi) * std::cos(theta);
        ca2[i*3 + 1] = 3.0f * std::sin(phi) * std::sin(theta);
        ca2[i*3 + 2] = 3.0f * std::cos(phi);
    }

    // Compute distance matrices
    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca1.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca2.data(), L, dist2.data());

    // Perfect alignment
    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    // Compute DALI
    structural_metrics::DALIParams params;
    auto result = structural_metrics::dali_score<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, L, L, params
    );

    std::cout << "  DALI score: " << result.score << "\n";
    std::cout << "  DALI Z-score: " << result.Z << "\n";

    // Different structures should have low Z-score (<2)
    if (result.Z < 2.0f) {
        std::cout << "  ✓ Low Z-score for different structures\n";
        return true;
    } else {
        std::cout << "  ✗ Expected Z < 2\n";
        return false;
    }
}

bool test_dali_with_gaps() {
    std::cout << "\n=== Test: DALI With Gaps ===\n";

    int L = 50;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i);
        ca_coords[i*3 + 1] = static_cast<float>(i);
        ca_coords[i*3 + 2] = 0.0f;
    }

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Alignment with 10% gaps
    std::vector<int> alignment;
    for (int i = 0; i < L; i++) {
        if (i % 10 == 2) {
            alignment.push_back(i);
            alignment.push_back(-1);  // Gap
        } else if (i % 10 == 3) {
            alignment.push_back(-1);  // Gap
            alignment.push_back(i);
        } else {
            alignment.push_back(i);
            alignment.push_back(i);
        }
    }
    int aligned_length = alignment.size() / 2;

    // Compute DALI
    structural_metrics::DALIParams params;
    auto result = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), aligned_length, L, L, params
    );

    std::cout << "  DALI score with gaps: " << result.score << "\n";
    std::cout << "  DALI Z-score: " << result.Z << "\n";

    // Should still have good Z for mostly-identical structures despite gaps
    // With 10% gaps, expect Z > 4 (still high confidence)
    if (result.Z > 4.0f) {
        std::cout << "  ✓ DALI handles gaps correctly (Z > 4)\n";
        return true;
    } else {
        std::cout << "  ✗ Expected Z > 4 despite gaps (got " << result.Z << ")\n";
        return false;
    }
}

bool test_dali_horizon_parameter() {
    std::cout << "\n=== Test: DALI Horizon Parameter ===\n";

    int L = 10;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i * 3);
        ca_coords[i*3 + 1] = 0.0f;
        ca_coords[i*3 + 2] = 0.0f;
    }

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    // Test different horizon values
    structural_metrics::DALIParams params_10, params_20, params_40;
    params_10.horizon = 10.0f;
    params_20.horizon = 20.0f;
    params_40.horizon = 40.0f;

    auto result_10 = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params_10
    );
    auto result_20 = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params_20
    );
    auto result_40 = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params_40
    );

    std::cout << "  DALI (horizon=10): " << result_10.score << " (Z=" << result_10.Z << ")\n";
    std::cout << "  DALI (horizon=20): " << result_20.score << " (Z=" << result_20.Z << ")\n";
    std::cout << "  DALI (horizon=40): " << result_40.score << " (Z=" << result_40.Z << ")\n";

    // Larger horizon should give higher scores (weights decay slower)
    if (result_40.score > result_20.score && result_20.score > result_10.score) {
        std::cout << "  ✓ Horizon parameter affects scores as expected\n";
        return true;
    } else {
        std::cout << "  ✗ Horizon parameter behavior unexpected\n";
        return false;
    }
}

bool test_dali_length_normalization() {
    std::cout << "\n=== Test: DALI Length Normalization ===\n";

    // Test that Z-score properly normalizes for length
    // Similar structures of different lengths should have comparable Z-scores

    auto test_length = [](int L) {
        std::vector<float> ca_coords(L * 3);
        for (int i = 0; i < L; i++) {
            ca_coords[i*3 + 0] = static_cast<float>(i);
            ca_coords[i*3 + 1] = static_cast<float>(i * 0.5);
            ca_coords[i*3 + 2] = 0.0f;
        }

        std::vector<float> dist_mx(L * L);
        structural_metrics::compute_distance_matrix<ScalarBackend>(
            ca_coords.data(), L, dist_mx.data()
        );

        std::vector<int> alignment(L * 2);
        for (int i = 0; i < L; i++) {
            alignment[i*2 + 0] = i;
            alignment[i*2 + 1] = i;
        }

        structural_metrics::DALIParams params;
        auto result = structural_metrics::dali_score<ScalarBackend>(
            dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params
        );

        return result;
    };

    auto result_50 = test_length(50);
    auto result_100 = test_length(100);
    auto result_200 = test_length(200);

    std::cout << "  L=50:  score=" << result_50.score << ", Z=" << result_50.Z << "\n";
    std::cout << "  L=100: score=" << result_100.score << ", Z=" << result_100.Z << "\n";
    std::cout << "  L=200: score=" << result_200.score << ", Z=" << result_200.Z << "\n";

    // All should have high Z-scores (perfect self-match)
    // Z-scores should increase moderately with length (formula accounts for this)
    bool all_high = (result_50.Z > 5.0f) && (result_100.Z > 5.0f) && (result_200.Z > 5.0f);
    bool increasing = (result_200.Z > result_100.Z) && (result_100.Z > result_50.Z);

    if (all_high && increasing) {
        std::cout << "  ✓ Z-scores properly normalized and increase with length\n";
        return true;
    } else {
        std::cout << "  ✗ Z-score normalization issue\n";
        return false;
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  DALI Scoring Tests\n";
    std::cout << "========================================\n";

    int passed = 0;
    int total = 0;

    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) { \
            passed++; \
            std::cout << "  PASS\n"; \
        } else { \
            std::cout << "  FAIL\n"; \
        }

    RUN_TEST(test_dali_z_score_formula);
    RUN_TEST(test_dali_perfect_match);
    RUN_TEST(test_dali_similar_structures);
    RUN_TEST(test_dali_different_structures);
    RUN_TEST(test_dali_with_gaps);
    RUN_TEST(test_dali_horizon_parameter);
    RUN_TEST(test_dali_length_normalization);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
