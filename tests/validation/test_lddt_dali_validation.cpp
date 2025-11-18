/**
 * Validation tests for LDDT and DALI implementations.
 *
 * These tests validate correctness using:
 * 1. Known protein structures with expected results
 * 2. Self-consistency checks
 * 3. Comparison against published values
 */

#include "pfalign/primitives/structural_metrics/lddt_impl.h"
#include "pfalign/primitives/structural_metrics/dali_impl.h"
#include "pfalign/primitives/structural_metrics/distance_matrix.h"
#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/common/perf_timer.h"
#include "../test_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <filesystem>

using namespace pfalign;
using pfalign::testing::GoldenDataTest;

const float EPS = 1e-3f;

// ============================================================================
// Test Data Generation
// ============================================================================

/**
 * Generate synthetic helix coordinates.
 */
void generate_helix(int L, float* ca_coords, float pitch = 1.5f, float radius = 2.3f) {
    for (int i = 0; i < L; i++) {
        float t = i * 100.0f * M_PI / 180.0f;  // 100deg per residue
        ca_coords[i*3 + 0] = radius * std::cos(t);
        ca_coords[i*3 + 1] = radius * std::sin(t);
        ca_coords[i*3 + 2] = pitch * i;
    }
}

/**
 * Generate synthetic beta strand coordinates.
 */
void generate_strand(int L, float* ca_coords, float spacing = 3.5f) {
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = spacing * i;
        ca_coords[i*3 + 1] = 0.0f;
        ca_coords[i*3 + 2] = 0.0f;
    }
}

void generate_random_structure(int L, float* ca_coords) {
    for (int i = 0; i < L; ++i) {
        float t = static_cast<float>(i);
        ca_coords[i * 3 + 0] = 5.0f * std::sin(0.5f * t) + 2.0f * std::cos(0.1f * t);
        ca_coords[i * 3 + 1] = 4.0f * std::cos(0.3f * t) + 1.5f * std::sin(0.07f * t);
        ca_coords[i * 3 + 2] = 3.0f * std::sin(0.17f * t) + 0.5f * t;
    }
}

void add_deterministic_noise(std::vector<float>& coords, float amplitude) {
    for (size_t idx = 0; idx < coords.size(); ++idx) {
        float t = static_cast<float>(idx);
        float noise = amplitude * std::sin(0.013f * t + 0.3f * static_cast<float>(idx % 3));
        coords[idx] += noise;
    }
}

// ============================================================================
// LDDT Validation Tests
// ============================================================================

bool validate_lddt_self_match() {
    std::cout << "\n=== Validate: LDDT Self-Match ===\n";

    int L = 20;
    std::vector<float> ca_coords(L * 3);
    generate_helix(L, ca_coords.data());

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Perfect self-alignment
    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, params
    );

    std::cout << "  LDDT (self-match): " << std::fixed << std::setprecision(4) << lddt << "\n";

    if (std::abs(lddt - 1.0f) < EPS) {
        std::cout << "  ✓ PASS: Self-match gives LDDT = 1.0\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected LDDT = 1.0, got " << lddt << "\n";
        return false;
    }
}

bool validate_lddt_conserved_structure() {
    std::cout << "\n=== Validate: LDDT Conserved Structure ===\n";

    // Two helices with identical geometry but different absolute positions
    int L = 15;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    generate_helix(L, ca1.data());

    // Second helix: same geometry, shifted in space
    generate_helix(L, ca2.data());
    for (int i = 0; i < L * 3; i++) {
        ca2[i] += 100.0f;  // Large translation
    }

    // Compute distance matrices
    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca1.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca2.data(), L, dist2.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, params
    );

    std::cout << "  LDDT (identical geometry, different position): " << lddt << "\n";

    // LDDT is superposition-free: identical internal geometry → LDDT = 1.0
    if (std::abs(lddt - 1.0f) < EPS) {
        std::cout << "  ✓ PASS: Translation-invariant (LDDT = 1.0)\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected LDDT = 1.0, got " << lddt << "\n";
        return false;
    }
}

bool validate_lddt_different_folds() {
    std::cout << "\n=== Validate: LDDT Different Folds ===\n";

    int L = 20;
    std::vector<float> ca_helix(L * 3), ca_strand(L * 3);

    generate_helix(L, ca_helix.data());
    generate_strand(L, ca_strand.data());

    std::vector<float> dist_helix(L * L), dist_strand(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_helix.data(), L, dist_helix.data()
    );
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_strand.data(), L, dist_strand.data()
    );

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_helix.data(), dist_strand.data(), alignment.data(), L, params
    );

    std::cout << "  LDDT (helix vs strand): " << lddt << "\n";

    // Different folds should have low LDDT
    if (lddt < 0.6f) {
        std::cout << "  ✓ PASS: Different folds have low LDDT\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected LDDT < 0.6, got " << lddt << "\n";
        return false;
    }
}

bool validate_lddt_symmetry_consistency() {
    std::cout << "\n=== Validate: LDDT Symmetry Mode Consistency ===\n";

    int L = 15;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    generate_helix(L, ca1.data(), 1.5f, 2.3f);
    generate_helix(L, ca2.data(), 1.8f, 2.8f);  // Different pitch and radius

    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca1.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca2.data(), L, dist2.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    structural_metrics::LDDTParams params_first, params_both, params_either;
    params_first.symmetry = "first";
    params_both.symmetry = "both";
    params_either.symmetry = "either";

    float lddt_first = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, params_first
    );
    float lddt_both = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, params_both
    );
    float lddt_either = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, params_either
    );

    std::cout << "  LDDT (first):  " << lddt_first << "\n";
    std::cout << "  LDDT (both):   " << lddt_both << "\n";
    std::cout << "  LDDT (either): " << lddt_either << "\n";

    // Verify ordering: either >= first (always holds due to OR condition)
    // Note: The relationship between 'first' and 'both' can vary:
    //   - If all d2 <= R0: both == first (same pairs considered)
    //   - If some d2 > R0 and those pairs have poor LDDT: both > first (excludes bad pairs)
    //   - If some d2 > R0 and those pairs have good LDDT: both < first (excludes good pairs)
    bool ordering_correct = (lddt_either >= lddt_first - EPS);

    if (ordering_correct) {
        std::cout << "  ✓ PASS: Symmetry modes behave correctly (either >= first)\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Symmetry mode ordering incorrect\n";
        return false;
    }
}

bool validate_lddt_noise_tolerance() {
    std::cout << "\n=== Validate: LDDT Noise Tolerance ===\n";

    int L = 60;
    std::vector<float> clean(L * 3), noisy(L * 3);
    generate_helix(L, clean.data());
    noisy = clean;
    add_deterministic_noise(noisy, 0.25f);

    std::vector<float> dist_clean(L * L), dist_noisy(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(clean.data(), L, dist_clean.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(noisy.data(), L, dist_noisy.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; ++i) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_clean.data(), dist_noisy.data(), alignment.data(), L
    );

    std::cout << "  LDDT (moderate noise): " << lddt << "\n";
    if (lddt > 0.9f) {
        std::cout << "  ✓ PASS: Small perturbations keep LDDT high\n";
        return true;
    }
    std::cout << "  ✗ FAIL: Expected LDDT > 0.9 under small perturbations\n";
    return false;
}

bool validate_lddt_radius_sensitivity() {
    std::cout << "\n=== Validate: LDDT Radius Sensitivity ===\n";

    int L = 40;
    std::vector<float> coords_ref(L * 3), coords_shifted(L * 3);
    generate_helix(L, coords_ref.data());
    coords_shifted = coords_ref;
    for (int i = L / 2; i < L; ++i) {
        coords_shifted[i * 3 + 0] += 25.0f;  // translate half the chain
    }

    std::vector<float> dist_ref(L * L), dist_shifted(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(coords_ref.data(), L, dist_ref.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(coords_shifted.data(), L, dist_shifted.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; ++i) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    structural_metrics::LDDTParams default_params;
    structural_metrics::LDDTParams tight_params;
    tight_params.R0 = 6.0f;

    float lddt_default = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_ref.data(), dist_shifted.data(), alignment.data(), L, default_params
    );
    float lddt_tight = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_ref.data(), dist_shifted.data(), alignment.data(), L, tight_params
    );

    std::cout << "  LDDT (R0=15Å): " << lddt_default << "\n";
    std::cout << "  LDDT (R0=6Å):  " << lddt_tight << "\n";

    if (lddt_tight > lddt_default + 0.05f) {
        std::cout << "  ✓ PASS: Smaller R0 mitigates distant domain shifts\n";
        return true;
    }
    std::cout << "  ✗ FAIL: Expected tighter R0 to produce noticeably higher score\n";
    return false;
}

bool validate_lddt_gap_robustness() {
    std::cout << "\n=== Validate: LDDT Gap Robustness ===\n";

    int L = 30;
    std::vector<float> coords(L * 3);
    generate_helix(L, coords.data());

    std::vector<float> dist(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(coords.data(), L, dist.data());

    std::vector<int> alignment_full(L * 2);
    std::vector<int> alignment_gappy;
    alignment_gappy.reserve(L * 2);

    for (int i = 0; i < L; ++i) {
        alignment_full[i*2 + 0] = i;
        alignment_full[i*2 + 1] = i;

        alignment_gappy.push_back(i);
        if (i % 5 == 0) {
            alignment_gappy.push_back(-1);  // simulate gap in second structure
        } else if (i % 7 == 0) {
            alignment_gappy.back() = i;
            alignment_gappy.push_back(-1);  // gap in first structure
        } else {
            alignment_gappy.push_back(i);
        }
    }

    float lddt_full = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist.data(), dist.data(), alignment_full.data(), L
    );
    float lddt_gappy = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist.data(), dist.data(), alignment_gappy.data(), static_cast<int>(alignment_gappy.size() / 2)
    );

    std::cout << "  LDDT (full alignment):  " << lddt_full << "\n";
    std::cout << "  LDDT (with gaps):       " << lddt_gappy << "\n";

    if (std::abs(lddt_full - lddt_gappy) < 1e-4f) {
        std::cout << "  ✓ PASS: Gaps are ignored as expected\n";
        return true;
    }
    std::cout << "  ✗ FAIL: Gap handling changed overall score\n";
    return false;
}

// ============================================================================
// DALI Validation Tests
// ============================================================================

bool validate_dali_self_match() {
    std::cout << "\n=== Validate: DALI Self-Match ===\n";

    // Use L=100 for meaningful Z-scores (formula is calibrated for L~50-400)
    int L = 100;
    std::vector<float> ca_coords(L * 3);
    generate_helix(L, ca_coords.data());

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

    std::cout << "  DALI score: " << std::fixed << std::setprecision(2) << result.score << "\n";
    std::cout << "  DALI Z:     " << result.Z << "\n";

    // Self-match should have very high Z-score (Z > 8 for L=100)
    // Note: Empirically, L=100 self-match gives Z ~ 8.7
    if (result.Z > 8.0f) {
        std::cout << "  ✓ PASS: Self-match has very high Z-score\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected Z > 8, got " << result.Z << "\n";
        return false;
    }
}

bool validate_dali_z_normalization() {
    std::cout << "\n=== Validate: DALI Z-Score Normalization ===\n";

    // Test that Z-scores are properly normalized across different lengths
    // Note: The Z-score formula accounts for length, so perfect self-matches
    // will have increasing Z-scores with length (this is expected behavior)
    auto compute_self_match_z = [](int L) {
        std::vector<float> ca_coords(L * 3);
        generate_helix(L, ca_coords.data());

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

        return result.Z;
    };

    float z50 = compute_self_match_z(50);
    float z100 = compute_self_match_z(100);
    float z200 = compute_self_match_z(200);

    std::cout << "  Z (L=50):  " << z50 << "\n";
    std::cout << "  Z (L=100): " << z100 << "\n";
    std::cout << "  Z (L=200): " << z200 << "\n";

    // All should be high (> 5), and increase moderately with length
    bool all_high = (z50 > 5.0f) && (z100 > 5.0f) && (z200 > 5.0f);
    bool increasing = (z200 > z100) && (z100 > z50);

    if (all_high && increasing) {
        std::cout << "  ✓ PASS: Z-scores properly normalized and increase with length\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Z-score normalization issue\n";
        return false;
    }
}

bool validate_dali_different_structures() {
    std::cout << "\n=== Validate: DALI Different Structures ===\n";

    int L = 20;
    std::vector<float> ca_helix(L * 3), ca_strand(L * 3);

    generate_helix(L, ca_helix.data());
    generate_strand(L, ca_strand.data());

    std::vector<float> dist_helix(L * L), dist_strand(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_helix.data(), L, dist_helix.data()
    );
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_strand.data(), L, dist_strand.data()
    );

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    structural_metrics::DALIParams params;
    auto result = structural_metrics::dali_score<ScalarBackend>(
        dist_helix.data(), dist_strand.data(), alignment.data(), L, L, L, params
    );

    std::cout << "  DALI Z (helix vs strand): " << result.Z << "\n";

    // Different structures should have low Z
    if (result.Z < 3.0f) {
        std::cout << "  ✓ PASS: Different structures have low Z\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected Z < 3.0, got " << result.Z << "\n";
        return false;
    }
}

bool validate_dali_horizon_effect() {
    std::cout << "\n=== Validate: DALI Horizon Parameter ===\n";

    int L = 20;
    std::vector<float> ca_coords(L * 3);
    generate_helix(L, ca_coords.data());

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    // Test three different horizon values
    structural_metrics::DALIParams params_small, params_default, params_large;
    params_small.horizon = 10.0f;
    params_default.horizon = 20.0f;
    params_large.horizon = 40.0f;

    auto r_small = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params_small
    );
    auto r_default = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params_default
    );
    auto r_large = structural_metrics::dali_score<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, L, L, params_large
    );

    std::cout << "  Score (horizon=10): " << r_small.score << "\n";
    std::cout << "  Score (horizon=20): " << r_default.score << "\n";
    std::cout << "  Score (horizon=40): " << r_large.score << "\n";

    // Larger horizon should give higher scores (weights decay slower)
    bool ordering_correct = (r_large.score > r_default.score) &&
                           (r_default.score > r_small.score);

    if (ordering_correct) {
        std::cout << "  ✓ PASS: Horizon affects scores correctly\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Horizon ordering incorrect\n";
        return false;
    }
}

// ============================================================================
// Metric Fixture Validation
// ============================================================================

struct MetricResult {
    float lddt = 0.0f;
    float dali_score = 0.0f;
    float dali_Z = 0.0f;
    float tm_norm_l1 = 0.0f;
    float tm_norm_l2 = 0.0f;
};

std::filesystem::path metrics_root() {
    return std::filesystem::path(pfalign::test::get_validation_path("metrics"));
}

void load_metric_arrays(
    const std::filesystem::path& dir,
    std::vector<float>& coords1,
    std::vector<size_t>& shape1,
    std::vector<float>& coords2,
    std::vector<size_t>& shape2,
    std::vector<float>& alignment,
    std::vector<size_t>& align_shape
) {
    GoldenDataTest test(dir.string());
    std::tie(coords1, shape1) = test.load_with_shape("coords1.npy");
    std::tie(coords2, shape2) = test.load_with_shape("coords2.npy");
    std::tie(alignment, align_shape) = test.load_with_shape("alignment.npy");
}

void apply_transform(
    const float* R,
    const float* t,
    const std::vector<float>& src,
    std::vector<float>& dst
) {
    const int N = static_cast<int>(src.size() / 3);
    dst.resize(src.size());
    for (int i = 0; i < N; ++i) {
        const float x = src[i * 3 + 0];
        const float y = src[i * 3 + 1];
        const float z = src[i * 3 + 2];
        dst[i * 3 + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
        dst[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
        dst[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
    }
}

MetricResult compute_metric_fixture(const std::filesystem::path& dir) {
    std::vector<float> coords1, coords2, alignment;
    std::vector<size_t> shape1, shape2, align_shape;
    load_metric_arrays(dir, coords1, shape1, coords2, shape2, alignment, align_shape);

    if (shape1.size() != 2 || shape2.size() != 2 || shape1[1] != 3 || shape2[1] != 3) {
        throw std::runtime_error("Metric fixture has invalid coordinate shape");
    }
    if (align_shape.size() != 2 || align_shape[1] != 2) {
        throw std::runtime_error("Metric fixture alignment has invalid shape");
    }

    const int L1 = static_cast<int>(shape1[0]);
    const int L2 = static_cast<int>(shape2[0]);
    std::vector<int> alignment_pairs;
    alignment_pairs.reserve(align_shape[0] * 2);
    std::vector<float> matched1;
    std::vector<float> matched2;

    for (size_t i = 0; i < align_shape[0]; ++i) {
        int a = static_cast<int>(alignment[i * 2 + 0]);
        int b = static_cast<int>(alignment[i * 2 + 1]);
        if (a < 0 || b < 0 || a >= L1 || b >= L2) {
            continue;
        }
        alignment_pairs.push_back(a);
        alignment_pairs.push_back(b);
        matched1.push_back(coords1[a * 3 + 0]);
        matched1.push_back(coords1[a * 3 + 1]);
        matched1.push_back(coords1[a * 3 + 2]);
        matched2.push_back(coords2[b * 3 + 0]);
        matched2.push_back(coords2[b * 3 + 1]);
        matched2.push_back(coords2[b * 3 + 2]);
    }

    const int aligned_length = static_cast<int>(alignment_pairs.size() / 2);
    if (aligned_length < 3) {
        throw std::runtime_error("Metric fixture requires at least 3 aligned residues");
    }

    float R[9];
    float t[3];
    float rmsd = 0.0f;
    pfalign::kabsch::kabsch_align<ScalarBackend>(
        matched1.data(),
        matched2.data(),
        aligned_length,
        R,
        t,
        &rmsd
    );

    std::vector<float> matched1_aligned;
    apply_transform(R, t, matched1, matched1_aligned);

    const float tm_l1 = structural_metrics::compute_tm_score<ScalarBackend>(
        matched1_aligned.data(),
        matched2.data(),
        aligned_length,
        L1
    );
    const float tm_l2 = structural_metrics::compute_tm_score<ScalarBackend>(
        matched1_aligned.data(),
        matched2.data(),
        aligned_length,
        L2
    );

    std::vector<float> dist1(L1 * L1, 0.0f);
    std::vector<float> dist2(L2 * L2, 0.0f);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        coords1.data(),
        L1,
        dist1.data()
    );
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        coords2.data(),
        L2,
        dist2.data()
    );

    structural_metrics::LDDTParams lddt_params;
    const float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(),
        dist2.data(),
        alignment_pairs.data(),
        aligned_length,
        lddt_params,
        nullptr
    );

    structural_metrics::DALIParams dali_params;
    auto dali = structural_metrics::dali_score<ScalarBackend>(
        dist1.data(),
        dist2.data(),
        alignment_pairs.data(),
        aligned_length,
        L1,
        L2,
        dali_params
    );

    return MetricResult{
        lddt,
        dali.score,
        dali.Z,
        tm_l1,
        tm_l2
    };
}

bool load_expected_metrics(const std::filesystem::path& file, MetricResult& expected) {
    if (!std::filesystem::exists(file)) {
        return false;
    }

    std::ifstream input(file);
    if (!input) {
        return false;
    }

    std::map<std::string, float> values;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, pos);
        const std::string value_str = line.substr(pos + 1);
        values[key] = std::stof(value_str);
    }

    if (values.size() < 5) {
        return false;
    }

    expected.lddt = values["lddt"];
    expected.dali_score = values["dali_score"];
    expected.dali_Z = values["dali_z"];
    expected.tm_norm_l1 = values["tm_norm_l1"];
    expected.tm_norm_l2 = values["tm_norm_l2"];
    return true;
}

void save_expected_metrics(const std::filesystem::path& file, const MetricResult& result) {
    std::ofstream output(file);
    output << std::setprecision(10)
           << "lddt=" << result.lddt << "\n"
           << "dali_score=" << result.dali_score << "\n"
           << "dali_z=" << result.dali_Z << "\n"
           << "tm_norm_l1=" << result.tm_norm_l1 << "\n"
           << "tm_norm_l2=" << result.tm_norm_l2 << "\n";
}

bool run_metric_fixture(
    const std::filesystem::path& dir,
    bool rewrite_metrics
) {
    MetricResult result = compute_metric_fixture(dir);
    const auto metrics_file = dir / "metrics.txt";

    if (rewrite_metrics) {
        save_expected_metrics(metrics_file, result);
        std::cout << "  ✎ Updated " << metrics_file << std::endl;
        return true;
    }

    MetricResult expected;
    if (!load_expected_metrics(metrics_file, expected)) {
        std::cout << "  ✗ Missing metrics.txt (run with --rewrite-metrics)" << std::endl;
        return false;
    }

    auto diff_ok = [](float a, float b) {
        return std::fabs(a - b) <= 5e-4f;
    };

    const bool ok =
        diff_ok(result.lddt, expected.lddt) &&
        diff_ok(result.dali_score, expected.dali_score) &&
        diff_ok(result.dali_Z, expected.dali_Z) &&
        diff_ok(result.tm_norm_l1, expected.tm_norm_l1) &&
        diff_ok(result.tm_norm_l2, expected.tm_norm_l2);

    std::cout << "  LDDT diff: " << std::fabs(result.lddt - expected.lddt) << "\n";
    std::cout << "  DALI score diff: " << std::fabs(result.dali_score - expected.dali_score) << "\n";
    std::cout << "  DALI Z diff: " << std::fabs(result.dali_Z - expected.dali_Z) << "\n";
    std::cout << "  TM(L1) diff: " << std::fabs(result.tm_norm_l1 - expected.tm_norm_l1) << "\n";
    std::cout << "  TM(L2) diff: " << std::fabs(result.tm_norm_l2 - expected.tm_norm_l2) << "\n";

    std::cout << (ok ? "  ✓ PASS" : "  ✗ FAIL") << std::endl;
    return ok;
}

bool run_metric_fixture_suite(bool rewrite_metrics) {
    const auto root = metrics_root();
    if (!std::filesystem::exists(root)) {
        std::cout << "SKIP: No metric fixtures at " << root << std::endl;
        return true;
    }

    bool all_passed = true;
    bool any_fixture = false;

    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        any_fixture = true;
        std::cout << "\n=== Metric Fixture: " << entry.path().filename().string() << " ===\n";
        if (!run_metric_fixture(entry.path(), rewrite_metrics)) {
            all_passed = false;
        }
    }

    if (!any_fixture) {
        std::cout << "SKIP: Metric fixture directory is empty\n";
        return true;
    }
    return all_passed;
}

bool validate_dali_symmetry() {
    std::cout << "\n=== Validate: DALI Symmetry ===\n";

    int L = 60;
    std::vector<float> coords1(L * 3), coords2(L * 3);
    generate_helix(L, coords1.data());
    coords2 = coords1;
    for (int i = 0; i < L; ++i) {
        coords2[i * 3 + 0] += 40.0f;  // rigid-body translate
    }

    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(coords1.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(coords2.data(), L, dist2.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; ++i) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    auto forward = structural_metrics::dali_score<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, L, L
    );
    auto reverse = structural_metrics::dali_score<ScalarBackend>(
        dist2.data(), dist1.data(), alignment.data(), L, L, L
    );

    std::cout << "  Forward score/Z: " << forward.score << " / " << forward.Z << "\n";
    std::cout << "  Reverse score/Z: " << reverse.score << " / " << reverse.Z << "\n";

    if (std::abs(forward.score - reverse.score) < 1e-4f &&
        std::abs(forward.Z - reverse.Z) < 1e-4f) {
        std::cout << "  ✓ PASS: DALI scores are symmetric\n";
        return true;
    }
    std::cout << "  ✗ FAIL: Score/Z mismatch when swapping inputs\n";
    return false;
}

bool validate_dali_alignment_length_effect() {
    std::cout << "\n=== Validate: DALI Alignment Length Effect ===\n";

    int L = 80;
    std::vector<float> coords(L * 3);
    generate_helix(L, coords.data());

    std::vector<float> dist(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(coords.data(), L, dist.data());

    std::vector<int> alignment_full(L * 2);
    for (int i = 0; i < L; ++i) {
        alignment_full[i*2 + 0] = i;
        alignment_full[i*2 + 1] = i;
    }

    std::vector<int> alignment_partial;
    alignment_partial.reserve(60);
    for (int i = 0; i < 30; ++i) {
        alignment_partial.push_back(i);
        alignment_partial.push_back(i);
    }

    auto result_full = structural_metrics::dali_score<ScalarBackend>(
        dist.data(), dist.data(), alignment_full.data(), L, L, L
    );
    auto result_partial = structural_metrics::dali_score<ScalarBackend>(
        dist.data(), dist.data(), alignment_partial.data(), 30, L, L
    );

    std::cout << "  Z (full length):    " << result_full.Z << "\n";
    std::cout << "  Z (30 residues):    " << result_partial.Z << "\n";

    if (result_full.Z > result_partial.Z + 2.0f) {
        std::cout << "  ✓ PASS: Longer alignments boost Z-score as expected\n";
        return true;
    }
    std::cout << "  ✗ FAIL: Z-score did not improve appreciably with length\n";
    return false;
}

bool validate_dali_random_baseline() {
    std::cout << "\n=== Validate: DALI Random Baseline ===\n";

    int L = 60;
    std::vector<float> helix(L * 3), random_coords(L * 3);
    generate_helix(L, helix.data());
    generate_random_structure(L, random_coords.data());

    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(helix.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(random_coords.data(), L, dist2.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; ++i) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    auto result = structural_metrics::dali_score<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, L, L
    );

    std::cout << "  DALI Z (helix vs random): " << result.Z << "\n";
    if (result.Z < 2.0f) {
        std::cout << "  ✓ PASS: Random alignments stay below significance threshold\n";
        return true;
    }
    std::cout << "  ✗ FAIL: Random structures produced suspiciously high Z-score\n";
    return false;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("test_lddt_dali_validation");
    bool rewrite_metrics = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rewrite-metrics") {
            rewrite_metrics = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    std::cout << "========================================\n";
    std::cout << "  LDDT & DALI Validation Tests\n";
    std::cout << "========================================\n";

    int passed = 0;
    int total = 0;

    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) { \
            passed++; \
        }

    // LDDT tests
    RUN_TEST(validate_lddt_self_match);
    RUN_TEST(validate_lddt_conserved_structure);
    RUN_TEST(validate_lddt_different_folds);
    RUN_TEST(validate_lddt_symmetry_consistency);
    RUN_TEST(validate_lddt_noise_tolerance);
    RUN_TEST(validate_lddt_radius_sensitivity);
    RUN_TEST(validate_lddt_gap_robustness);

    // DALI tests
    RUN_TEST(validate_dali_self_match);
    RUN_TEST(validate_dali_z_normalization);
    RUN_TEST(validate_dali_different_structures);
    RUN_TEST(validate_dali_horizon_effect);
    RUN_TEST(validate_dali_symmetry);
    RUN_TEST(validate_dali_alignment_length_effect);
    RUN_TEST(validate_dali_random_baseline);

    total++;
    if (run_metric_fixture_suite(rewrite_metrics)) {
        passed++;
    }

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
