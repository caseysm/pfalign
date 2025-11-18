/**
 * Unit tests for MSA all-vs-all structural metrics (RMSD, TM-score).
 */

#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace pfalign;
using namespace pfalign::structural_metrics;

const float EPS = 1e-3f;  // Tolerance for floating point comparisons

/**
 * Generate ideal alpha-helix coordinates.
 */
void generate_helix(int L, float* ca_coords, float pitch = 1.5f, float radius = 2.3f) {
    for (int i = 0; i < L; i++) {
        float t = i * 100.0f * M_PI / 180.0f;  // 100 degrees per residue
        ca_coords[i*3 + 0] = radius * std::cos(t);
        ca_coords[i*3 + 1] = radius * std::sin(t);
        ca_coords[i*3 + 2] = pitch * i;
    }
}

/**
 * Generate ideal beta-strand coordinates.
 */
void generate_strand(int L, float* ca_coords, float spacing = 3.5f) {
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = spacing * i;
        ca_coords[i*3 + 1] = 0.0f;
        ca_coords[i*3 + 2] = 0.0f;
    }
}

bool test_rmsd_msa_selfmatch() {
    std::cout << "\n=== Test: RMSD MSA Self-Match ===\n";

    // Create 3 identical helices
    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), helix2(L * 3), helix3(L * 3);
    generate_helix(L, helix1.data());
    generate_helix(L, helix2.data());
    generate_helix(L, helix3.data());

    const float* ca_coords[] = {helix1.data(), helix2.data(), helix3.data()};

    // Perfect alignment (no gaps)
    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    // Compute RMSD
    float rmsd = rmsd_msa_allvsall<ScalarBackend>(ca_coords, col2pos, num_seqs, num_cols);

    std::cout << "  RMSD (3 identical helices): " << rmsd << " Å\n";

    // Should be ~0 (identical structures)
    if (rmsd < 1e-4f) {
        std::cout << "  ✓ PASS: RMSD ~= 0 for identical structures\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected RMSD ~= 0, got " << rmsd << "\n";
        return false;
    }
}

bool test_rmsd_msa_similar_helices() {
    std::cout << "\n=== Test: RMSD MSA Similar Helices ===\n";

    // Create 3 similar helices with slightly different parameters
    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), helix2(L * 3), helix3(L * 3);
    generate_helix(L, helix1.data(), 1.5f, 2.3f);
    generate_helix(L, helix2.data(), 1.6f, 2.4f);  // Slightly different
    generate_helix(L, helix3.data(), 1.7f, 2.5f);  // Even more different

    const float* ca_coords[] = {helix1.data(), helix2.data(), helix3.data()};

    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    float rmsd = rmsd_msa_allvsall<ScalarBackend>(ca_coords, col2pos, num_seqs, num_cols);

    std::cout << "  RMSD (3 similar helices): " << rmsd << " Å\n";

    // Should be small but non-zero (< 2Å typically)
    if (rmsd > 0.1f && rmsd < 2.0f) {
        std::cout << "  ✓ PASS: RMSD in expected range for similar structures\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected 0.1 < RMSD < 2.0, got " << rmsd << "\n";
        return false;
    }
}

bool test_rmsd_msa_different_folds() {
    std::cout << "\n=== Test: RMSD MSA Different Folds ===\n";

    // Mix of helices and strands
    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), strand1(L * 3), strand2(L * 3);
    generate_helix(L, helix1.data());
    generate_strand(L, strand1.data());
    generate_strand(L, strand2.data());

    const float* ca_coords[] = {helix1.data(), strand1.data(), strand2.data()};

    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    float rmsd = rmsd_msa_allvsall<ScalarBackend>(ca_coords, col2pos, num_seqs, num_cols);

    std::cout << "  RMSD (helix + 2 strands): " << rmsd << " Å\n";

    // Helix vs strand should have high RMSD (> 2Å)
    if (rmsd > 2.0f) {
        std::cout << "  ✓ PASS: High RMSD for different folds\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected RMSD > 2.0, got " << rmsd << "\n";
        return false;
    }
}

bool test_tm_score_msa_selfmatch() {
    std::cout << "\n=== Test: TM-score MSA Self-Match ===\n";

    // Create 3 identical helices
    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), helix2(L * 3), helix3(L * 3);
    generate_helix(L, helix1.data());
    generate_helix(L, helix2.data());
    generate_helix(L, helix3.data());

    const float* ca_coords[] = {helix1.data(), helix2.data(), helix3.data()};

    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    int seq_lengths[] = {L, L, L};

    float tm = tm_score_msa_allvsall<ScalarBackend>(
        ca_coords, col2pos, num_seqs, num_cols, seq_lengths
    );

    std::cout << "  TM-score (3 identical helices): " << tm << "\n";

    // Should be ~1.0 (perfect match)
    if (tm > 0.99f) {
        std::cout << "  ✓ PASS: TM-score ~= 1.0 for identical structures\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected TM ~= 1.0, got " << tm << "\n";
        return false;
    }
}

bool test_tm_score_msa_similar_helices() {
    std::cout << "\n=== Test: TM-score MSA Similar Helices ===\n";

    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), helix2(L * 3), helix3(L * 3);
    generate_helix(L, helix1.data(), 1.5f, 2.3f);
    generate_helix(L, helix2.data(), 1.6f, 2.4f);
    generate_helix(L, helix3.data(), 1.7f, 2.5f);

    const float* ca_coords[] = {helix1.data(), helix2.data(), helix3.data()};

    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    int seq_lengths[] = {L, L, L};

    float tm = tm_score_msa_allvsall<ScalarBackend>(
        ca_coords, col2pos, num_seqs, num_cols, seq_lengths
    );

    std::cout << "  TM-score (3 similar helices): " << tm << "\n";

    // Should be high (> 0.5 for same fold)
    if (tm > 0.5f && tm < 1.0f) {
        std::cout << "  ✓ PASS: TM-score > 0.5 for similar fold\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected 0.5 < TM < 1.0, got " << tm << "\n";
        return false;
    }
}

bool test_tm_score_msa_different_folds() {
    std::cout << "\n=== Test: TM-score MSA Different Folds ===\n";

    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), strand1(L * 3), strand2(L * 3);
    generate_helix(L, helix1.data());
    generate_strand(L, strand1.data());
    generate_strand(L, strand2.data());

    const float* ca_coords[] = {helix1.data(), strand1.data(), strand2.data()};

    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    int seq_lengths[] = {L, L, L};

    float tm = tm_score_msa_allvsall<ScalarBackend>(
        ca_coords, col2pos, num_seqs, num_cols, seq_lengths
    );

    std::cout << "  TM-score (helix + 2 strands): " << tm << "\n";

    // Helix vs strand pairs should have low TM (< 0.5)
    // But strand vs strand should have high TM
    // Average should be moderate
    if (tm > 0.0f && tm < 0.8f) {
        std::cout << "  ✓ PASS: TM-score in expected range\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Expected 0.0 < TM < 0.8, got " << tm << "\n";
        return false;
    }
}

bool test_msa_metrics_bundle() {
    std::cout << "\n=== Test: MSA Metrics Bundle ===\n";

    int L = 15;
    int num_seqs = 3;
    int num_cols = L;

    std::vector<float> helix1(L * 3), helix2(L * 3), helix3(L * 3);
    generate_helix(L, helix1.data(), 1.5f, 2.3f);
    generate_helix(L, helix2.data(), 1.6f, 2.4f);
    generate_helix(L, helix3.data(), 1.7f, 2.5f);

    const float* ca_coords[] = {helix1.data(), helix2.data(), helix3.data()};

    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        col2pos_1[i] = i;
        col2pos_2[i] = i;
        col2pos_3[i] = i;
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    int seq_lengths[] = {L, L, L};

    float avg_rmsd, avg_tm, avg_tm_sym;
    msa_metrics_allvsall<ScalarBackend>(
        ca_coords, col2pos, num_seqs, num_cols, seq_lengths,
        &avg_rmsd, &avg_tm, &avg_tm_sym
    );

    std::cout << "  Average RMSD: " << avg_rmsd << " Å\n";
    std::cout << "  Average TM (forward): " << avg_tm << "\n";
    std::cout << "  Average TM (symmetric): " << avg_tm_sym << "\n";

    // Sanity checks
    bool pass = true;
    if (avg_rmsd < 0.1f || avg_rmsd > 5.0f) {
        std::cout << "  ✗ FAIL: RMSD out of expected range\n";
        pass = false;
    }
    if (avg_tm < 0.3f || avg_tm > 1.0f) {
        std::cout << "  ✗ FAIL: TM-score out of expected range\n";
        pass = false;
    }
    if (avg_tm_sym < 0.3f || avg_tm_sym > 1.0f) {
        std::cout << "  ✗ FAIL: Symmetric TM-score out of expected range\n";
        pass = false;
    }

    if (pass) {
        std::cout << "  ✓ PASS: All metrics in expected ranges\n";
    }

    return pass;
}

bool test_rmsd_msa_with_gaps() {
    std::cout << "\n=== Test: RMSD MSA With Gaps ===\n";

    int L = 15;
    int num_seqs = 3;
    int num_cols = 20;  // More columns than residues (has gaps)

    std::vector<float> helix1(L * 3), helix2(L * 3), helix3(L * 3);
    generate_helix(L, helix1.data());
    generate_helix(L, helix2.data());
    generate_helix(L, helix3.data());

    const float* ca_coords[] = {helix1.data(), helix2.data(), helix3.data()};

    // Alignment with gaps
    std::vector<int> col2pos_1(num_cols), col2pos_2(num_cols), col2pos_3(num_cols);
    for (int i = 0; i < num_cols; i++) {
        if (i < L) {
            col2pos_1[i] = i;
            col2pos_2[i] = (i % 2 == 0) ? i / 2 : -1;  // Gaps every other column
            col2pos_3[i] = (i < L) ? i : -1;
        } else {
            col2pos_1[i] = -1;
            col2pos_2[i] = -1;
            col2pos_3[i] = -1;
        }
    }

    const int* col2pos[] = {col2pos_1.data(), col2pos_2.data(), col2pos_3.data()};

    float rmsd = rmsd_msa_allvsall<ScalarBackend>(ca_coords, col2pos, num_seqs, num_cols);

    std::cout << "  RMSD (with gaps): " << rmsd << " Å\n";

    // Should still compute reasonable RMSD for aligned regions
    if (std::isfinite(rmsd) && rmsd >= 0.0f) {
        std::cout << "  ✓ PASS: RMSD computed with gaps\n";
        return true;
    } else {
        std::cout << "  ✗ FAIL: Invalid RMSD with gaps\n";
        return false;
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  MSA Structural Metrics Tests\n";
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

    RUN_TEST(test_rmsd_msa_selfmatch);
    RUN_TEST(test_rmsd_msa_similar_helices);
    RUN_TEST(test_rmsd_msa_different_folds);
    RUN_TEST(test_tm_score_msa_selfmatch);
    RUN_TEST(test_tm_score_msa_similar_helices);
    RUN_TEST(test_tm_score_msa_different_folds);
    RUN_TEST(test_msa_metrics_bundle);
    RUN_TEST(test_rmsd_msa_with_gaps);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
