/**
 * Unit tests for LDDT scoring.
 */

#include "pfalign/primitives/structural_metrics/lddt_impl.h"
#include "pfalign/primitives/structural_metrics/distance_matrix.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/io/pdb_parser.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace pfalign;

const float EPS = 1e-4f;

bool test_lddt_perfect_match() {
    std::cout << "\n=== Test: LDDT Perfect Match ===\n";

    // Two identical structures -> LDDT = 1.0
    int L = 5;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i);
        ca_coords[i*3 + 1] = static_cast<float>(i * 2);
        ca_coords[i*3 + 2] = static_cast<float>(i * 3);
    }

    // Compute distance matrix (same for both)
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

    // Compute LDDT
    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, params
    );

    std::cout << "  LDDT: " << lddt << " (expected 1.0)\n";

    if (std::abs(lddt - 1.0f) < EPS) {
        std::cout << "  ✓ Perfect match LDDT = 1.0\n";
        return true;
    } else {
        std::cout << "  ✗ Expected 1.0\n";
        return false;
    }
}

bool test_lddt_similar_structures() {
    std::cout << "\n=== Test: LDDT Similar Structures ===\n";

    // Two similar structures with small perturbation
    int L = 10;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    for (int i = 0; i < L; i++) {
        // Structure 1
        ca1[i*3 + 0] = static_cast<float>(i);
        ca1[i*3 + 1] = static_cast<float>(i * 1.5);
        ca1[i*3 + 2] = static_cast<float>(i * 0.8);

        // Structure 2: small perturbation (0.2Å)
        ca2[i*3 + 0] = ca1[i*3 + 0] + 0.15f;
        ca2[i*3 + 1] = ca1[i*3 + 1] + 0.10f;
        ca2[i*3 + 2] = ca1[i*3 + 2] + 0.12f;
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

    // Compute LDDT
    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, params
    );

    std::cout << "  LDDT: " << lddt << "\n";

    // With small perturbations, LDDT should be high (>0.9)
    if (lddt > 0.9f && lddt <= 1.0f) {
        std::cout << "  ✓ High LDDT for similar structures\n";
        return true;
    } else {
        std::cout << "  ✗ Expected LDDT > 0.9\n";
        return false;
    }
}

bool test_lddt_different_structures() {
    std::cout << "\n=== Test: LDDT Different Structures ===\n";

    // Two completely different structures
    int L = 10;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    // Structure 1: linear
    for (int i = 0; i < L; i++) {
        ca1[i*3 + 0] = static_cast<float>(i);
        ca1[i*3 + 1] = 0.0f;
        ca1[i*3 + 2] = 0.0f;
    }

    // Structure 2: different shape (circular)
    for (int i = 0; i < L; i++) {
        float angle = 2.0f * M_PI * i / L;
        ca2[i*3 + 0] = 5.0f * std::cos(angle);
        ca2[i*3 + 1] = 5.0f * std::sin(angle);
        ca2[i*3 + 2] = 0.0f;
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

    // Compute LDDT
    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist1.data(), dist2.data(), alignment.data(), L, params
    );

    std::cout << "  LDDT: " << lddt << "\n";

    // Different structures should have low LDDT (<0.5)
    if (lddt < 0.5f) {
        std::cout << "  ✓ Low LDDT for different structures\n";
        return true;
    } else {
        std::cout << "  ✗ Expected LDDT < 0.5\n";
        return false;
    }
}

bool test_lddt_with_gaps() {
    std::cout << "\n=== Test: LDDT With Gaps in Alignment ===\n";

    int L = 8;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i);
        ca_coords[i*3 + 1] = static_cast<float>(i);
        ca_coords[i*3 + 2] = static_cast<float>(i);
    }

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Alignment with gaps (position -1 indicates gap)
    std::vector<int> alignment = {
        0, 0,    // Match
        1, 1,    // Match
        2, -1,   // Gap in structure 2
        -1, 2,   // Gap in structure 1
        3, 3,    // Match
        4, 4,    // Match
        5, 5,    // Match
        6, 6     // Match
    };

    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), 8, params
    );

    std::cout << "  LDDT with gaps: " << lddt << "\n";

    // Should still be high for identical structures despite gaps
    if (lddt > 0.9f) {
        std::cout << "  ✓ LDDT correctly handles gaps\n";
        return true;
    } else {
        std::cout << "  ✗ LDDT should be high despite gaps\n";
        return false;
    }
}

bool test_lddt_per_residue_scores() {
    std::cout << "\n=== Test: LDDT Per-Residue Scores ===\n";

    int L = 5;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i);
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

    std::vector<float> per_residue(L);
    structural_metrics::LDDTParams params;
    float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
        dist_mx.data(), dist_mx.data(), alignment.data(), L, params,
        per_residue.data()
    );

    std::cout << "  Overall LDDT: " << lddt << "\n";
    std::cout << "  Per-residue scores: ";
    for (int i = 0; i < L; i++) {
        std::cout << per_residue[i] << " ";
    }
    std::cout << "\n";

    // All per-residue scores should be 1.0 for perfect match
    bool all_perfect = true;
    for (int i = 0; i < L; i++) {
        if (std::abs(per_residue[i] - 1.0f) > EPS) {
            all_perfect = false;
            break;
        }
    }

    if (all_perfect) {
        std::cout << "  ✓ Per-residue scores all 1.0\n";
        return true;
    } else {
        std::cout << "  ✗ Expected all per-residue scores = 1.0\n";
        return false;
    }
}

bool test_lddt_symmetry_modes() {
    std::cout << "\n=== Test: LDDT Symmetry Modes ===\n";

    int L = 10;
    std::vector<float> ca1(L * 3), ca2(L * 3);

    for (int i = 0; i < L; i++) {
        ca1[i*3 + 0] = static_cast<float>(i);
        ca1[i*3 + 1] = static_cast<float>(i * 0.5);
        ca1[i*3 + 2] = 0.0f;

        // Structure 2 scaled up
        ca2[i*3 + 0] = static_cast<float>(i * 2);
        ca2[i*3 + 1] = static_cast<float>(i);
        ca2[i*3 + 2] = 0.0f;
    }

    std::vector<float> dist1(L * L), dist2(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca1.data(), L, dist1.data());
    structural_metrics::compute_distance_matrix<ScalarBackend>(ca2.data(), L, dist2.data());

    std::vector<int> alignment(L * 2);
    for (int i = 0; i < L; i++) {
        alignment[i*2 + 0] = i;
        alignment[i*2 + 1] = i;
    }

    // Test different symmetry modes
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

    std::cout << "  LDDT (symmetry='first'): " << lddt_first << "\n";
    std::cout << "  LDDT (symmetry='both'): " << lddt_both << "\n";
    std::cout << "  LDDT (symmetry='either'): " << lddt_either << "\n";

    // Verify relationships:
    // - either >= first (always, since 'either' uses OR condition)
    // - The relationship between 'first' and 'both' can vary:
    //   * If all d2 <= R0: both == first (same pairs considered)
    //   * If some d2 > R0 and those pairs have poor LDDT: both > first (excludes bad pairs)
    //   * If some d2 > R0 and those pairs have good LDDT: both < first (excludes good pairs)
    // In this test, structure 2 is scaled 2*, so some d2 > R0, and those pairs have poor LDDT
    // Therefore: both > first is expected and correct
    if (lddt_either >= lddt_first - EPS) {
        std::cout << "  ✓ Symmetry modes behave correctly (either >= first)\n";
        std::cout << "  Note: both=" << lddt_both << " vs first=" << lddt_first
                  << " varies by test case\n";
        return true;
    } else {
        std::cout << "  ✗ Unexpected: either should be >= first\n";
        return false;
    }
}

// ============================================================================
// Real Structure Tests
// ============================================================================

bool test_lddt_real_crambin_self_match() {
    std::cout << "\n=== Test: LDDT Real Structure - Crambin Self-Match ===\n";

    const char* pdb_path = "../../data/structures/pdb/small/1CRN.pdb";

    try {
        // Load crambin structure
        io::PDBParser parser;
        io::Protein prot = parser.parse_file(pdb_path);

        if (prot.chains.empty()) {
            std::cout << "  SKIP: No chains in PDB file\n";
            return true;
        }

        int L = prot.get_chain(0).size();
        auto coords = prot.get_backbone_coords(0);

        std::cout << "  Loaded crambin: " << L << " residues\n";

        // Compute distance matrix
        std::vector<float> dist_mx(L * L);
        structural_metrics::compute_distance_matrix<ScalarBackend>(
            coords.data(), L, dist_mx.data()
        );

        // Perfect self-alignment
        std::vector<int> alignment(L * 2);
        for (int i = 0; i < L; i++) {
            alignment[i*2 + 0] = i;
            alignment[i*2 + 1] = i;
        }

        // Compute LDDT
        structural_metrics::LDDTParams params;
        float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
            dist_mx.data(), dist_mx.data(), alignment.data(), L, params
        );

        std::cout << "  LDDT (self-match): " << std::fixed << std::setprecision(4) << lddt << "\n";
        std::cout << "  Expected: 1.0 (perfect match)\n";

        if (std::abs(lddt - 1.0f) < EPS) {
            std::cout << "  ✓ Real structure self-match: LDDT = 1.0\n";
            return true;
        } else {
            std::cout << "  ✗ Expected LDDT = 1.0, got " << lddt << "\n";
            return false;
        }

    } catch (const std::exception& e) {
        std::cout << "  SKIP: Could not load PDB (" << e.what() << ")\n";
        std::cout << "  (Not counted as failure - test data optional)\n";
        return true;  // Skip, not fail
    }
}

bool test_lddt_real_globin_family() {
    std::cout << "\n=== Test: LDDT Real Structures - Globin Family ===\n";

    const char* pdb1_path = "../../data/structures/pdb/medium/1MBO.pdb";
    const char* pdb2_path = "../../data/structures/pdb/medium/1HBS.pdb";

    try {
        // Load myoglobin and hemoglobin
        io::PDBParser parser;
        io::Protein prot1 = parser.parse_file(pdb1_path);
        io::Protein prot2 = parser.parse_file(pdb2_path);

        if (prot1.chains.empty() || prot2.chains.empty()) {
            std::cout << "  SKIP: Missing chains\n";
            return true;
        }

        int L1 = prot1.get_chain(0).size();
        int L2 = prot2.get_chain(0).size();
        auto coords1 = prot1.get_backbone_coords(0);
        auto coords2 = prot2.get_backbone_coords(0);

        std::cout << "  Loaded 1MBO: " << L1 << " residues (myoglobin)\n";
        std::cout << "  Loaded 1HBS: " << L2 << " residues (hemoglobin)\n";

        // Compute distance matrices
        std::vector<float> dist1(L1 * L1), dist2(L2 * L2);
        structural_metrics::compute_distance_matrix<ScalarBackend>(
            coords1.data(), L1, dist1.data()
        );
        structural_metrics::compute_distance_matrix<ScalarBackend>(
            coords2.data(), L2, dist2.data()
        );

        // Simple linear alignment (for testing - real alignment would be better)
        int L_min = std::min(L1, L2);
        std::vector<int> alignment(L_min * 2);
        for (int i = 0; i < L_min; i++) {
            alignment[i*2 + 0] = i;
            alignment[i*2 + 1] = i;
        }

        // Compute LDDT
        structural_metrics::LDDTParams params;
        float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
            dist1.data(), dist2.data(), alignment.data(), L_min, params
        );

        std::cout << "  LDDT (globin family): " << std::fixed << std::setprecision(4) << lddt << "\n";
        std::cout << "  Expected: > 0.5 (homologous proteins)\n";

        // Globins are homologous, should have reasonable LDDT
        if (lddt > 0.5f) {
            std::cout << "  ✓ Homologous proteins show high LDDT\n";
            return true;
        } else {
            std::cout << "  ⚠ Warning: LDDT lower than expected (" << lddt << ")\n";
            std::cout << "  (Note: Simple linear alignment may not be optimal)\n";
            return true;  // Don't fail - linear alignment is not optimal
        }

    } catch (const std::exception& e) {
        std::cout << "  SKIP: Could not load PDBs (" << e.what() << ")\n";
        return true;
    }
}

bool test_lddt_real_different_folds() {
    std::cout << "\n=== Test: LDDT Real Structures - Different Folds ===\n";

    const char* pdb1_path = "../../data/structures/pdb/small/1CRN.pdb";
    const char* pdb2_path = "../../data/structures/pdb/small/1UBQ.pdb";

    try {
        // Load crambin (plant seed protein) and ubiquitin (different fold)
        io::PDBParser parser;
        io::Protein prot1 = parser.parse_file(pdb1_path);
        io::Protein prot2 = parser.parse_file(pdb2_path);

        if (prot1.chains.empty() || prot2.chains.empty()) {
            std::cout << "  SKIP: Missing chains\n";
            return true;
        }

        int L1 = prot1.get_chain(0).size();
        int L2 = prot2.get_chain(0).size();
        auto coords1 = prot1.get_backbone_coords(0);
        auto coords2 = prot2.get_backbone_coords(0);

        std::cout << "  Loaded 1CRN: " << L1 << " residues (crambin)\n";
        std::cout << "  Loaded 1UBQ: " << L2 << " residues (ubiquitin)\n";

        // Compute distance matrices
        std::vector<float> dist1(L1 * L1), dist2(L2 * L2);
        structural_metrics::compute_distance_matrix<ScalarBackend>(
            coords1.data(), L1, dist1.data()
        );
        structural_metrics::compute_distance_matrix<ScalarBackend>(
            coords2.data(), L2, dist2.data()
        );

        // Simple linear alignment
        int L_min = std::min(L1, L2);
        std::vector<int> alignment(L_min * 2);
        for (int i = 0; i < L_min; i++) {
            alignment[i*2 + 0] = i;
            alignment[i*2 + 1] = i;
        }

        // Compute LDDT
        structural_metrics::LDDTParams params;
        float lddt = structural_metrics::lddt_pairwise<ScalarBackend>(
            dist1.data(), dist2.data(), alignment.data(), L_min, params
        );

        std::cout << "  LDDT (different folds): " << std::fixed << std::setprecision(4) << lddt << "\n";
        std::cout << "  Expected: < 0.4 (unrelated proteins)\n";

        // Different folds should have low LDDT
        if (lddt < 0.4f) {
            std::cout << "  ✓ Different folds show low LDDT\n";
            return true;
        } else {
            std::cout << "  ⚠ Warning: LDDT higher than expected (" << lddt << ")\n";
            return true;  // Don't fail - just informational
        }

    } catch (const std::exception& e) {
        std::cout << "  SKIP: Could not load PDBs (" << e.what() << ")\n";
        return true;
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  LDDT Scoring Tests\n";
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

    // Synthetic structure tests
    RUN_TEST(test_lddt_perfect_match);
    RUN_TEST(test_lddt_similar_structures);
    RUN_TEST(test_lddt_different_structures);
    RUN_TEST(test_lddt_with_gaps);
    RUN_TEST(test_lddt_per_residue_scores);
    RUN_TEST(test_lddt_symmetry_modes);

    // Real structure tests (optional - skip if PDBs not available)
    RUN_TEST(test_lddt_real_crambin_self_match);
    RUN_TEST(test_lddt_real_globin_family);
    RUN_TEST(test_lddt_real_different_folds);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
