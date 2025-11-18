/**
 * Unit tests for alignment decoding.
 */

#include "pfalign/primitives/alignment_decode/alignment_decode.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

using pfalign::ScalarBackend;
using pfalign::alignment_decode::decode_alignment;
using pfalign::AlignmentPair;
using pfalign::TracebackDirection;

constexpr float TOLERANCE = 1e-5f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

void print_path(const AlignmentPair* path, int path_len) {
    for (int k = 0; k < path_len; k++) {
        if (path[k].i == -1) {
            std::cout << "    Gap in seq1 (j=" << path[k].j << ")" << std::endl;
        } else if (path[k].j == -1) {
            std::cout << "    Gap in seq2 (i=" << path[k].i << ")" << std::endl;
        } else {
            std::cout << "    Match: " << path[k].i << " -> " << path[k].j
                      << " (p=" << std::fixed << std::setprecision(3)
                      << path[k].posterior << ")" << std::endl;
        }
    }
}

bool check_monotonicity(const AlignmentPair* path, int path_len) {
    int prev_i = -1;
    int prev_j = -1;

    for (int k = 0; k < path_len; k++) {
        int i = path[k].i;
        int j = path[k].j;

        // Check i is non-decreasing (skipping gaps in seq1)
        if (i != -1) {
            if (prev_i != -1 && i < prev_i) {
                return false;
            }
            prev_i = i;
        }

        // Check j is non-decreasing (skipping gaps in seq2)
        if (j != -1) {
            if (prev_j != -1 && j < prev_j) {
                return false;
            }
            prev_j = j;
        }
    }

    return true;
}

//==============================================================================
// Test 1: Perfect Diagonal - Strong Match
//==============================================================================

bool test_perfect_diagonal() {
    std::cout << "=== Test 1: Perfect Diagonal (3*3) ===" << std::endl;

    // Posteriors strongly peaked on diagonal
    float posteriors[9] = {
        0.8f, 0.05f, 0.05f,
        0.01f, 0.8f, 0.05f,
        0.01f, 0.01f, 0.22f  // Sum = 1.0
    };

    // Setup buffers
    AlignmentPair path[6];  // Max length 3+3
    float dp_score[16];     // (3+1) * (3+1)
    uint8_t dp_traceback[16];

    // Decode with moderate gap penalty
    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 3, -2.0f,
        path, 6,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << " (expected: 3)" << std::endl;
    print_path(path, path_len);

    // Expected: diagonal alignment [(0,0), (1,1), (2,2)]
    bool passed = (path_len == 3);
    passed &= (path[0].i == 0 && path[0].j == 0);
    passed &= (path[1].i == 1 && path[1].j == 1);
    passed &= (path[2].i == 2 && path[2].j == 2);
    passed &= close(path[0].posterior, 0.8f);
    passed &= close(path[1].posterior, 0.8f);
    passed &= check_monotonicity(path, path_len);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 2: Shifted Alignment - Off-Diagonal
//==============================================================================

bool test_shifted_alignment() {
    std::cout << "=== Test 2: Shifted Alignment (3*3) ===" << std::endl;

    // Posteriors peaked one position to the right (leading gap in seq1)
    float posteriors[9] = {
        0.01f, 0.7f, 0.01f,
        0.01f, 0.01f, 0.7f,
        0.01f, 0.01f, 0.54f  // Sum = 1.0
    };

    AlignmentPair path[6];
    float dp_score[16];
    uint8_t dp_traceback[16];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 3, -2.0f,
        path, 6,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << std::endl;
    print_path(path, path_len);

    // Expected: leading gap in seq1, then diagonal
    // Path: [(-1, 0), (0, 1), (1, 2), (2, ?)]
    bool passed = (path_len >= 3);
    passed &= (path[0].i == -1);  // Gap in seq1
    passed &= check_monotonicity(path, path_len);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 3: Insertion - L1 > L2
//==============================================================================

bool test_insertion() {
    std::cout << "=== Test 3: Insertion (5*3) ===" << std::endl;

    // 5 residues in seq1, only 3 in seq2 → expect 2 gaps in seq2
    float posteriors[15] = {
        0.3f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.3f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.4f   // Sum = 1.0
    };

    AlignmentPair path[8];  // Max 5+3
    float dp_score[24];     // (5+1) * (3+1)
    uint8_t dp_traceback[24];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 5, 3, -2.0f,
        path, 8,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << " (expected: 5, with 2 gaps)" << std::endl;
    print_path(path, path_len);

    // Count gaps in seq2 (j == -1)
    int gaps_seq2 = 0;
    for (int k = 0; k < path_len; k++) {
        if (path[k].j == -1) gaps_seq2++;
    }

    std::cout << "  Gaps in seq2: " << gaps_seq2 << std::endl;

    bool passed = (path_len == 5);
    passed &= (gaps_seq2 == 2);
    passed &= check_monotonicity(path, path_len);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 4: Deletion - L1 < L2
//==============================================================================

bool test_deletion() {
    std::cout << "=== Test 4: Deletion (3*5) ===" << std::endl;

    // 3 residues in seq1, 5 in seq2 → expect 2 gaps in seq1
    float posteriors[15] = {
        0.3f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.3f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.4f   // Sum = 1.0
    };

    AlignmentPair path[8];
    float dp_score[24];  // (3+1) * (5+1)
    uint8_t dp_traceback[24];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 5, -2.0f,
        path, 8,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << " (expected: 5, with 2 gaps)" << std::endl;
    print_path(path, path_len);

    // Count gaps in seq1 (i == -1)
    int gaps_seq1 = 0;
    for (int k = 0; k < path_len; k++) {
        if (path[k].i == -1) gaps_seq1++;
    }

    std::cout << "  Gaps in seq1: " << gaps_seq1 << std::endl;

    bool passed = (path_len == 5);
    passed &= (gaps_seq1 == 2);
    passed &= check_monotonicity(path, path_len);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 5: Single Residue (Edge Case)
//==============================================================================

bool test_single_residue() {
    std::cout << "=== Test 5: Single Residue (1*1) ===" << std::endl;

    float posteriors[1] = {1.0f};  // Perfect match

    AlignmentPair path[2];
    float dp_score[4];  // (1+1) * (1+1)
    uint8_t dp_traceback[4];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 1, 1, -2.0f,
        path, 2,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << " (expected: 1)" << std::endl;
    print_path(path, path_len);

    bool passed = (path_len == 1);
    passed &= (path[0].i == 0 && path[0].j == 0);
    passed &= close(path[0].posterior, 1.0f);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 6: Extreme Length Ratio (10*2)
//==============================================================================

bool test_extreme_ratio() {
    std::cout << "=== Test 6: Extreme Length Ratio (10*2) ===" << std::endl;

    // 10 residues vs 2: expect 8 gaps in seq2
    float posteriors[20] = {
        0.4f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.6f   // Sum = 1.0
    };

    AlignmentPair path[12];
    float dp_score[33];  // (10+1) * (2+1)
    uint8_t dp_traceback[33];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 10, 2, -2.0f,
        path, 12,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << " (expected: 10)" << std::endl;
    print_path(path, path_len);

    int gaps_seq2 = 0;
    for (int k = 0; k < path_len; k++) {
        if (path[k].j == -1) gaps_seq2++;
    }

    std::cout << "  Gaps in seq2: " << gaps_seq2 << " (expected: 8)" << std::endl;

    bool passed = (path_len == 10);
    passed &= (gaps_seq2 == 8);
    passed &= check_monotonicity(path, path_len);

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 7: Uniform Posteriors (Ambiguous Alignment)
//==============================================================================

bool test_uniform_posteriors() {
    std::cout << "=== Test 7: Uniform Posteriors (3*3) ===" << std::endl;

    // All posteriors equal (completely ambiguous)
    float p = 1.0f / 9.0f;
    float posteriors[9] = {p, p, p,
                            p, p, p,
                            p, p, p};

    AlignmentPair path[6];
    float dp_score[16];
    uint8_t dp_traceback[16];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 3, -2.0f,
        path, 6,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << std::endl;
    print_path(path, path_len);

    // With uniform posteriors and gap penalty -2.0, should prefer diagonal
    // (fewer gaps = fewer penalties)
    bool passed = (path_len > 0);
    passed &= check_monotonicity(path, path_len);

    if (passed) {
        std::cout << "  ✓ PASS (valid path, monotonic)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 8: Zero Posteriors (Numerical Stability)
//==============================================================================

bool test_zero_posteriors() {
    std::cout << "=== Test 8: Zero Posteriors (3*3) ===" << std::endl;

    // Some zero posteriors (should not crash, use LOG_ZERO)
    float posteriors[9] = {
        0.8f, 0.0f, 0.0f,
        0.0f, 0.2f, 0.0f,
        0.0f, 0.0f, 0.0f   // Sum = 1.0
    };

    AlignmentPair path[6];
    float dp_score[16];
    uint8_t dp_traceback[16];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 3, 3, -2.0f,
        path, 6,
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << std::endl;
    print_path(path, path_len);

    // Should not crash and produce valid path
    bool passed = (path_len > 0);
    passed &= check_monotonicity(path, path_len);

    // Should prefer (0,0) and (1,1) with high posteriors
    bool found_00 = false;
    bool found_11 = false;
    for (int k = 0; k < path_len; k++) {
        if (path[k].i == 0 && path[k].j == 0) found_00 = true;
        if (path[k].i == 1 && path[k].j == 1) found_11 = true;
    }

    passed &= found_00 && found_11;

    if (passed) {
        std::cout << "  ✓ PASS (no crash, includes high-posterior matches)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 9: Gap Penalty Sensitivity
//==============================================================================

bool test_gap_penalty_sensitivity() {
    std::cout << "=== Test 9: Gap Penalty Sensitivity ===" << std::endl;

    // Same posteriors, different gap penalties
    float posteriors[9] = {
        0.3f, 0.1f, 0.0f,
        0.1f, 0.3f, 0.0f,
        0.0f, 0.0f, 0.2f   // Sum = 1.0
    };

    // Lenient gap penalty (should allow more gaps)
    AlignmentPair path_lenient[6];
    float dp_score_lenient[16];
    uint8_t dp_traceback_lenient[16];

    int len_lenient = decode_alignment<ScalarBackend>(
        posteriors, 3, 3, -1.0f,  // Lenient
        path_lenient, 6,
        dp_score_lenient, dp_traceback_lenient
    );

    // Stringent gap penalty (should prefer matches, avoid gaps)
    AlignmentPair path_stringent[6];
    float dp_score_stringent[16];
    uint8_t dp_traceback_stringent[16];

    int len_stringent = decode_alignment<ScalarBackend>(
        posteriors, 3, 3, -5.0f,  // Stringent
        path_stringent, 6,
        dp_score_stringent, dp_traceback_stringent
    );

    std::cout << "  Lenient (-1.0):" << std::endl;
    print_path(path_lenient, len_lenient);

    std::cout << "  Stringent (-5.0):" << std::endl;
    print_path(path_stringent, len_stringent);

    // Count gaps in each
    int gaps_lenient = 0, gaps_stringent = 0;
    for (int k = 0; k < len_lenient; k++) {
        if (path_lenient[k].i == -1 || path_lenient[k].j == -1) gaps_lenient++;
    }
    for (int k = 0; k < len_stringent; k++) {
        if (path_stringent[k].i == -1 || path_stringent[k].j == -1) gaps_stringent++;
    }

    std::cout << "  Gaps (lenient): " << gaps_lenient << std::endl;
    std::cout << "  Gaps (stringent): " << gaps_stringent << std::endl;

    // Stringent should have fewer or equal gaps
    bool passed = (gaps_stringent <= gaps_lenient);
    passed &= check_monotonicity(path_lenient, len_lenient);
    passed &= check_monotonicity(path_stringent, len_stringent);

    if (passed) {
        std::cout << "  ✓ PASS (stringent has fewer gaps, both monotonic)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 10: Buffer Overflow Protection
//==============================================================================

bool test_buffer_overflow() {
    std::cout << "=== Test 10: Buffer Overflow Protection ===" << std::endl;

    // 5*5 alignment, but only provide buffer for 3 entries
    float posteriors[25];
    for (int i = 0; i < 25; i++) posteriors[i] = 0.04f;  // Sum = 1.0

    AlignmentPair path[3];  // Too small!
    float dp_score[36];     // (5+1) * (5+1)
    uint8_t dp_traceback[36];

    int path_len = decode_alignment<ScalarBackend>(
        posteriors, 5, 5, -2.0f,
        path, 3,  // max_path_length = 3, but needs ~5
        dp_score, dp_traceback
    );

    std::cout << "  Path length: " << path_len << " (expected: -1, overflow)" << std::endl;

    bool passed = (path_len == -1);  // Should return error

    if (passed) {
        std::cout << "  ✓ PASS (overflow detected)" << std::endl;
    } else {
        std::cout << "  ✗ FAIL (overflow not detected!)" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Alignment Decode Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 10;

    if (test_perfect_diagonal()) passed++;
    if (test_shifted_alignment()) passed++;
    if (test_insertion()) passed++;
    if (test_deletion()) passed++;
    if (test_single_residue()) passed++;
    if (test_extreme_ratio()) passed++;
    if (test_uniform_posteriors()) passed++;
    if (test_zero_posteriors()) passed++;
    if (test_gap_penalty_sensitivity()) passed++;
    if (test_buffer_overflow()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
