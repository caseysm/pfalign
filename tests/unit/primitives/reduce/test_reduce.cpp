/**
 * Unit tests for reduce operations.
 */

#include "pfalign/primitives/reduce/reduce.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using pfalign::ScalarBackend;
using pfalign::reduce::elementwise_multiply;
using pfalign::reduce::matrix_sum;
using pfalign::reduce::frobenius_norm;

constexpr float TOLERANCE = 1e-5f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

//==============================================================================
// Test 1: Element-wise Multiplication - Simple Case
//==============================================================================

bool test_elementwise_multiply_simple() {
    std::cout << "=== Test 1: Element-wise Multiply (Simple) ===" << std::endl;

    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float C[4];
    float expected[4] = {2.0f, 6.0f, 12.0f, 20.0f};

    elementwise_multiply<ScalarBackend>(A, B, C, 4);

    bool passed = true;
    for (int i = 0; i < 4; i++) {
        if (!close(C[i], expected[i])) {
            std::cout << "  Mismatch at index " << i << ": "
                      << C[i] << " vs " << expected[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 2: Element-wise Multiplication - Zeros
//==============================================================================

bool test_elementwise_multiply_zeros() {
    std::cout << "=== Test 2: Element-wise Multiply (Zeros) ===" << std::endl;

    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float C[4];

    elementwise_multiply<ScalarBackend>(A, B, C, 4);

    bool passed = true;
    for (int i = 0; i < 4; i++) {
        if (!close(C[i], 0.0f)) {
            std::cout << "  Expected 0.0 at index " << i << ", got " << C[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 3: Matrix Sum - Simple Case
//==============================================================================

bool test_matrix_sum_simple() {
    std::cout << "=== Test 3: Matrix Sum (Simple) ===" << std::endl;

    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sum = matrix_sum<ScalarBackend>(A, 4);
    float expected = 10.0f;

    bool passed = close(sum, expected);

    std::cout << "  Sum: " << sum << " (expected: " << expected << ")" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 4: Matrix Sum - All Zeros
//==============================================================================

bool test_matrix_sum_zeros() {
    std::cout << "=== Test 4: Matrix Sum (Zeros) ===" << std::endl;

    float A[100];
    for (int i = 0; i < 100; i++) A[i] = 0.0f;

    float sum = matrix_sum<ScalarBackend>(A, 100);

    bool passed = close(sum, 0.0f);

    std::cout << "  Sum: " << sum << " (expected: 0.0)" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 5: Matrix Sum - Negative Values
//==============================================================================

bool test_matrix_sum_negative() {
    std::cout << "=== Test 5: Matrix Sum (Negative Values) ===" << std::endl;

    float A[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float sum = matrix_sum<ScalarBackend>(A, 4);
    float expected = -2.0f;

    bool passed = close(sum, expected);

    std::cout << "  Sum: " << sum << " (expected: " << expected << ")" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 6: Frobenius Norm - Simple Case (3-4-5 Triangle)
//==============================================================================

bool test_frobenius_norm_simple() {
    std::cout << "=== Test 6: Frobenius Norm (3-4-5 Triangle) ===" << std::endl;

    float A[3] = {3.0f, 4.0f, 0.0f};
    float norm = frobenius_norm<ScalarBackend>(A, 3);
    float expected = 5.0f;  // sqrt(9 + 16 + 0) = sqrt(25) = 5

    bool passed = close(norm, expected);

    std::cout << "  Norm: " << norm << " (expected: " << expected << ")" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 7: Frobenius Norm - Identity Matrix
//==============================================================================

bool test_frobenius_norm_identity() {
    std::cout << "=== Test 7: Frobenius Norm (Identity Matrix 3*3) ===" << std::endl;

    // Identity matrix: diag([1, 1, 1])
    float A[9] = {1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f};

    float norm = frobenius_norm<ScalarBackend>(A, 9);
    float expected = std::sqrt(3.0f);  // sqrt(1 + 1 + 1)

    bool passed = close(norm, expected);

    std::cout << "  Norm: " << norm << " (expected: " << expected << ")" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 8: Frobenius Norm - All Ones
//==============================================================================

bool test_frobenius_norm_ones() {
    std::cout << "=== Test 8: Frobenius Norm (All Ones) ===" << std::endl;

    const int size = 100;
    float A[size];
    for (int i = 0; i < size; i++) A[i] = 1.0f;

    float norm = frobenius_norm<ScalarBackend>(A, size);
    float expected = std::sqrt(static_cast<float>(size));  // sqrt(100) = 10

    bool passed = close(norm, expected);

    std::cout << "  Norm: " << norm << " (expected: " << expected << ")" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 9: Integration - Compute Alignment Score Formula
//==============================================================================

bool test_alignment_score_formula() {
    std::cout << "=== Test 9: Alignment Score Formula (Integration) ===" << std::endl;

    // Simple 3*3 diagonal similarity and posteriors
    float similarity[9] = {10.0f, 0.0f, 0.0f,
                            0.0f, 10.0f, 0.0f,
                            0.0f, 0.0f, 10.0f};

    float posteriors[9] = {0.33f, 0.0f, 0.0f,
                            0.0f, 0.33f, 0.0f,
                            0.0f, 0.0f, 0.34f};  // sum = 1.0

    // Step 1: Frobenius normalize similarity
    float sim_norm = frobenius_norm<ScalarBackend>(similarity, 9);
    float S_normalized[9];
    for (int i = 0; i < 9; i++) {
        S_normalized[i] = similarity[i] / sim_norm;
    }

    // Step 2: Element-wise multiply
    float weighted[9];
    elementwise_multiply<ScalarBackend>(S_normalized, posteriors, weighted, 9);

    // Step 3: Sum
    float score = matrix_sum<ScalarBackend>(weighted, 9);

    // Expected: diagonal elements contribute
    // sim_norm = sqrt(10^2 + 10^2 + 10^2) = sqrt(300) ~= 17.32
    // S_norm[0,0] = S_norm[1,1] = S_norm[2,2] ~= 0.577
    // score ~= 0.577 * 0.33 + 0.577 * 0.33 + 0.577 * 0.34 ~= 0.577
    float expected = (10.0f / sim_norm) * 1.0f;  // Simplified

    bool passed = close(score, expected, 1e-3f);

    std::cout << "  Similarity Frobenius norm: " << sim_norm << std::endl;
    std::cout << "  Score: " << score << " (expected: ~" << expected << ")" << std::endl;
    std::cout << "  Score in [0, 1]: " << (score >= 0.0f && score <= 1.0f ? "YES" : "NO") << std::endl;

    if (passed && score >= 0.0f && score <= 1.0f) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Test 10: Large Array (Performance Check)
//==============================================================================

bool test_large_array() {
    std::cout << "=== Test 10: Large Array (10000 elements) ===" << std::endl;

    const int size = 10000;
    std::vector<float> A(size, 1.0f);
    std::vector<float> B(size, 2.0f);
    std::vector<float> C(size);

    // Element-wise multiply
    elementwise_multiply<ScalarBackend>(A.data(), B.data(), C.data(), size);

    // Check a few elements
    bool multiply_ok = close(C[0], 2.0f) && close(C[size-1], 2.0f);

    // Matrix sum
    float sum = matrix_sum<ScalarBackend>(C.data(), size);
    bool sum_ok = close(sum, 20000.0f);  // 10000 * 2.0

    // Frobenius norm
    float norm = frobenius_norm<ScalarBackend>(C.data(), size);
    bool norm_ok = close(norm, std::sqrt(40000.0f));  // sqrt(10000 * 4)

    bool passed = multiply_ok && sum_ok && norm_ok;

    std::cout << "  Multiply check: " << (multiply_ok ? "OK" : "FAIL") << std::endl;
    std::cout << "  Sum: " << sum << " (expected: 20000.0)" << std::endl;
    std::cout << "  Norm: " << norm << " (expected: " << std::sqrt(40000.0f) << ")" << std::endl;

    if (passed) {
        std::cout << "  ✓ PASS" << std::endl;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Reduce Operations Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 10;

    if (test_elementwise_multiply_simple()) passed++;
    if (test_elementwise_multiply_zeros()) passed++;
    if (test_matrix_sum_simple()) passed++;
    if (test_matrix_sum_zeros()) passed++;
    if (test_matrix_sum_negative()) passed++;
    if (test_frobenius_norm_simple()) passed++;
    if (test_frobenius_norm_identity()) passed++;
    if (test_frobenius_norm_ones()) passed++;
    if (test_alignment_score_formula()) passed++;
    if (test_large_array()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
