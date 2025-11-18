/**
 * Simple test for scalar GEMM implementation.
 *
 * Compile and run:
 *   g++ -std=c++17 -I../../.. test_gemm.cpp gemm_scalar.cpp ../../dispatch/runtime_dispatch.cpp -o test_gemm
 *   ./test_gemm
 */

#include "pfalign/primitives/gemm/gemm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using pfalign::ScalarBackend;
using pfalign::gemm::gemm;
using pfalign::gemm::gemv;

// Helper: Print matrix
void print_matrix(const char* name, const float* M, int rows, int cols) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << M[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper: Check if matrices are close
bool matrices_close(const float* A, const float* B, int M, int N, float tol = 1e-5f) {
    for (int i = 0; i < M * N; i++) {
        if (std::abs(A[i] - B[i]) > tol) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Test 1: Small matrix multiply
bool test_small_gemm() {
    std::cout << "=== Test 1: Small GEMM (4x4) ===" << std::endl;

    // A = [[1, 2], [3, 4], [5, 6], [7, 8]]  (4*2)
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8};

    // B = [[1, 2, 3], [4, 5, 6]]  (2*3)
    float B[] = {1, 2, 3, 4, 5, 6};

    // C = A @ B (4*3)
    // Expected: [[9, 12, 15], [19, 26, 33], [29, 40, 51], [39, 54, 69]]
    float C[12] = {0};
    float expected[] = {9, 12, 15, 19, 26, 33, 29, 40, 51, 39, 54, 69};

    gemm<ScalarBackend>(A, B, C, 4, 3, 2, 1.0f, 0.0f, 2, 3, 3);

    print_matrix("A", A, 4, 2);
    print_matrix("B", B, 2, 3);
    print_matrix("C (result)", C, 4, 3);
    print_matrix("C (expected)", expected, 4, 3);

    bool pass = matrices_close(C, expected, 4, 3);
    std::cout << (pass ? "✓ PASS" : "✗ FAIL") << std::endl << std::endl;
    return pass;
}

// Test 2: Identity matrix
bool test_identity() {
    std::cout << "=== Test 2: Identity Matrix ===" << std::endl;

    int N = 4;
    float A[16];
    float I[16] = {0};  // Identity
    float C[16] = {0};

    // Create identity matrix
    for (int i = 0; i < N; i++) {
        I[i * N + i] = 1.0f;
    }

    // Create test matrix A
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(i + 1);
    }

    // C = A @ I (should equal A)
    gemm<ScalarBackend>(A, I, C, N, N, N, 1.0f, 0.0f, N, N, N);

    print_matrix("A", A, N, N);
    print_matrix("I", I, N, N);
    print_matrix("C = A @ I", C, N, N);

    bool pass = matrices_close(C, A, N, N);
    std::cout << (pass ? "✓ PASS" : "✗ FAIL") << std::endl << std::endl;
    return pass;
}

// Test 3: Alpha and beta scaling
bool test_alpha_beta() {
    std::cout << "=== Test 3: Alpha/Beta Scaling ===" << std::endl;

    float A[] = {1, 2, 3, 4};  // 2*2
    float B[] = {5, 6, 7, 8};  // 2*2
    float C[] = {1, 1, 1, 1};  // 2*2 (initial value)

    // C = 2.0 * A @ B + 0.5 * C
    // A @ B = [[19, 22], [43, 50]]
    // Result = 2 * [[19, 22], [43, 50]] + 0.5 * [[1, 1], [1, 1]]
    //        = [[38, 44], [86, 100]] + [[0.5, 0.5], [0.5, 0.5]]
    //        = [[38.5, 44.5], [86.5, 100.5]]
    float expected[] = {38.5f, 44.5f, 86.5f, 100.5f};

    gemm<ScalarBackend>(A, B, C, 2, 2, 2, 2.0f, 0.5f, 2, 2, 2);

    print_matrix("Result", C, 2, 2);
    print_matrix("Expected", expected, 2, 2);

    bool pass = matrices_close(C, expected, 2, 2, 1e-4f);
    std::cout << (pass ? "✓ PASS" : "✗ FAIL") << std::endl << std::endl;
    return pass;
}

// Test 4: GEMV (matrix-vector multiply)
bool test_gemv() {
    std::cout << "=== Test 4: GEMV ===" << std::endl;

    // A = [[1, 2, 3], [4, 5, 6]]  (2*3)
    float A[] = {1, 2, 3, 4, 5, 6};

    // x = [1, 2, 3]
    float x[] = {1, 2, 3};

    // y = A @ x = [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
    float y[2] = {0};
    float expected[] = {14, 32};

    gemv<ScalarBackend>(A, x, y, 2, 3, 1.0f, 0.0f, 3);

    std::cout << "y = [" << y[0] << ", " << y[1] << "]" << std::endl;
    std::cout << "Expected = [" << expected[0] << ", " << expected[1] << "]" << std::endl;

    bool pass = matrices_close(y, expected, 2, 1);
    std::cout << (pass ? "✓ PASS" : "✗ FAIL") << std::endl << std::endl;
    return pass;
}

// Test 5: Protein-scale matrix (typical MPNN size)
bool test_protein_scale() {
    std::cout << "=== Test 5: Protein-Scale Matrix (100*128 @ 128*64) ===" << std::endl;

    const int M = 100;  // Protein length
    const int K = 128;  // Embedding dimension
    const int N = 64;   // Output dimension

    // Allocate matrices
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];

    // Initialize with simple pattern
    for (int i = 0; i < M * K; i++) A[i] = 0.01f * (i % 10);
    for (int i = 0; i < K * N; i++) B[i] = 0.01f * (i % 10);

    // Compute GEMM
    gemm<ScalarBackend>(A, B, C, M, N, K, 1.0f, 0.0f, K, N, N);

    // Sanity check: C should not be all zeros or all same
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) sum += C[i];

    std::cout << "Matrix size: " << M << " * " << N << std::endl;
    std::cout << "Sum of elements: " << sum << std::endl;
    std::cout << "Sample values: C[0] = " << C[0] << ", C[M*N-1] = " << C[M*N-1] << std::endl;

    bool pass = (std::abs(sum) > 1e-3f);  // Should have non-trivial values

    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << (pass ? "✓ PASS" : "✗ FAIL") << std::endl << std::endl;
    return pass;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Scalar GEMM Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    int passed = 0;
    int total = 5;

    if (test_small_gemm()) passed++;
    if (test_identity()) passed++;
    if (test_alpha_beta()) passed++;
    if (test_gemv()) passed++;
    if (test_protein_scale()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
