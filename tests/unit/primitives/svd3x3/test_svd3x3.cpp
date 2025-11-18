/**
 * Unit tests for 3*3 SVD primitive.
 *
 * Tests:
 * 1. Orthogonality: U Uᵀ = I, V Vᵀ = I
 * 2. Reconstruction: A = U Sigma Vᵀ
 * 3. Singular value ordering: S[0] >= S[1] >= S[2] >= 0
 * 4. Known matrices: identity, rotation, scaling
 */

#include "pfalign/primitives/svd3x3/svd3x3_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>

using namespace pfalign;

// ============================================================================
// Test Utilities
// ============================================================================

const float EPS = 1e-4f;  // Tolerance for floating-point comparison

void print_matrix3x3(const char* name, const float* M) {
    std::cout << name << ":\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(5) << M[i*3 + j];
            if (j < 2) std::cout << ",";
        }
        std::cout << " ]\n";
    }
}

void print_vector3(const char* name, const float* v) {
    std::cout << name << ": [";
    for (int i = 0; i < 3; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(5) << v[i];
        if (i < 2) std::cout << ",";
    }
    std::cout << " ]\n";
}

bool check_orthogonal(const float* M, const char* name) {
    // Compute M Mᵀ
    float MMT[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                MMT[i*3 + j] += M[i*3 + k] * M[j*3 + k];
            }
        }
    }

    // Check if M Mᵀ = I
    float error = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float diff = std::abs(MMT[i*3 + j] - expected);
            error = std::max(error, diff);
        }
    }

    std::cout << "  " << name << " orthogonality error: " << error;
    if (error < EPS) {
        std::cout << " ✓\n";
        return true;
    } else {
        std::cout << " ✗ (expected < " << EPS << ")\n";
        print_matrix3x3("MMT", MMT);
        return false;
    }
}

bool check_reconstruction(const float* A_orig, const float* U, const float* S, const float* V) {
    // Compute U Sigma Vᵀ
    float Sigma[9] = {
        S[0], 0, 0,
        0, S[1], 0,
        0, 0, S[2]
    };

    float US[9];  // U @ Sigma
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            US[i*3 + j] = U[i*3 + j] * S[j];
        }
    }

    float A_reconstructed[9];  // (U @ Sigma) @ Vᵀ
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += US[i*3 + k] * V[j*3 + k];  // Vᵀ: transpose indices
            }
            A_reconstructed[i*3 + j] = sum;
        }
    }

    // Compute ||A - U Sigma Vᵀ||_F
    float error = 0.0f;
    for (int i = 0; i < 9; i++) {
        float diff = A_orig[i] - A_reconstructed[i];
        error += diff * diff;
    }
    error = std::sqrt(error);

    std::cout << "  Reconstruction error: " << error;
    if (error < EPS) {
        std::cout << " ✓\n";
        return true;
    } else {
        std::cout << " ✗ (expected < " << EPS << ")\n";
        print_matrix3x3("A_original", A_orig);
        print_matrix3x3("A_reconstructed", A_reconstructed);
        return false;
    }
}

bool check_singular_values_ordered(const float* S) {
    bool ordered = (S[0] >= S[1] - EPS) && (S[1] >= S[2] - EPS) && (S[2] >= -EPS);
    std::cout << "  Singular values ordered: ";
    print_vector3("S", S);
    if (ordered) {
        std::cout << "  ✓ Descending order\n";
        return true;
    } else {
        std::cout << "  ✗ Not in descending order\n";
        return false;
    }
}

// ============================================================================
// Test Cases
// ============================================================================

bool test_identity() {
    std::cout << "\n=== Test: Identity Matrix ===\n";

    float A[9] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    float A_orig[9];
    std::memcpy(A_orig, A, sizeof(A));

    float U[9], S[3], V[9];
    svd3x3::svd3x3<ScalarBackend>(A, U, S, V);

    print_vector3("Singular values", S);

    bool pass = true;
    pass &= check_orthogonal(U, "U");
    pass &= check_orthogonal(V, "V");
    pass &= check_reconstruction(A_orig, U, S, V);
    pass &= check_singular_values_ordered(S);

    // For identity: expect S = [1, 1, 1]
    float s_error = std::abs(S[0] - 1.0f) + std::abs(S[1] - 1.0f) + std::abs(S[2] - 1.0f);
    std::cout << "  Identity singular values error: " << s_error;
    if (s_error < EPS) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
        pass = false;
    }

    return pass;
}

bool test_diagonal() {
    std::cout << "\n=== Test: Diagonal Matrix ===\n";

    float A[9] = {
        5, 0, 0,
        0, 3, 0,
        0, 0, 1
    };
    float A_orig[9];
    std::memcpy(A_orig, A, sizeof(A));

    float U[9], S[3], V[9];
    svd3x3::svd3x3<ScalarBackend>(A, U, S, V);

    print_vector3("Singular values", S);

    bool pass = true;
    pass &= check_orthogonal(U, "U");
    pass &= check_orthogonal(V, "V");
    pass &= check_reconstruction(A_orig, U, S, V);
    pass &= check_singular_values_ordered(S);

    // For diagonal [5, 3, 1]: expect S = [5, 3, 1]
    float s_error = std::abs(S[0] - 5.0f) + std::abs(S[1] - 3.0f) + std::abs(S[2] - 1.0f);
    std::cout << "  Diagonal singular values error: " << s_error;
    if (s_error < EPS) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
        pass = false;
    }

    return pass;
}

bool test_rotation() {
    std::cout << "\n=== Test: Rotation Matrix (90deg around Z) ===\n";

    // 90deg rotation around Z-axis
    float A[9] = {
        0, -1, 0,
        1,  0, 0,
        0,  0, 1
    };
    float A_orig[9];
    std::memcpy(A_orig, A, sizeof(A));

    float U[9], S[3], V[9];
    svd3x3::svd3x3<ScalarBackend>(A, U, S, V);

    print_vector3("Singular values", S);

    bool pass = true;
    pass &= check_orthogonal(U, "U");
    pass &= check_orthogonal(V, "V");
    pass &= check_reconstruction(A_orig, U, S, V);
    pass &= check_singular_values_ordered(S);

    // For rotation: expect S = [1, 1, 1] (orthogonal matrix)
    float s_error = std::abs(S[0] - 1.0f) + std::abs(S[1] - 1.0f) + std::abs(S[2] - 1.0f);
    std::cout << "  Rotation singular values error: " << s_error;
    if (s_error < EPS) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗\n";
        pass = false;
    }

    return pass;
}

bool test_general() {
    std::cout << "\n=== Test: General Matrix ===\n";

    float A[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    float A_orig[9];
    std::memcpy(A_orig, A, sizeof(A));

    float U[9], S[3], V[9];
    svd3x3::svd3x3<ScalarBackend>(A, U, S, V);

    print_vector3("Singular values", S);

    bool pass = true;
    pass &= check_orthogonal(U, "U");
    pass &= check_orthogonal(V, "V");
    pass &= check_reconstruction(A_orig, U, S, V);
    pass &= check_singular_values_ordered(S);

    return pass;
}

bool test_rank_deficient() {
    std::cout << "\n=== Test: Rank-Deficient Matrix ===\n";

    // Rank 1: all rows are multiples of [1, 2, 3]
    float A[9] = {
        1, 2, 3,
        2, 4, 6,
        3, 6, 9
    };
    float A_orig[9];
    std::memcpy(A_orig, A, sizeof(A));

    float U[9], S[3], V[9];
    svd3x3::svd3x3<ScalarBackend>(A, U, S, V);

    print_vector3("Singular values", S);

    bool pass = true;
    pass &= check_orthogonal(U, "U");
    pass &= check_orthogonal(V, "V");
    pass &= check_reconstruction(A_orig, U, S, V);
    pass &= check_singular_values_ordered(S);

    // For rank-1: expect S[1] ~= 0, S[2] ~= 0
    std::cout << "  Checking rank deficiency (S[1], S[2] ~= 0): ";
    if (S[1] < EPS && S[2] < EPS) {
        std::cout << "✓\n";
    } else {
        std::cout << "✗ (S[1]=" << S[1] << ", S[2]=" << S[2] << ")\n";
        pass = false;
    }

    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  3*3 SVD Primitive Tests\n";
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

    RUN_TEST(test_identity);
    RUN_TEST(test_diagonal);
    RUN_TEST(test_rotation);
    RUN_TEST(test_general);
    RUN_TEST(test_rank_deficient);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
