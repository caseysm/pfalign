/**
 * Unit tests for Kabsch algorithm.
 *
 * Tests:
 * 1. Identity transform (same point set → RMSD = 0)
 * 2. Pure translation
 * 3. Pure rotation
 * 4. Rotation + translation
 * 5. Rotation with noise
 */

#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>

using namespace pfalign;

// ============================================================================
// Test Utilities
// ============================================================================

const float EPS = 1e-4f;

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

bool check_rotation_matrix(const float* R) {
    // Check orthogonality: R Rᵀ = I
    float RRT[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                RRT[i*3 + j] += R[i*3 + k] * R[j*3 + k];
            }
        }
    }

    float error = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            error = std::max(error, std::abs(RRT[i*3 + j] - expected));
        }
    }

    std::cout << "  Orthogonality error: " << error;
    if (error < EPS) {
        std::cout << " ✓\n";
        return true;
    } else {
        std::cout << " ✗\n";
        return false;
    }
}

bool check_determinant(const float* R) {
    float det = R[0] * (R[4]*R[8] - R[5]*R[7])
              - R[1] * (R[3]*R[8] - R[5]*R[6])
              + R[2] * (R[3]*R[7] - R[4]*R[6]);

    std::cout << "  Determinant: " << det;
    if (std::abs(det - 1.0f) < EPS) {
        std::cout << " ✓ (proper rotation)\n";
        return true;
    } else {
        std::cout << " ✗ (expected 1.0)\n";
        return false;
    }
}

// ============================================================================
// Test Cases
// ============================================================================

bool test_identity() {
    std::cout << "\n=== Test: Identity (Same Point Set) ===\n";

    int N = 5;
    float P[15] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f
    };

    float R[9], t[3], rmsd;
    kabsch::kabsch_align<ScalarBackend>(P, P, N, R, t, &rmsd);

    print_matrix3x3("R (rotation)", R);
    print_vector3("t (translation)", t);
    std::cout << "RMSD: " << rmsd << "\n";

    bool pass = true;
    pass &= check_rotation_matrix(R);
    pass &= check_determinant(R);

    // For identical point sets: expect RMSD ~= 0
    std::cout << "  RMSD check: " << rmsd;
    if (rmsd < EPS) {
        std::cout << " ✓ (~= 0)\n";
    } else {
        std::cout << " ✗ (expected ~= 0)\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    else std::cout << "  FAIL\n";
    return pass;
}

bool test_pure_translation() {
    std::cout << "\n=== Test: Pure Translation ===\n";

    int N = 5;
    float P[15] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f
    };

    // Q = P + [10, 20, 30]
    float Q[15];
    for (int i = 0; i < N; i++) {
        Q[i*3 + 0] = P[i*3 + 0] + 10.0f;
        Q[i*3 + 1] = P[i*3 + 1] + 20.0f;
        Q[i*3 + 2] = P[i*3 + 2] + 30.0f;
    }

    float R[9], t[3], rmsd;
    kabsch::kabsch_align<ScalarBackend>(P, Q, N, R, t, &rmsd);

    print_matrix3x3("R (rotation)", R);
    print_vector3("t (translation)", t);
    std::cout << "RMSD: " << rmsd << "\n";

    bool pass = true;
    pass &= check_rotation_matrix(R);
    pass &= check_determinant(R);

    // Check translation: should be [10, 20, 30]
    float t_error = std::abs(t[0] - 10.0f) + std::abs(t[1] - 20.0f) + std::abs(t[2] - 30.0f);
    std::cout << "  Translation check:";
    if (t_error < EPS && rmsd < EPS) {
        std::cout << " ✓ (t=[10,20,30], rmsd~=0)\n";
    } else {
        std::cout << " ✗ (t_error=" << t_error << ", rmsd=" << rmsd << ")\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    else std::cout << "  FAIL\n";
    return pass;
}

bool test_rotation_90z() {
    std::cout << "\n=== Test: 90deg Rotation Around Z-Axis ===\n";

    int N = 4;
    float P[12] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        -1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f
    };

    // Q = R_z(90deg) @ P
    float Q[12] = {
        0.0f, 1.0f, 0.0f,
        -1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
        1.0f, 0.0f, 0.0f
    };

    float R[9], t[3], rmsd;
    kabsch::kabsch_align<ScalarBackend>(P, Q, N, R, t, &rmsd);

    print_matrix3x3("R (rotation)", R);
    print_vector3("t (translation)", t);
    std::cout << "RMSD: " << rmsd << "\n";

    bool pass = true;
    pass &= check_rotation_matrix(R);
    pass &= check_determinant(R);

    // Check RMSD ~= 0 (perfect alignment)
    std::cout << "  RMSD check: " << rmsd;
    if (rmsd < EPS) {
        std::cout << " ✓\n";
    } else {
        std::cout << " ✗ (expected ~= 0)\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    else std::cout << "  FAIL\n";
    return pass;
}

bool test_rotation_with_noise() {
    std::cout << "\n=== Test: Rotation + Small Noise ===\n";

    int N = 10;
    float P[30];
    float Q[30];

    // Generate random points
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int i = 0; i < N; i++) {
        P[i*3 + 0] = dist(gen);
        P[i*3 + 1] = dist(gen);
        P[i*3 + 2] = dist(gen);
    }

    // Apply 45deg rotation around Y-axis + small noise
    float cos45 = std::cos(M_PI / 4.0f);
    float sin45 = std::sin(M_PI / 4.0f);
    float R_true[9] = {
        cos45, 0, sin45,
        0, 1, 0,
        -sin45, 0, cos45
    };

    std::normal_distribution<float> noise(0.0f, 0.01f);  // Small noise
    for (int i = 0; i < N; i++) {
        Q[i*3 + 0] = R_true[0]*P[i*3+0] + R_true[1]*P[i*3+1] + R_true[2]*P[i*3+2] + noise(gen);
        Q[i*3 + 1] = R_true[3]*P[i*3+0] + R_true[4]*P[i*3+1] + R_true[5]*P[i*3+2] + noise(gen);
        Q[i*3 + 2] = R_true[6]*P[i*3+0] + R_true[7]*P[i*3+1] + R_true[8]*P[i*3+2] + noise(gen);
    }

    float R[9], t[3], rmsd;
    kabsch::kabsch_align<ScalarBackend>(P, Q, N, R, t, &rmsd);

    print_matrix3x3("R (recovered)", R);
    print_matrix3x3("R_true", R_true);
    std::cout << "RMSD: " << rmsd << "\n";

    bool pass = true;
    pass &= check_rotation_matrix(R);
    pass &= check_determinant(R);

    // With noise, RMSD should be small but non-zero
    std::cout << "  RMSD check (with noise): " << rmsd;
    if (rmsd < 0.1f) {
        std::cout << " ✓ (small, as expected with noise)\n";
    } else {
        std::cout << " ✗ (too large)\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    else std::cout << "  FAIL\n";
    return pass;
}

bool test_apply_transformation() {
    std::cout << "\n=== Test: Apply Transformation to Full Structure ===\n";

    // Simple transform: R = I, t = [10, 20, 30]
    float R[9] = {1,0,0, 0,1,0, 0,0,1};
    float t[3] = {10, 20, 30};

    // Input: single residue (4 atoms)
    int L = 1;
    float coords_in[12] = {
        0,1,0,  // N
        0,1,0,  // CA
        0,1,0,  // C
        0,1,0   // O
    };

    float coords_out[12];
    kabsch::apply_transformation<ScalarBackend>(R, t, coords_in, coords_out, L);

    // Check first atom transformed correctly
    std::cout << "  First atom transformed: [" << coords_out[0] << ", "
              << coords_out[1] << ", " << coords_out[2] << "]\n";

    bool pass = (std::abs(coords_out[0] - 10.0f) < EPS &&
                 std::abs(coords_out[1] - 21.0f) < EPS &&
                 std::abs(coords_out[2] - 30.0f) < EPS);

    if (pass) {
        std::cout << "  ✓ Transformation correct\n";
        std::cout << "  PASS\n";
    } else {
        std::cout << "  ✗ Transformation incorrect\n";
        std::cout << "  FAIL\n";
    }

    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  Kabsch Algorithm Tests\n";
    std::cout << "========================================\n";

    int passed = 0;
    int total = 5;

    passed += test_identity();
    passed += test_pure_translation();
    passed += test_rotation_90z();
    passed += test_rotation_with_noise();
    passed += test_apply_transformation();

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
