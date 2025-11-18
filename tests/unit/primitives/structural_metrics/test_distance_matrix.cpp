/**
 * Unit tests for distance matrix computation.
 */

#include "pfalign/primitives/structural_metrics/distance_matrix.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace pfalign;

const float EPS = 1e-4f;

bool test_extract_ca_atoms() {
    std::cout << "\n=== Test: Extract CA Atoms ===\n";

    // Create mock backbone coords: [3 residues * 4 atoms * 3 dims]
    int L = 3;
    std::vector<float> backbone(L * 4 * 3);

    // Fill with test data: residue i, atom j, coord k → value = i*100 + j*10 + k
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < 4; j++) {  // N, CA, C, O
            for (int k = 0; k < 3; k++) {  // x, y, z
                backbone[(i * 4 + j) * 3 + k] = static_cast<float>(i*100 + j*10 + k);
            }
        }
    }

    // Extract CA atoms (j=1)
    std::vector<float> ca_coords(L * 3);
    structural_metrics::extract_ca_atoms(backbone.data(), L, ca_coords.data());

    // Verify
    bool pass = true;
    for (int i = 0; i < L; i++) {
        for (int k = 0; k < 3; k++) {
            float expected = static_cast<float>(i*100 + 1*10 + k);  // CA is atom 1
            float actual = ca_coords[i*3 + k];
            if (std::abs(actual - expected) > EPS) {
                std::cout << "  ✗ Mismatch at residue " << i << " dim " << k
                          << ": expected " << expected << ", got " << actual << "\n";
                pass = false;
            }
        }
    }

    if (pass) {
        std::cout << "  ✓ CA atoms extracted correctly\n";
    }
    return pass;
}

bool test_distance_matrix_simple() {
    std::cout << "\n=== Test: Distance Matrix (Simple 3-Point) ===\n";

    // Three points: (0,0,0), (1,0,0), (0,1,0)
    int L = 3;
    std::vector<float> ca_coords = {
        0.0f, 0.0f, 0.0f,  // Point 0
        1.0f, 0.0f, 0.0f,  // Point 1 (distance 1 from point 0)
        0.0f, 1.0f, 0.0f   // Point 2 (distance 1 from point 0, sqrt(2) from point 1)
    };

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Expected distances
    float expected[3][3] = {
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, std::sqrt(2.0f)},
        {1.0f, std::sqrt(2.0f), 0.0f}
    };

    bool pass = true;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            float actual = dist_mx[i * L + j];
            float exp = expected[i][j];
            if (std::abs(actual - exp) > EPS) {
                std::cout << "  ✗ Distance [" << i << "][" << j << "]: "
                          << "expected " << exp << ", got " << actual << "\n";
                pass = false;
            }
        }
    }

    if (pass) {
        std::cout << "  ✓ All distances correct\n";
        std::cout << "    d(0,1) = " << dist_mx[0*L + 1] << " (expected 1.0)\n";
        std::cout << "    d(1,2) = " << dist_mx[1*L + 2] << " (expected "
                  << std::sqrt(2.0f) << ")\n";
    }
    return pass;
}

bool test_distance_matrix_symmetry() {
    std::cout << "\n=== Test: Distance Matrix Symmetry ===\n";

    // Random-ish points
    int L = 5;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L; i++) {
        ca_coords[i*3 + 0] = static_cast<float>(i * 1.5);
        ca_coords[i*3 + 1] = static_cast<float>(i * 2.3);
        ca_coords[i*3 + 2] = static_cast<float>(i * 0.7);
    }

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Check symmetry: dist[i][j] == dist[j][i]
    bool pass = true;
    for (int i = 0; i < L; i++) {
        for (int j = i + 1; j < L; j++) {
            float d_ij = dist_mx[i * L + j];
            float d_ji = dist_mx[j * L + i];
            if (std::abs(d_ij - d_ji) > EPS) {
                std::cout << "  ✗ Asymmetry at [" << i << "][" << j << "]: "
                          << d_ij << " vs " << d_ji << "\n";
                pass = false;
            }
        }
    }

    if (pass) {
        std::cout << "  ✓ Matrix is symmetric\n";
    }
    return pass;
}

bool test_distance_matrix_diagonal() {
    std::cout << "\n=== Test: Distance Matrix Diagonal ===\n";

    int L = 10;
    std::vector<float> ca_coords(L * 3);
    for (int i = 0; i < L * 3; i++) {
        ca_coords[i] = static_cast<float>(i);
    }

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Check diagonal is zero
    bool pass = true;
    for (int i = 0; i < L; i++) {
        float d_ii = dist_mx[i * L + i];
        if (std::abs(d_ii) > EPS) {
            std::cout << "  ✗ Diagonal [" << i << "][" << i << "] = "
                      << d_ii << " (expected 0)\n";
            pass = false;
        }
    }

    if (pass) {
        std::cout << "  ✓ Diagonal is zero\n";
    }
    return pass;
}

bool test_distance_matrix_realistic() {
    std::cout << "\n=== Test: Distance Matrix (Realistic Protein) ===\n";

    // Simulate CA atoms along a helix
    // Helix: ~3.6 residues per turn, rise ~1.5Å per residue
    int L = 10;
    std::vector<float> ca_coords(L * 3);

    for (int i = 0; i < L; i++) {
        float t = static_cast<float>(i) * 100.0f * M_PI / 180.0f;  // 100deg rotation per residue
        float r = 2.3f;  // Helix radius
        float rise = 1.5f;  // Rise per residue

        ca_coords[i*3 + 0] = r * std::cos(t);
        ca_coords[i*3 + 1] = r * std::sin(t);
        ca_coords[i*3 + 2] = rise * i;
    }

    std::vector<float> dist_mx(L * L);
    structural_metrics::compute_distance_matrix<ScalarBackend>(
        ca_coords.data(), L, dist_mx.data()
    );

    // Check that adjacent CA atoms are ~3.8Å apart (typical CA-CA distance)
    float sum_neighbor_dist = 0.0f;
    for (int i = 0; i < L - 1; i++) {
        float d = dist_mx[i * L + (i+1)];
        sum_neighbor_dist += d;
    }
    float avg_neighbor_dist = sum_neighbor_dist / (L - 1);

    std::cout << "  Average neighbor CA-CA distance: " << avg_neighbor_dist << " Å\n";

    // Typical CA-CA distance is 3.8Å (range 3.7-3.9)
    if (avg_neighbor_dist > 3.0f && avg_neighbor_dist < 4.5f) {
        std::cout << "  ✓ Realistic CA-CA distances\n";
        return true;
    } else {
        std::cout << "  ✗ Unrealistic CA-CA distances (expected ~3.8Å)\n";
        return false;
    }
}

bool test_distance_3d_helper() {
    std::cout << "\n=== Test: Distance 3D Helper Function ===\n";

    float p1[3] = {0.0f, 0.0f, 0.0f};
    float p2[3] = {3.0f, 4.0f, 0.0f};

    float dist = structural_metrics::compute_distance_3d(p1, p2);
    float expected = 5.0f;  // 3-4-5 triangle

    if (std::abs(dist - expected) < EPS) {
        std::cout << "  ✓ Distance = " << dist << " (expected 5.0)\n";
        return true;
    } else {
        std::cout << "  ✗ Distance = " << dist << " (expected 5.0)\n";
        return false;
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Distance Matrix Tests\n";
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

    RUN_TEST(test_extract_ca_atoms);
    RUN_TEST(test_distance_matrix_simple);
    RUN_TEST(test_distance_matrix_symmetry);
    RUN_TEST(test_distance_matrix_diagonal);
    RUN_TEST(test_distance_matrix_realistic);
    RUN_TEST(test_distance_3d_helper);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
