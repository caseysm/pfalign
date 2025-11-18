/**
 * Unit tests for KNN search.
 *
 * Tests scalar implementation using nanoflann KD-tree.
 */

#include "pfalign/primitives/knn/knn_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

using pfalign::ScalarBackend;
using pfalign::knn::knn_search;
using pfalign::knn::knn_query;

constexpr float TOLERANCE = 1e-4f;

float distance_sq(const float* p1, const float* p2) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return dx * dx + dy * dy + dz * dz;
}

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * Test 1: Simple 4-point cloud
 */
bool test_simple_knn() {
    std::cout << "=== Test 1: Simple 4-Point KNN ===" << std::endl;

    // 4 points in a square
    float coords[4 * 3] = {
        0.0f, 0.0f, 0.0f,  // Point 0
        1.0f, 0.0f, 0.0f,  // Point 1
        0.0f, 1.0f, 0.0f,  // Point 2
        1.0f, 1.0f, 0.0f   // Point 3
    };

    int indices[4 * 3];      // 4 points, k=3 neighbors each
    float distances_sq[4 * 3];

    knn_search<ScalarBackend>(coords, 4, 3, indices, distances_sq);

    std::cout << "Point 0 neighbors: ";
    for (int i = 0; i < 3; i++) {
        std::cout << indices[i] << " (d^2=" << std::fixed << std::setprecision(2)
                  << distances_sq[i] << ") ";
    }
    std::cout << std::endl;

    // Point 0 should have neighbors: 0 (self), 1, 2
    // JAX approx_min_k INCLUDES self as first neighbor (distance 0)
    std::cout << "Expected: 0 (d^2=0.00) 1 (d^2=1.00) 2 (d^2=1.00)" << std::endl;

    // First neighbor should be self (distance 0)
    if (indices[0] != 0 || !close(distances_sq[0], 0.0f)) {
        std::cout << "✗ FAIL: Expected self as first neighbor (JAX behavior)" << std::endl;
        return false;
    }

    // Second and third neighbors should be at distance 1.0
    if (!close(distances_sq[1], 1.0f) || !close(distances_sq[2], 1.0f)) {
        std::cout << "✗ FAIL: Expected distances 1.0 for neighbors 1 and 2" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: Single query point
 */
bool test_single_query() {
    std::cout << "=== Test 2: Single Query Point ===" << std::endl;

    // 5 points on a line
    float coords[5 * 3] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
        4.0f, 0.0f, 0.0f
    };

    float query[3] = {2.0f, 0.0f, 0.0f};  // Same as point 2
    int indices[3];
    float distances_sq[3];

    knn_query<ScalarBackend>(coords, 5, query, 3, indices, distances_sq);

    std::cout << "Query point: (2.0, 0.0, 0.0)" << std::endl;
    std::cout << "Nearest neighbors: ";
    for (int i = 0; i < 3; i++) {
        std::cout << indices[i] << " (d^2=" << std::fixed << std::setprecision(2)
                  << distances_sq[i] << ") ";
    }
    std::cout << std::endl;

    // Should find point 2 (distance 0), then points 1 and 3 (distance 1)
    if (!close(distances_sq[0], 0.0f)) {
        std::cout << "✗ FAIL: Expected first distance to be 0.0" << std::endl;
        return false;
    }

    if (!close(distances_sq[1], 1.0f) || !close(distances_sq[2], 1.0f)) {
        std::cout << "✗ FAIL: Expected next two distances to be 1.0" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Self-inclusion in knn_search (JAX behavior)
 */
bool test_self_inclusion() {
    std::cout << "=== Test 3: Self-Inclusion (JAX behavior) ===" << std::endl;

    // 3 points
    float coords[3 * 3] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f
    };

    int indices[3 * 2];      // k=2
    float distances_sq[3 * 2];

    knn_search<ScalarBackend>(coords, 3, 2, indices, distances_sq);

    // Check that each point HAS itself as first neighbor (JAX behavior)
    bool all_passed = true;
    for (int i = 0; i < 3; i++) {
        std::cout << "Point " << i << " neighbors: ";
        for (int j = 0; j < 2; j++) {
            std::cout << indices[i * 2 + j] << " (d^2=" << distances_sq[i * 2 + j] << ") ";
        }
        std::cout << std::endl;

        // First neighbor should be self (JAX approx_min_k behavior)
        if (indices[i * 2 + 0] != i || !close(distances_sq[i * 2 + 0], 0.0f)) {
            std::cout << "✗ FAIL: Point " << i << " should have itself as first neighbor (JAX behavior)" << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "✓ PASS" << std::endl;
    }
    std::cout << std::endl;
    return all_passed;
}

/**
 * Test 4: Protein-scale test (100 Ca atoms)
 */
bool test_protein_scale() {
    std::cout << "=== Test 4: Protein-Scale (100 points, k=30) ===" << std::endl;

    // Generate 100 random points in a 50*50*50 box (typical protein dimensions)
    const int N = 100;
    const int k = 30;
    float coords[N * 3];

    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        coords[i * 3 + 0] = (rand() % 500) / 10.0f;  // 0-50 Å
        coords[i * 3 + 1] = (rand() % 500) / 10.0f;
        coords[i * 3 + 2] = (rand() % 500) / 10.0f;
    }

    int* indices = new int[N * k];
    float* distances_sq = new float[N * k];

    knn_search<ScalarBackend>(coords, N, k, indices, distances_sq);

    // Verify results for first point
    std::cout << "Point 0 neighbors (first 5):" << std::endl;
    for (int i = 0; i < 5; i++) {
        int idx = indices[i];
        float dist = std::sqrt(distances_sq[i]);
        std::cout << "  " << idx << " at " << std::fixed << std::setprecision(2)
                  << dist << " Å" << std::endl;

        // Sanity checks
        if (idx < 0 || idx >= N) {
            std::cout << "✗ FAIL: Invalid index " << idx << std::endl;
            delete[] indices;
            delete[] distances_sq;
            return false;
        }

        // First neighbor should be self (i==0 for point 0)
        if (i == 0 && idx != 0) {
            std::cout << "✗ FAIL: First neighbor should be self (JAX behavior)" << std::endl;
            delete[] indices;
            delete[] distances_sq;
            return false;
        }

        // Verify distance
        float expected_dist_sq = distance_sq(coords, coords + idx * 3);
        if (!close(distances_sq[i], expected_dist_sq, 0.01f)) {
            std::cout << "✗ FAIL: Distance mismatch. Expected " << expected_dist_sq
                      << ", got " << distances_sq[i] << std::endl;
            delete[] indices;
            delete[] distances_sq;
            return false;
        }
    }

    // Verify distances are sorted (ascending)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < k - 1; j++) {
            if (indices[i * k + j] == -1) break;  // No more valid neighbors

            if (distances_sq[i * k + j] > distances_sq[i * k + j + 1] + TOLERANCE) {
                std::cout << "✗ FAIL: Distances not sorted for point " << i << std::endl;
                delete[] indices;
                delete[] distances_sq;
                return false;
            }
        }
    }

    delete[] indices;
    delete[] distances_sq;

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 5: Edge case - k > N
 */
bool test_k_larger_than_n() {
    std::cout << "=== Test 5: k > N (Edge Case) ===" << std::endl;

    // 3 points, request k=5
    float coords[3 * 3] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f
    };

    int indices[3 * 5];
    float distances_sq[3 * 5];

    knn_search<ScalarBackend>(coords, 3, 5, indices, distances_sq);

    // Each point should have N valid neighbors (including self)
    // With N=3, k=5: expect 3 valid (0, 1, 2) + 2 sentinels (-1)
    for (int i = 0; i < 3; i++) {
        int valid_neighbors = 0;
        for (int j = 0; j < 5; j++) {
            if (indices[i * 5 + j] >= 0) {
                valid_neighbors++;
            }
        }

        std::cout << "Point " << i << " has " << valid_neighbors << " valid neighbors" << std::endl;

        if (valid_neighbors != 3) {
            std::cout << "✗ FAIL: Expected 3 valid neighbors (N=3, including self), got " << valid_neighbors << std::endl;
            return false;
        }

        // First should be self
        if (indices[i * 5 + 0] != i) {
            std::cout << "✗ FAIL: Point " << i << " first neighbor should be self" << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 6: Verify distance computation
 */
bool test_distance_accuracy() {
    std::cout << "=== Test 6: Distance Computation Accuracy ===" << std::endl;

    // Known points with known distances
    float coords[4 * 3] = {
        0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,   // Distance: 3.0
        0.0f, 4.0f, 0.0f,   // Distance: 4.0
        0.0f, 0.0f, 5.0f    // Distance: 5.0
    };

    float query[3] = {0.0f, 0.0f, 0.0f};
    int indices[3];
    float distances_sq[3];

    knn_query<ScalarBackend>(coords, 4, query, 3, indices, distances_sq);

    // Expected squared distances: 0, 9, 16, 25
    float expected_sq[3] = {0.0f, 9.0f, 16.0f};  // Closest 3 after self

    std::cout << "Query at origin, distances to (3,0,0), (0,4,0), (0,0,5):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  d^2=" << distances_sq[i] << " (expected " << expected_sq[i] << ")" << std::endl;

        if (!close(distances_sq[i], expected_sq[i])) {
            std::cout << "✗ FAIL: Distance mismatch" << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Scalar KNN Search Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 6;

    if (test_simple_knn()) passed++;
    if (test_single_query()) passed++;
    if (test_self_inclusion()) passed++;
    if (test_protein_scale()) passed++;
    if (test_k_larger_than_n()) passed++;
    if (test_distance_accuracy()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
