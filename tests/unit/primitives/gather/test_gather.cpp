/**
 * Unit tests for Gather operations.
 *
 * Tests scalar implementation for correctness.
 */

#include "pfalign/primitives/gather/gather_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <iomanip>
#include <cstring>

using pfalign::ScalarBackend;
using pfalign::gather::gather;
using pfalign::gather::gather_edges;
using pfalign::gather::scatter_add;

constexpr float TOLERANCE = 1e-5f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * Test 1: Basic gather
 */
bool test_basic_gather() {
    std::cout << "=== Test 1: Basic Gather ===" << std::endl;

    // 4 embeddings of dimension 3
    float embeddings[4 * 3] = {
        1.0f, 2.0f, 3.0f,   // Embedding 0
        4.0f, 5.0f, 6.0f,   // Embedding 1
        7.0f, 8.0f, 9.0f,   // Embedding 2
        10.0f, 11.0f, 12.0f // Embedding 3
    };

    // Gather indices: 2 queries, each with 2 neighbors
    int indices[2 * 2] = {
        1, 3,  // Query 0: neighbors 1, 3
        0, 2   // Query 1: neighbors 0, 2
    };

    float output[2 * 2 * 3];  // 2 queries * 2 neighbors * 3 dim

    gather<ScalarBackend>(embeddings, indices, output, 2, 2, 3);

    // Verify query 0, neighbor 0 (should be embedding 1)
    std::cout << "Query 0, neighbor 0: [";
    for (int i = 0; i < 3; i++) {
        std::cout << output[0 * 2 * 3 + 0 * 3 + i];
        if (i < 2) std::cout << ", ";
    }
    std::cout << "] (expected [4, 5, 6])" << std::endl;

    if (!close(output[0], 4.0f) || !close(output[1], 5.0f) || !close(output[2], 6.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    // Verify query 1, neighbor 1 (should be embedding 2)
    int idx = 1 * 2 * 3 + 1 * 3;  // Query 1, neighbor 1
    if (!close(output[idx], 7.0f) || !close(output[idx+1], 8.0f) || !close(output[idx+2], 9.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: Gather with invalid indices
 */
bool test_gather_invalid_indices() {
    std::cout << "=== Test 2: Gather with Invalid Indices ===" << std::endl;

    float embeddings[3 * 2] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    // Include -1 (invalid index marker)
    int indices[2 * 3] = {
        0, 1, -1,  // Query 0: valid, valid, invalid
        2, -1, 1   // Query 1: valid, invalid, valid
    };

    float output[2 * 3 * 2];

    gather<ScalarBackend>(embeddings, indices, output, 2, 3, 2);

    // Check that invalid indices are filled with zeros
    int invalid_idx = 0 * 3 * 2 + 2 * 2;  // Query 0, neighbor 2
    if (!close(output[invalid_idx], 0.0f) || !close(output[invalid_idx+1], 0.0f)) {
        std::cout << "✗ FAIL: Invalid index not zero-filled" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Gather edges (concatenation)
 */
bool test_gather_edges() {
    std::cout << "=== Test 3: Gather Edges ===" << std::endl;

    // 3 embeddings of dimension 2
    float embeddings[3 * 2] = {
        1.0f, 2.0f,   // Embedding 0
        3.0f, 4.0f,   // Embedding 1
        5.0f, 6.0f    // Embedding 2
    };

    // Query 0 with 2 neighbors
    int indices[1 * 2] = {1, 2};

    float output[1 * 2 * 4];  // 1 query * 2 neighbors * (2*2) dim

    gather_edges<ScalarBackend>(embeddings, indices, output, 1, 2, 2);

    // Edge 0: [query_0, neighbor_1] = [1, 2, 3, 4]
    std::cout << "Edge 0: [";
    for (int i = 0; i < 4; i++) {
        std::cout << output[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "] (expected [1, 2, 3, 4])" << std::endl;

    if (!close(output[0], 1.0f) || !close(output[1], 2.0f) ||
        !close(output[2], 3.0f) || !close(output[3], 4.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    // Edge 1: [query_0, neighbor_2] = [1, 2, 5, 6]
    if (!close(output[4], 1.0f) || !close(output[5], 2.0f) ||
        !close(output[6], 5.0f) || !close(output[7], 6.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 4: Scatter-add
 */
bool test_scatter_add() {
    std::cout << "=== Test 4: Scatter-Add ===" << std::endl;

    // Input: 2 sources, each with 2 values of dimension 2
    float input[2 * 2 * 2] = {
        1.0f, 2.0f,   // Source 0, value 0
        3.0f, 4.0f,   // Source 0, value 1
        5.0f, 6.0f,   // Source 1, value 0
        7.0f, 8.0f    // Source 1, value 1
    };

    // Scatter to indices
    int indices[2 * 2] = {
        0, 1,  // Source 0 → targets 0, 1
        1, 2   // Source 1 → targets 1, 2
    };

    float output[3 * 2];  // 3 targets * 2 dim
    std::memset(output, 0, sizeof(output));

    scatter_add<ScalarBackend>(input, indices, output, 2, 2, 2, 3);

    // Target 0 should have: [1, 2] (from source 0, value 0)
    std::cout << "Target 0: [" << output[0] << ", " << output[1] << "] (expected [1, 2])" << std::endl;

    if (!close(output[0], 1.0f) || !close(output[1], 2.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    // Target 1 should have: [3, 4] + [5, 6] = [8, 10]
    std::cout << "Target 1: [" << output[2] << ", " << output[3] << "] (expected [8, 10])" << std::endl;

    if (!close(output[2], 8.0f) || !close(output[3], 10.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    // Target 2 should have: [7, 8]
    std::cout << "Target 2: [" << output[4] << ", " << output[5] << "] (expected [7, 8])" << std::endl;

    if (!close(output[4], 7.0f) || !close(output[5], 8.0f)) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 5: Protein-scale gather
 */
bool test_protein_scale() {
    std::cout << "=== Test 5: Protein-Scale Gather ===" << std::endl;

    const int N = 100;  // 100 residues
    const int k = 30;   // 30 neighbors
    const int D = 128;  // 128-dim embeddings

    float* embeddings = new float[N * D];
    int* indices = new int[N * k];
    float* output = new float[N * k * D];

    // Initialize embeddings with unique values
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < D; d++) {
            embeddings[i * D + d] = i * 100.0f + d;
        }
    }

    // Initialize indices (each point's neighbors are i+1, i+2, ..., i+k)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < k; j++) {
            indices[i * k + j] = (i + j + 1) % N;
        }
    }

    gather<ScalarBackend>(embeddings, indices, output, N, k, D);

    // Verify: output[0, 5, :] should equal embeddings[6, :]
    int test_idx = 0 * k * D + 5 * D;
    int expected_idx = 6 * D;

    bool matches = true;
    for (int d = 0; d < D; d++) {
        if (!close(output[test_idx + d], embeddings[expected_idx + d])) {
            matches = false;
            break;
        }
    }

    std::cout << "Verified: output[0, 5, :] == embeddings[6, :]: " << (matches ? "YES" : "NO") << std::endl;

    delete[] embeddings;
    delete[] indices;
    delete[] output;

    if (!matches) {
        std::cout << "✗ FAIL" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Scalar Gather Operations Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 5;

    if (test_basic_gather()) passed++;
    if (test_gather_invalid_indices()) passed++;
    if (test_gather_edges()) passed++;
    if (test_scatter_add()) passed++;
    if (test_protein_scale()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
