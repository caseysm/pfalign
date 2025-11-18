/**
 * Unit tests for similarity computation.
 */

#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

using pfalign::ScalarBackend;
using pfalign::similarity::compute_similarity;

constexpr float TOLERANCE = 1e-4f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * Test 1: Simple 2*2 case with known values.
 */
bool test_simple() {
    std::cout << "=== Test 1: Simple 2*2 ===" << std::endl;

    // emb1: [[1, 0], [0, 1]]
    float emb1[4] = {1.0f, 0.0f, 0.0f, 1.0f};

    // emb2: [[1, 0], [0, 1]]
    float emb2[4] = {1.0f, 0.0f, 0.0f, 1.0f};

    float similarity[4];

    compute_similarity<ScalarBackend>(emb1, emb2, similarity, 2, 2, 2);

    // Expected: [[1, 0], [0, 1]] (identity matrix)
    float expected[4] = {1.0f, 0.0f, 0.0f, 1.0f};

    bool passed = true;
    for (int i = 0; i < 4; i++) {
        if (!close(similarity[i], expected[i])) {
            std::cout << "Mismatch at index " << i << ": "
                      << similarity[i] << " vs " << expected[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

/**
 * Test 2: Orthogonal vectors should give zero similarity.
 */
bool test_orthogonal() {
    std::cout << "=== Test 2: Orthogonal Vectors ===" << std::endl;

    // emb1: [[1, 0, 0]]
    float emb1[3] = {1.0f, 0.0f, 0.0f};

    // emb2: [[0, 1, 0], [0, 0, 1]]
    float emb2[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    float similarity[2];

    compute_similarity<ScalarBackend>(emb1, emb2, similarity, 1, 2, 3);

    // Expected: [0, 0]
    bool passed = close(similarity[0], 0.0f) && close(similarity[1], 0.0f);

    std::cout << "Similarity: [" << similarity[0] << ", " << similarity[1] << "]" << std::endl;

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

/**
 * Test 3: Identical embeddings should give perfect similarity.
 */
bool test_identical() {
    std::cout << "=== Test 3: Identical Embeddings ===" << std::endl;

    const int L = 5;
    const int D = 8;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float emb[L * D];
    for (int i = 0; i < L * D; i++) {
        emb[i] = dist(rng);
    }

    float similarity[L * L];

    compute_similarity<ScalarBackend>(emb, emb, similarity, L, L, D);

    // Diagonal should be dot(v, v) for each embedding
    // Off-diagonal should be symmetric
    bool passed = true;

    for (int i = 0; i < L; i++) {
        // Diagonal: compute expected dot product
        float expected_diag = 0.0f;
        for (int d = 0; d < D; d++) {
            expected_diag += emb[i * D + d] * emb[i * D + d];
        }

        if (!close(similarity[i * L + i], expected_diag, 1e-3f)) {
            std::cout << "Diagonal mismatch at " << i << ": "
                      << similarity[i * L + i] << " vs " << expected_diag << std::endl;
            passed = false;
        }
    }

    // Check symmetry
    for (int i = 0; i < L; i++) {
        for (int j = i + 1; j < L; j++) {
            if (!close(similarity[i * L + j], similarity[j * L + i], 1e-5f)) {
                std::cout << "Symmetry broken at (" << i << "," << j << ")" << std::endl;
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return passed;
}

/**
 * Test 4: Protein-scale similarity (100*150).
 */
bool test_protein_scale() {
    std::cout << "=== Test 4: Protein-Scale (100*150) ===" << std::endl;

    const int L1 = 100;
    const int L2 = 150;
    const int D = 128;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* emb1 = new float[L1 * D];
    float* emb2 = new float[L2 * D];

    for (int i = 0; i < L1 * D; i++) emb1[i] = dist(rng);
    for (int i = 0; i < L2 * D; i++) emb2[i] = dist(rng);

    float* similarity = new float[L1 * L2];

    compute_similarity<ScalarBackend>(emb1, emb2, similarity, L1, L2, D);

    // Check output shape and values are reasonable
    bool all_finite = true;
    float min_sim = similarity[0];
    float max_sim = similarity[0];

    for (int i = 0; i < L1 * L2; i++) {
        if (!std::isfinite(similarity[i])) {
            all_finite = false;
            break;
        }
        min_sim = std::min(min_sim, similarity[i]);
        max_sim = std::max(max_sim, similarity[i]);
    }

    std::cout << "Similarity range: [" << min_sim << ", " << max_sim << "]" << std::endl;
    std::cout << "All finite: " << (all_finite ? "YES" : "NO") << std::endl;

    // For random normalized vectors, expect similarity in range [-D, D]
    bool reasonable = all_finite && std::abs(min_sim) < 50.0f && std::abs(max_sim) < 50.0f;

    delete[] emb1;
    delete[] emb2;
    delete[] similarity;

    if (reasonable) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return reasonable;
}

/**
 * Test 5: Deterministic output.
 */
bool test_deterministic() {
    std::cout << "=== Test 5: Deterministic Output ===" << std::endl;

    const int L1 = 20;
    const int L2 = 30;
    const int D = 64;

    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float* emb1 = new float[L1 * D];
    float* emb2 = new float[L2 * D];

    for (int i = 0; i < L1 * D; i++) emb1[i] = dist(rng);
    for (int i = 0; i < L2 * D; i++) emb2[i] = dist(rng);

    float* sim1 = new float[L1 * L2];
    float* sim2 = new float[L1 * L2];

    // Compute twice
    compute_similarity<ScalarBackend>(emb1, emb2, sim1, L1, L2, D);
    compute_similarity<ScalarBackend>(emb1, emb2, sim2, L1, L2, D);

    // Check identical
    bool identical = true;
    for (int i = 0; i < L1 * L2; i++) {
        if (!close(sim1[i], sim2[i], 1e-6f)) {
            std::cout << "Mismatch at index " << i << ": "
                      << sim1[i] << " vs " << sim2[i] << std::endl;
            identical = false;
            break;
        }
    }

    std::cout << "Outputs identical: " << (identical ? "YES" : "NO") << std::endl;

    delete[] emb1;
    delete[] emb2;
    delete[] sim1;
    delete[] sim2;

    if (identical) {
        std::cout << "✓ PASS" << std::endl;
    } else {
        std::cout << "✗ FAIL" << std::endl;
    }

    std::cout << std::endl;
    return identical;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Similarity Computation Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 5;

    if (test_simple()) passed++;
    if (test_orthogonal()) passed++;
    if (test_identical()) passed++;
    if (test_protein_scale()) passed++;
    if (test_deterministic()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
