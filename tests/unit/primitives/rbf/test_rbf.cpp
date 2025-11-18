/**
 * Unit tests for RBF kernel.
 *
 * Tests scalar implementation for correctness and parity with JAX reference.
 */

#include "pfalign/primitives/rbf/rbf_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using pfalign::ScalarBackend;
using pfalign::rbf::rbf_single;
using pfalign::rbf::rbf_batch;
using pfalign::rbf::rbf_initialize_centers;
using pfalign::rbf::rbf_compute_inv_sigma_sq;

// Tolerance for floating-point comparisons
constexpr float TOLERANCE = 1e-5f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

void print_features(const char* label, const float* features, int num_bins) {
    std::cout << label << ": [";
    for (int i = 0; i < num_bins; i++) {
        std::cout << std::fixed << std::setprecision(4) << features[i];
        if (i < num_bins - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * Test 1: RBF center initialization
 */
bool test_centers() {
    std::cout << "=== Test 1: RBF Center Initialization ===" << std::endl;

    float centers[16];
    rbf_initialize_centers(centers, 16, 2.0f, 22.0f);

    // Expected: evenly spaced from 2.0 to 22.0
    // Step = (22 - 2) / (16 - 1) = 20 / 15 = 1.333...
    float expected_step = 20.0f / 15.0f;

    std::cout << "Centers: [";
    for (int i = 0; i < 16; i++) {
        std::cout << std::fixed << std::setprecision(2) << centers[i];
        if (i < 15) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Check first, middle, and last
    if (!close(centers[0], 2.0f)) {
        std::cout << "✗ FAIL: centers[0] = " << centers[0] << ", expected 2.0" << std::endl;
        return false;
    }
    if (!close(centers[15], 22.0f)) {
        std::cout << "✗ FAIL: centers[15] = " << centers[15] << ", expected 22.0" << std::endl;
        return false;
    }
    if (!close(centers[1] - centers[0], expected_step)) {
        std::cout << "✗ FAIL: step = " << (centers[1] - centers[0]) << ", expected " << expected_step << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: Single distance RBF (distance at center)
 */
bool test_single_at_center() {
    std::cout << "=== Test 2: RBF at Center (should be 1.0) ===" << std::endl;

    float centers[16];
    rbf_initialize_centers(centers, 16, 2.0f, 22.0f);
    float inv_sigma_sq = rbf_compute_inv_sigma_sq(2.0f, 22.0f, 16);

    // Test distance exactly at center[8]
    float distance = centers[8];
    float features[16];

    rbf_single<ScalarBackend>(distance, centers, inv_sigma_sq, features, 16);

    std::cout << "Distance: " << distance << " (at center 8)" << std::endl;
    print_features("Features", features, 16);

    // At the center, Gaussian should be 1.0
    if (!close(features[8], 1.0f)) {
        std::cout << "✗ FAIL: features[8] = " << features[8] << ", expected 1.0" << std::endl;
        return false;
    }

    // Other centers should be < 1.0
    for (int i = 0; i < 16; i++) {
        if (i == 8) continue;
        if (features[i] >= 1.0f) {
            std::cout << "✗ FAIL: features[" << i << "] = " << features[i] << " >= 1.0" << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Known values (manual calculation)
 */
bool test_known_values() {
    std::cout << "=== Test 3: Known RBF Values ===" << std::endl;

    // Simple setup: 3 bins, sigma = 1.0
    float centers[3] = {0.0f, 1.0f, 2.0f};
    float sigma = 1.0f;
    float inv_sigma_sq = 1.0f / (sigma * sigma);

    float distance = 1.0f;  // At center[1]
    float features[3];

    rbf_single<ScalarBackend>(distance, centers, inv_sigma_sq, features, 3);

    // Manual calculation:
    // features[0] = exp(-((1.0 - 0.0) / 1.0)^2) = exp(-1) ~= 0.3679
    // features[1] = exp(-((1.0 - 1.0) / 1.0)^2) = exp(0) = 1.0
    // features[2] = exp(-((1.0 - 2.0) / 1.0)^2) = exp(-1) ~= 0.3679

    float expected[3] = {std::exp(-1.0f), 1.0f, std::exp(-1.0f)};

    std::cout << "Distance: " << distance << std::endl;
    print_features("Features", features, 3);
    print_features("Expected", expected, 3);

    for (int i = 0; i < 3; i++) {
        if (!close(features[i], expected[i])) {
            std::cout << "✗ FAIL: features[" << i << "] = " << features[i]
                      << ", expected " << expected[i] << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 4: Batch processing
 */
bool test_batch() {
    std::cout << "=== Test 4: Batch RBF Processing ===" << std::endl;

    float centers[16];
    rbf_initialize_centers(centers, 16, 2.0f, 22.0f);
    float inv_sigma_sq = rbf_compute_inv_sigma_sq(2.0f, 22.0f, 16);

    // Test 3 distances
    float distances[3] = {5.0f, 10.0f, 15.0f};
    float features_batch[3 * 16];

    rbf_batch<ScalarBackend>(distances, centers, inv_sigma_sq, features_batch, 3, 16);

    // Compare with individual calls
    for (int n = 0; n < 3; n++) {
        float features_single[16];
        rbf_single<ScalarBackend>(distances[n], centers, inv_sigma_sq, features_single, 16);

        std::cout << "Distance " << distances[n] << ":" << std::endl;
        print_features("  Batch ", features_batch + n * 16, 16);
        print_features("  Single", features_single, 16);

        for (int i = 0; i < 16; i++) {
            if (!close(features_batch[n * 16 + i], features_single[i])) {
                std::cout << "✗ FAIL: Batch mismatch at distance " << n << ", bin " << i << std::endl;
                return false;
            }
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 5: Symmetry (distance from center should be symmetric)
 */
bool test_symmetry() {
    std::cout << "=== Test 5: Symmetry ===" << std::endl;

    float centers[16];
    rbf_initialize_centers(centers, 16, 2.0f, 22.0f);
    float inv_sigma_sq = rbf_compute_inv_sigma_sq(2.0f, 22.0f, 16);

    float center = centers[8];  // Middle center
    float offset = 2.0f;

    float features_left[16];
    float features_right[16];

    rbf_single<ScalarBackend>(center - offset, centers, inv_sigma_sq, features_left, 16);
    rbf_single<ScalarBackend>(center + offset, centers, inv_sigma_sq, features_right, 16);

    std::cout << "Center: " << center << std::endl;
    std::cout << "Left distance:  " << (center - offset) << std::endl;
    std::cout << "Right distance: " << (center + offset) << std::endl;

    // features[8] should be the same (equidistant from center[8])
    if (!close(features_left[8], features_right[8])) {
        std::cout << "✗ FAIL: features_left[8] = " << features_left[8]
                  << " != features_right[8] = " << features_right[8] << std::endl;
        return false;
    }

    std::cout << "features_left[8] = features_right[8] = " << features_left[8] << std::endl;
    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 6: Protein-scale distances (realistic test)
 */
bool test_protein_scale() {
    std::cout << "=== Test 6: Protein-Scale Distances ===" << std::endl;

    float centers[16];
    rbf_initialize_centers(centers, 16, 2.0f, 22.0f);
    float inv_sigma_sq = rbf_compute_inv_sigma_sq(2.0f, 22.0f, 16);

    // Typical Ca-Ca distances in proteins
    float distances[5] = {3.8f, 5.5f, 8.0f, 12.0f, 18.0f};
    float features[5 * 16];

    rbf_batch<ScalarBackend>(distances, centers, inv_sigma_sq, features, 5, 16);

    std::cout << "Protein Ca-Ca distances:" << std::endl;
    for (int n = 0; n < 5; n++) {
        std::cout << "  Distance " << distances[n] << " Å" << std::endl;
        print_features("    Features", features + n * 16, 16);

        // Sanity checks
        float sum = 0.0f;
        float max_val = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += features[n * 16 + i];
            max_val = std::max(max_val, features[n * 16 + i]);
        }

        std::cout << "    Sum: " << sum << ", Max: " << max_val << std::endl;

        // Max should be <= 1.0
        if (max_val > 1.0f + TOLERANCE) {
            std::cout << "✗ FAIL: max_val = " << max_val << " > 1.0" << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Scalar RBF Kernel Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 6;

    if (test_centers()) passed++;
    if (test_single_at_center()) passed++;
    if (test_known_values()) passed++;
    if (test_batch()) passed++;
    if (test_symmetry()) passed++;
    if (test_protein_scale()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
