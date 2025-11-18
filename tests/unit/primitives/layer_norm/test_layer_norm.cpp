/**
 * Unit tests for Layer Normalization.
 *
 * Tests scalar implementation for correctness.
 */

#include "pfalign/primitives/layer_norm/layer_norm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using pfalign::ScalarBackend;
using pfalign::layer_norm::layer_norm_forward;
using pfalign::layer_norm::layer_norm_batch;
using pfalign::layer_norm::rms_norm_forward;

constexpr float TOLERANCE = 1e-4f;

bool close(float a, float b, float tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

void print_vec(const char* label, const float* vec, int D) {
    std::cout << label << ": [";
    for (int i = 0; i < std::min(D, 8); i++) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < std::min(D, 8) - 1) std::cout << ", ";
    }
    if (D > 8) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

/**
 * Test 1: Simple normalization (no gamma/beta)
 */
bool test_simple_normalization() {
    std::cout << "=== Test 1: Simple Normalization ===" << std::endl;

    // Input: [1, 2, 3, 4, 5]
    float input[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float output[5];

    // Mean = 3.0, Variance = 2.0, StdDev = sqrt(2.0) ~= 1.414
    layer_norm_forward<ScalarBackend>(input, output, nullptr, nullptr, 5);

    print_vec("Input ", input, 5);
    print_vec("Output", output, 5);

    // After normalization: mean=0, variance=1
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < 5; i++) {
        sum += output[i];
        sum_sq += output[i] * output[i];
    }
    float mean = sum / 5;
    float variance = sum_sq / 5 - mean * mean;

    std::cout << "Output mean: " << mean << " (expected ~0.0)" << std::endl;
    std::cout << "Output variance: " << variance << " (expected ~1.0)" << std::endl;

    if (!close(mean, 0.0f, 1e-5f)) {
        std::cout << "✗ FAIL: Mean not close to 0" << std::endl;
        return false;
    }

    if (!close(variance, 1.0f, 1e-4f)) {
        std::cout << "✗ FAIL: Variance not close to 1" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 2: With gamma and beta
 */
bool test_with_affine() {
    std::cout << "=== Test 2: With Gamma and Beta ===" << std::endl;

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    float gamma[4] = {2.0f, 2.0f, 2.0f, 2.0f};  // Scale by 2
    float beta[4] = {1.0f, 1.0f, 1.0f, 1.0f};   // Shift by 1

    layer_norm_forward<ScalarBackend>(input, output, gamma, beta, 4);

    print_vec("Input ", input, 4);
    print_vec("Gamma ", gamma, 4);
    print_vec("Beta  ", beta, 4);
    print_vec("Output", output, 4);

    // Check that affine transformation is applied
    // After normalization, each element is scaled by 2 and shifted by 1
    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 3: Constant input (all same values)
 */
bool test_constant_input() {
    std::cout << "=== Test 3: Constant Input ===" << std::endl;

    float input[5] = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    float output[5];

    // All values same → variance = 0 → need epsilon to avoid div by zero
    layer_norm_forward<ScalarBackend>(input, output, nullptr, nullptr, 5);

    print_vec("Input ", input, 5);
    print_vec("Output", output, 5);

    // All outputs should be 0 (mean-centered with zero variance)
    for (int i = 0; i < 5; i++) {
        if (!close(output[i], 0.0f, 1e-4f)) {
            std::cout << "✗ FAIL: output[" << i << "] = " << output[i] << ", expected 0.0" << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 4: Batch normalization
 */
bool test_batch() {
    std::cout << "=== Test 4: Batch Normalization ===" << std::endl;

    // 3 vectors of dimension 4
    float input[3 * 4] = {
        1.0f, 2.0f, 3.0f, 4.0f,  // Vector 0
        2.0f, 4.0f, 6.0f, 8.0f,  // Vector 1
        0.0f, 1.0f, 2.0f, 3.0f   // Vector 2
    };
    float output[3 * 4];
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    layer_norm_batch<ScalarBackend>(input, output, gamma, beta, 3, 4);

    // Check each vector independently
    for (int n = 0; n < 3; n++) {
        std::cout << "Vector " << n << ":" << std::endl;
        print_vec("  Input ", input + n * 4, 4);
        print_vec("  Output", output + n * 4, 4);

        // Verify mean ~= 0, variance ~= 1
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int i = 0; i < 4; i++) {
            sum += output[n * 4 + i];
            sum_sq += output[n * 4 + i] * output[n * 4 + i];
        }
        float mean = sum / 4;
        float variance = sum_sq / 4 - mean * mean;

        if (!close(mean, 0.0f, 1e-4f)) {
            std::cout << "✗ FAIL: Vector " << n << " mean = " << mean << std::endl;
            return false;
        }

        if (!close(variance, 1.0f, 1e-3f)) {
            std::cout << "✗ FAIL: Vector " << n << " variance = " << variance << std::endl;
            return false;
        }
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 5: RMS Normalization
 */
bool test_rms_norm() {
    std::cout << "=== Test 5: RMS Normalization ===" << std::endl;

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    rms_norm_forward<ScalarBackend>(input, output, gamma, 4);

    print_vec("Input ", input, 4);
    print_vec("Output", output, 4);

    // RMS = sqrt((1^2 + 2^2 + 3^2 + 4^2) / 4) = sqrt(30/4) = sqrt(7.5) ~= 2.739
    float expected_rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f);
    std::cout << "Expected RMS: " << expected_rms << std::endl;

    // Check that output has correct RMS
    float output_sum_sq = 0.0f;
    for (int i = 0; i < 4; i++) {
        output_sum_sq += output[i] * output[i];
    }
    float output_rms = std::sqrt(output_sum_sq / 4.0f);

    std::cout << "Output RMS: " << output_rms << " (expected ~1.0)" << std::endl;

    if (!close(output_rms, 1.0f, 1e-4f)) {
        std::cout << "✗ FAIL: RMS not normalized to 1.0" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

/**
 * Test 6: Protein-scale (128-dimensional embeddings)
 */
bool test_protein_scale() {
    std::cout << "=== Test 6: Protein-Scale (D=128) ===" << std::endl;

    const int D = 128;
    std::vector<float> input(D);
    std::vector<float> output(D);
    std::vector<float> gamma(D, 1.0f);
    std::vector<float> beta(D, 0.0f);

    // Generate random input
    srand(42);
    for (int i = 0; i < D; i++) {
        input[i] = (rand() % 1000) / 100.0f - 5.0f;  // Range: -5 to +5
    }

    layer_norm_forward<ScalarBackend>(
        input.data(), output.data(), gamma.data(), beta.data(), D
    );

    // Verify statistics
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        sum += output[i];
        sum_sq += output[i] * output[i];
    }
    float mean = sum / D;
    float variance = sum_sq / D - mean * mean;

    std::cout << "Input: D=" << D << " embeddings" << std::endl;
    std::cout << "Output mean: " << mean << " (expected ~0.0)" << std::endl;
    std::cout << "Output variance: " << variance << " (expected ~1.0)" << std::endl;

    if (!close(mean, 0.0f, 1e-4f)) {
        std::cout << "✗ FAIL: Mean not close to 0" << std::endl;
        return false;
    }

    if (!close(variance, 1.0f, 1e-3f)) {
        std::cout << "✗ FAIL: Variance not close to 1" << std::endl;
        return false;
    }

    std::cout << "✓ PASS" << std::endl;
    std::cout << std::endl;
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Scalar Layer Norm Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 6;

    if (test_simple_normalization()) passed++;
    if (test_with_affine()) passed++;
    if (test_constant_input()) passed++;
    if (test_batch()) passed++;
    if (test_rms_norm()) passed++;
    if (test_protein_scale()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
