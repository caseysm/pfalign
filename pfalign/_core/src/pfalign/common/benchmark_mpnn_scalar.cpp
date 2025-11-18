#include "benchmark.h"
#include "arena_allocator.h"
#include <iostream>
#include <cmath>
#include <random>

using namespace pfalign::benchmark;

using namespace pfalign::memory;

// ============================================================================
// Scalar Reference Implementations (to be optimized with NEON)
// ============================================================================

/**
 * GEMM: Y = X * W^T (matrix-vector multiply)
 * X: [in_dim]
 * W: [out_dim, in_dim]
 * Y: [out_dim]
 */
void gemm_scalar(const float* X, const float* W, float* Y, size_t in_dim, size_t out_dim) {
    for (size_t i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < in_dim; j++) {
            sum += X[j] * W[i * in_dim + j];
        }
        Y[i] = sum;
    }
}

/**
 * Layer Norm: y = (x - mean) / sqrt(var + eps)
 * x: [dim]
 * y: [dim]
 */
void layer_norm_scalar(const float* x, float* y, size_t dim, float eps = 1e-5f) {
    // Compute mean
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += x[i];
    }
    float mean = sum / dim;

    // Compute variance
    float var_sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / dim;

    // Normalize
    float inv_std = 1.0f / std::sqrt(variance + eps);
    for (size_t i = 0; i < dim; i++) {
        y[i] = (x[i] - mean) * inv_std;
    }
}

/**
 * RBF Kernel: rbf[i] = exp(-gamma * dist2[i])
 * dist2: [num_neighbors] - squared distances
 * rbf: [num_neighbors] - RBF kernel values
 */
void rbf_kernel_scalar(const float* dist2, float* rbf, size_t num_neighbors, float gamma) {
    for (size_t i = 0; i < num_neighbors; i++) {
        rbf[i] = std::exp(-gamma * dist2[i]);
    }
}

/**
 * Horizontal sum reduction: sum(x)
 * x: [dim]
 * Returns: sum of all elements
 */
float reduce_sum_scalar(const float* x, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += x[i];
    }
    return sum;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

void benchmark_gemm() {
    std::cout << "\n=== GEMM Benchmark (Y = X * W^T) ===\n";

    // MPNN typical sizes: in_dim=192, out_dim=64
    constexpr size_t in_dim = 192;
    constexpr size_t out_dim = 64;

    Arena arena(1024 * 1024);

    float* X = arena.allocate<float>(in_dim);
    float* W = arena.allocate<float>(out_dim * in_dim);
    float* Y = arena.allocate<float>(out_dim);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < in_dim; i++)
        X[i] = dist(rng);
    for (size_t i = 0; i < out_dim * in_dim; i++)
        W[i] = dist(rng);

    // Benchmark
    Benchmark bench(5, 20);
    auto stats =
        bench.run("GEMM Scalar [192->64]", [&]() { gemm_scalar(X, W, Y, in_dim, out_dim); });

    stats.print("GEMM Scalar Baseline");
    std::cout << "Expected NEON target: " << std::fixed << std::setprecision(3)
              << (stats.mean_ms / 4.0) << "ms (4* speedup)\n";
}

void benchmark_layer_norm() {
    std::cout << "\n=== Layer Norm Benchmark ===\n";

    // MPNN typical size: dim=192
    constexpr size_t dim = 192;

    Arena arena(1024 * 1024);

    float* x = arena.allocate<float>(dim);
    float* y = arena.allocate<float>(dim);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dim; i++)
        x[i] = dist(rng);

    // Benchmark
    Benchmark bench(5, 20);
    auto stats = bench.run("LayerNorm Scalar [192]", [&]() { layer_norm_scalar(x, y, dim); });

    stats.print("LayerNorm Scalar Baseline");
    std::cout << "Expected NEON target: " << std::fixed << std::setprecision(3)
              << (stats.mean_ms / 3.0) << "ms (3* speedup)\n";
}

void benchmark_rbf() {
    std::cout << "\n=== RBF Kernel Benchmark ===\n";

    // MPNN typical size: 64 neighbors
    constexpr size_t num_neighbors = 64;
    constexpr float gamma = 1.0f;

    Arena arena(1024 * 1024);

    float* dist2 = arena.allocate<float>(num_neighbors);
    float* rbf = arena.allocate<float>(num_neighbors);

    // Initialize with realistic distances (0 to 100 Å^2)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (size_t i = 0; i < num_neighbors; i++)
        dist2[i] = dist(rng);

    // Benchmark
    Benchmark bench(5, 20);
    auto stats = bench.run("RBF Scalar [64 neighbors]",
                           [&]() { rbf_kernel_scalar(dist2, rbf, num_neighbors, gamma); });

    stats.print("RBF Scalar Baseline");
    std::cout << "Expected NEON target (Phase 5a): " << std::fixed << std::setprecision(3)
              << (stats.mean_ms / 2.0) << "ms (2* speedup, scalar exp fallback)\n";
    std::cout << "Expected NEON target (Phase 5b): " << std::fixed << std::setprecision(3)
              << (stats.mean_ms / 4.0) << "ms (4* speedup, vectorized exp)\n";
}

void benchmark_reduce() {
    std::cout << "\n=== Reduce Sum Benchmark ===\n";

    // MPNN typical size: 192 dimensions
    constexpr size_t dim = 192;

    Arena arena(1024 * 1024);

    float* x = arena.allocate<float>(dim);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dim; i++)
        x[i] = dist(rng);

    // Benchmark
    Benchmark bench(5, 20);
    float result = 0.0f;
    auto stats = bench.run("Reduce Scalar [192]", [&]() { result = reduce_sum_scalar(x, dim); });

    stats.print("Reduce Scalar Baseline");
    std::cout << "Expected NEON target: " << std::fixed << std::setprecision(3)
              << (stats.mean_ms / 3.0) << "ms (3* speedup with vaddvq_f32)\n";

    // Prevent optimization from removing the computation
    if (result == 0.0f)
        std::cout << "";
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "  MPNN Scalar Baseline Benchmarks\n";
    std::cout << "  (Phase 5a: ARM NEON Optimization)\n";
    std::cout << "==============================================\n";

    benchmark_gemm();
    benchmark_layer_norm();
    benchmark_rbf();
    benchmark_reduce();

    std::cout << "\n==============================================\n";
    std::cout << "  Summary\n";
    std::cout << "==============================================\n";
    std::cout << "These baselines will be used to measure NEON speedups.\n";
    std::cout << "\nNext steps:\n";
    std::cout << "  1. Implement NEON versions of each operation\n";
    std::cout << "  2. Validate against scalar (max error < 1e-5)\n";
    std::cout << "  3. Benchmark and compare speedups\n";
    std::cout << "  4. Target: 3-4* overall MPNN speedup (290ms -> 70-90ms)\n";
    std::cout << "==============================================\n";

    return 0;
}
