#pragma once

#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/primitives/gemm/gemm_impl.h"
#include "pfalign/common/profiling.h"
#include <algorithm>
#include <cmath>

namespace pfalign {
namespace mpnn {
namespace detail {

/**
 * Compile-time cache size detection.
 * L2 cache is typically 256 KB on ARM64, 256-512 KB on x86-64.
 */
#if defined(__ARM_ARCH)
constexpr size_t L2_CACHE_SIZE = 256 * 1024;  // ARM typical: 256 KB
#elif defined(__x86_64__)
constexpr size_t L2_CACHE_SIZE = 256 * 1024;  // x86 typical: 256-512 KB
#else
constexpr size_t L2_CACHE_SIZE = 256 * 1024;  // Conservative default
#endif

/**
 * Compute optimal block size for GEMM operations.
 *
 * Calculates how many rows fit in L2 cache given the matrix dimensions.
 *
 * Working set per block:
 * - Input block: block_size * K floats
 * - Weight matrix: K * N floats (constant, stays in cache)
 * - Output block: block_size * N floats
 *
 * Total: block_size * (K + N) * 4 bytes + K * N * 4 bytes
 *
 * @param K Inner dimension (columns of A, rows of B)
 * @param N Output dimension (columns of B)
 * @return Optimal block size (aligned to 64, clamped to [64, 512])
 */
constexpr int compute_block_size(int K, int N) {
    const size_t weight_size = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
    const size_t available_cache = L2_CACHE_SIZE - weight_size;

    // Reserve 20% headroom for other data
    const size_t target_size = available_cache * 80 / 100;

    const size_t bytes_per_row = (static_cast<size_t>(K) + static_cast<size_t>(N)) * sizeof(float);
    int block_size = static_cast<int>(target_size / bytes_per_row);

    // Round down to multiple of 64 for alignment
    block_size = (block_size / 64) * 64;

    // Clamp to reasonable range [64, 512]
    return std::max(64, std::min(block_size, 512));
}

}  // namespace detail

/**
 * Cache-aware blocked GEMM implementation.
 *
 * Processes matrix multiplication in blocks that fit in L2 cache,
 * keeping the weight matrix resident to minimize DRAM traffic.
 *
 * Key optimization: Weight matrix B stays hot in L2 cache across
 * all rows in a block, instead of being evicted on every row.
 *
 * @param A Input matrix [M, K] (row-major)
 * @param B Weight matrix [K, N] (row-major, kept in cache)
 * @param C Output matrix [M, N] (row-major)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension
 * @param alpha Scalar multiplier for A @ B (default: 1.0)
 * @param beta Scalar multiplier for C (default: 0.0, i.e., overwrite)
 * @param block_size Number of rows to process per block (default: 256)
 */
template <typename Backend>
static void blocked_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f, int block_size = 256) {
    // Process in blocks for cache locality
    for (int block_start = 0; block_start < M; block_start += block_size) {
        int block_end = std::min(block_start + block_size, M);
        int block_rows = block_end - block_start;

        // Compute this block: [block_rows, K] * [K, N] -> [block_rows, N]
        // Weight matrix B stays hot in L2 cache for all block_rows
        pfalign::gemm::gemm<Backend>(A + block_start * K,  // A block start
                                     B,                    // B (entire matrix, stays in cache)
                                     C + block_start * N,  // C block start
                                     block_rows,           // M for this block
                                     N,                    // N (unchanged)
                                     K,                    // K (unchanged)
                                     alpha, beta);
    }
}

/**
 * Fused blocked GEMM with bias and GELU activation.
 *
 * Applies bias and GELU immediately after each block while data is hot in cache.
 * This fusion eliminates an extra pass through memory.
 *
 * GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * @param A Input matrix [M, K] (row-major)
 * @param B Weight matrix [K, N] (row-major)
 * @param bias Bias vector [N]
 * @param C Output matrix [M, N] (row-major)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension
 * @param block_size Number of rows to process per block (default: 256)
 */
template <typename Backend>
static void blocked_gemm_bias_gelu(const float* A, const float* B, const float* bias, float* C,
                                   int M, int N, int K, int block_size = 256) {
    constexpr float sqrt_2 = 1.41421356237f;

    for (int block_start = 0; block_start < M; block_start += block_size) {
        int block_end = std::min(block_start + block_size, M);
        int block_rows = block_end - block_start;

        // Step 1: GEMM for this block
        pfalign::gemm::gemm<Backend>(A + block_start * K, B, C + block_start * N, block_rows, N, K,
                                     1.0f, 0.0f);

        // Step 2: Bias + GELU while block is hot in cache
        float* C_block = C + block_start * N;

        for (int i = 0; i < block_rows; i++) {
            float* row = C_block + i * N;

            // Apply bias and GELU
            for (int d = 0; d < N; d++) {
                float x = row[d] + bias[d];
                row[d] = 0.5f * x * (1.0f + std::erf(x / sqrt_2));
            }
        }
    }
}

/**
 * Fused blocked GEMM with bias (no activation).
 *
 * Applies bias immediately after each block while data is hot in cache.
 * Used for final layers that don't have GELU activation.
 *
 * @param A Input matrix [M, K] (row-major)
 * @param B Weight matrix [K, N] (row-major)
 * @param bias Bias vector [N]
 * @param C Output matrix [M, N] (row-major)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension
 * @param block_size Number of rows to process per block (default: 256)
 */
template <typename Backend>
static void blocked_gemm_bias(const float* A, const float* B, const float* bias, float* C, int M,
                              int N, int K, int block_size = 256) {
    for (int block_start = 0; block_start < M; block_start += block_size) {
        int block_end = std::min(block_start + block_size, M);
        int block_rows = block_end - block_start;

        // Step 1: GEMM for this block
        pfalign::gemm::gemm<Backend>(A + block_start * K, B, C + block_start * N, block_rows, N, K,
                                     1.0f, 0.0f);

        // Step 2: Bias while block is hot in cache
        float* C_block = C + block_start * N;

        for (int i = 0; i < block_rows; i++) {
            float* row = C_block + i * N;

            // Apply bias
            for (int d = 0; d < N; d++) {
                row[d] += bias[d];
            }
        }
    }
}

// ============================================================================
// Pre-Packed Weight Variants (Phase 8: Weight Pre-Packing Optimization)
// ============================================================================

/**
 * Cache-aware blocked GEMM with pre-packed weights.
 *
 * Optimized for static weight matrices that have been pre-packed at load time.
 * Eliminates runtime pack_B_panel overhead (~10-20% speedup).
 *
 * @param A Input matrix [M, K] (row-major)
 * @param B_packed Pre-packed weight matrix (from pack_weight_matrix())
 * @param C Output matrix [M, N] (row-major)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension
 * @param alpha Scalar multiplier for A @ B (default: 1.0)
 * @param beta Scalar multiplier for C (default: 0.0)
 * @param block_size Number of rows to process per block (default: 256)
 */
template <typename Backend>
static void blocked_gemm_prepacked(const float* A, const float* B_packed, float* C, int M, int N,
                                   int K, float alpha = 1.0f, float beta = 0.0f,
                                   int block_size = 256) {
    // Process in blocks for cache locality
    for (int block_start = 0; block_start < M; block_start += block_size) {
        int block_end = std::min(block_start + block_size, M);
        int block_rows = block_end - block_start;

        // Use pre-packed GEMM variant (skips pack_B_panel)
        pfalign::gemm::gemm_prepacked_rhs<Backend>(A + block_start * K,  // A block start
                                                   B_packed,  // Pre-packed B (entire matrix)
                                                   C + block_start * N,  // C block start
                                                   block_rows,           // M for this block
                                                   N,                    // N (unchanged)
                                                   K,                    // K (unchanged)
                                                   alpha, beta);
    }
}

/**
 * Fused blocked GEMM with pre-packed weights, bias, and GELU activation.
 *
 * Eliminates runtime B-packing overhead while fusing bias and activation.
 *
 * @param A Input matrix [M, K] (row-major)
 * @param B_packed Pre-packed weight matrix
 * @param bias Bias vector [N]
 * @param C Output matrix [M, N] (row-major)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension
 * @param block_size Number of rows to process per block (default: 256)
 */
template <typename Backend>
static void blocked_gemm_prepacked_bias_gelu(const float* A, const float* B_packed,
                                             const float* bias, float* C, int M, int N, int K,
                                             int block_size = 256) {
    constexpr float sqrt_2 = 1.41421356237f;

    for (int block_start = 0; block_start < M; block_start += block_size) {
        int block_end = std::min(block_start + block_size, M);
        int block_rows = block_end - block_start;

        // Step 1: GEMM with pre-packed weights
        pfalign::gemm::gemm_prepacked_rhs<Backend>(
            A + block_start * K, B_packed, C + block_start * N, block_rows, N, K, 1.0f, 0.0f);

        // Step 2: Bias + GELU while block is hot in cache
        float* C_block = C + block_start * N;

        for (int i = 0; i < block_rows; i++) {
            float* row = C_block + i * N;

            for (int d = 0; d < N; d++) {
                float x = row[d] + bias[d];
                row[d] = 0.5f * x * (1.0f + std::erf(x / sqrt_2));
            }
        }
    }
}

/**
 * Fused blocked GEMM with pre-packed weights and bias (no activation).
 *
 * Eliminates runtime B-packing overhead while fusing bias addition.
 *
 * @param A Input matrix [M, K] (row-major)
 * @param B_packed Pre-packed weight matrix
 * @param bias Bias vector [N]
 * @param C Output matrix [M, N] (row-major)
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Inner dimension
 * @param block_size Number of rows to process per block (default: 256)
 */
template <typename Backend>
static void blocked_gemm_prepacked_bias(const float* A, const float* B_packed, const float* bias,
                                        float* C, int M, int N, int K, int block_size = 256) {
    for (int block_start = 0; block_start < M; block_start += block_size) {
        int block_end = std::min(block_start + block_size, M);
        int block_rows = block_end - block_start;

        // Step 1: GEMM with pre-packed weights
        pfalign::gemm::gemm_prepacked_rhs<Backend>(
            A + block_start * K, B_packed, C + block_start * N, block_rows, N, K, 1.0f, 0.0f);

        // Step 2: Bias while block is hot in cache
        float* C_block = C + block_start * N;

        for (int i = 0; i < block_rows; i++) {
            float* row = C_block + i * N;

            for (int d = 0; d < N; d++) {
                row[d] += bias[d];
            }
        }
    }
}

}  // namespace mpnn
}  // namespace pfalign
