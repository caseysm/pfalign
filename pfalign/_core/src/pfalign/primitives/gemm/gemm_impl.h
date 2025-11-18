#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace gemm {

/**
 * General Matrix Multiply (GEMM): C = alpha * A @ B + beta * C
 *
 * Used for:
 * - MPNN linear layers
 * - Profile-profile similarity
 * - Various dense operations
 *
 * Matrix dimensions:
 * - A: [M, K]
 * - B: [K, N]
 * - C: [M, N]
 *
 * Layout: Row-major (C-style)
 *
 * Template backends provide cache-blocking and vectorization.
 */

/**
 * Matrix-Matrix multiply: C = alpha * A @ B + beta * C
 *
 * @param A       Input matrix A [M * K]
 * @param B       Input matrix B [K * N]
 * @param C       Output matrix C [M * N] (also input for beta != 0)
 * @param M       Number of rows in A and C
 * @param N       Number of columns in B and C
 * @param K       Number of columns in A, rows in B
 * @param alpha   Scalar multiplier for A @ B (default: 1.0)
 * @param beta    Scalar multiplier for C (default: 0.0, i.e., overwrite)
 * @param lda     Leading dimension of A (default: K)
 * @param ldb     Leading dimension of B (default: N)
 * @param ldc     Leading dimension of C (default: N)
 */
template <typename Backend>
void gemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha = 1.0f,
          float beta = 0.0f,
          int lda = -1,  // -1 means use default (K)
          int ldb = -1,  // -1 means use default (N)
          int ldc = -1   // -1 means use default (N)
);

/**
 * Matrix-Vector multiply: y = alpha * A @ x + beta * y
 *
 * Special case of GEMM optimized for vectors.
 *
 * @param A       Matrix A [M * K]
 * @param x       Vector x [K]
 * @param y       Vector y [M] (also input for beta != 0)
 * @param M       Number of rows in A
 * @param K       Number of columns in A
 * @param alpha   Scalar multiplier for A @ x (default: 1.0)
 * @param beta    Scalar multiplier for y (default: 0.0)
 * @param lda     Leading dimension of A (default: K)
 */
template <typename Backend>
void gemv(const float* A, const float* x, float* y, int M, int K, float alpha = 1.0f,
          float beta = 0.0f, int lda = -1);

/**
 * Matrix-Matrix multiply with pre-packed B matrix: C = alpha * A @ B_pack + beta * C
 *
 * Optimized variant for static weight matrices that have been pre-packed at load time.
 * Eliminates runtime packing overhead (~10-20% speedup for weight-dominant workloads).
 *
 * @param A            Input matrix A [M * K] (unpacked, row-major)
 * @param B_packed     Pre-packed matrix B (from pack_weight_matrix())
 * @param C            Output matrix C [M * N]
 * @param M            Number of rows in A and C
 * @param N            Number of columns in B and C
 * @param K            Number of columns in A, rows in B
 * @param alpha        Scalar multiplier for A @ B (default: 1.0)
 * @param beta         Scalar multiplier for C (default: 0.0)
 * @param lda          Leading dimension of A (default: K)
 * @param ldc          Leading dimension of C (default: N)
 */
template <typename Backend>
void gemm_prepacked_rhs(const float* A, const float* B_packed, float* C, int M, int N, int K,
                        float alpha = 1.0f, float beta = 0.0f, int lda = -1, int ldc = -1);

/**
 * Batched GEMM: C[i] = alpha * A[i] @ B[i] + beta * C[i] for each i
 *
 * Used for processing multiple proteins in parallel.
 *
 * @param batch_A    Array of pointers to A matrices
 * @param batch_B    Array of pointers to B matrices
 * @param batch_C    Array of pointers to C matrices
 * @param batch_size Number of matrices in batch
 * @param M          Rows in each A and C
 * @param N          Columns in each B and C
 * @param K          Columns in each A, rows in each B
 * @param alpha      Scalar multiplier
 * @param beta       Scalar multiplier for C
 */
template <typename Backend>
void gemm_batch(const float** batch_A, const float** batch_B, float** batch_C, int batch_size,
                int M, int N, int K, float alpha = 1.0f, float beta = 0.0f);

}  // namespace gemm
}  // namespace pfalign
