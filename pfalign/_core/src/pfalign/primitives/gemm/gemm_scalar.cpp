#include "gemm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/common/tuning.h"
#include <algorithm>
#include <cstring>

namespace pfalign {
namespace gemm {

/**
 * Scalar GEMM implementation with cache blocking.
 *
 * Strategy:
 * - Block matrices to fit in L1 cache (tunable block sizes)
 * - Use register blocking for inner kernel (tunable microkernel)
 * - Optimize for row-major layout
 *
 * Performance: ~2-4 GFLOPS on modern CPU (single-threaded)
 *
 * Tuning:
 * - Set SOFTALIGN_GEMM_MC, NC, KC, MR, NR via env vars
 * - Or define at compile time with CMake
 */

/**
 * 4*4 microkernel (innermost computation).
 *
 * Computes C_block[4*4] += A_panel[4*k] @ B_panel[k*4]
 * All data fits in registers (16 floats for C + inputs)
 */
static inline void gemm_microkernel_4x4(const float* A, const float* B, float* C, int K, int lda,
                                        int ldb, int ldc, float alpha, float beta) {
    // Accumulate in registers
    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

    // Accumulate: C[4*4] += A[4*K] @ B[K*4]
    for (int k = 0; k < K; k++) {
        // Load A column
        float a0 = A[0 * lda + k];
        float a1 = A[1 * lda + k];
        float a2 = A[2 * lda + k];
        float a3 = A[3 * lda + k];

        // Load B row
        float b0 = B[k * ldb + 0];
        float b1 = B[k * ldb + 1];
        float b2 = B[k * ldb + 2];
        float b3 = B[k * ldb + 3];

        // Compute outer product
        c00 += a0 * b0;
        c01 += a0 * b1;
        c02 += a0 * b2;
        c03 += a0 * b3;
        c10 += a1 * b0;
        c11 += a1 * b1;
        c12 += a1 * b2;
        c13 += a1 * b3;
        c20 += a2 * b0;
        c21 += a2 * b1;
        c22 += a2 * b2;
        c23 += a2 * b3;
        c30 += a3 * b0;
        c31 += a3 * b1;
        c32 += a3 * b2;
        c33 += a3 * b3;
    }

    // Scale and accumulate into C
    if (beta == 0.0f) {
        // Overwrite C (common case)
        C[0 * ldc + 0] = alpha * c00;
        C[0 * ldc + 1] = alpha * c01;
        C[0 * ldc + 2] = alpha * c02;
        C[0 * ldc + 3] = alpha * c03;
        C[1 * ldc + 0] = alpha * c10;
        C[1 * ldc + 1] = alpha * c11;
        C[1 * ldc + 2] = alpha * c12;
        C[1 * ldc + 3] = alpha * c13;
        C[2 * ldc + 0] = alpha * c20;
        C[2 * ldc + 1] = alpha * c21;
        C[2 * ldc + 2] = alpha * c22;
        C[2 * ldc + 3] = alpha * c23;
        C[3 * ldc + 0] = alpha * c30;
        C[3 * ldc + 1] = alpha * c31;
        C[3 * ldc + 2] = alpha * c32;
        C[3 * ldc + 3] = alpha * c33;
    } else {
        // Accumulate: C = alpha * AB + beta * C
        C[0 * ldc + 0] = alpha * c00 + beta * C[0 * ldc + 0];
        C[0 * ldc + 1] = alpha * c01 + beta * C[0 * ldc + 1];
        C[0 * ldc + 2] = alpha * c02 + beta * C[0 * ldc + 2];
        C[0 * ldc + 3] = alpha * c03 + beta * C[0 * ldc + 3];

        C[1 * ldc + 0] = alpha * c10 + beta * C[1 * ldc + 0];
        C[1 * ldc + 1] = alpha * c11 + beta * C[1 * ldc + 1];
        C[1 * ldc + 2] = alpha * c12 + beta * C[1 * ldc + 2];
        C[1 * ldc + 3] = alpha * c13 + beta * C[1 * ldc + 3];

        C[2 * ldc + 0] = alpha * c20 + beta * C[2 * ldc + 0];
        C[2 * ldc + 1] = alpha * c21 + beta * C[2 * ldc + 1];
        C[2 * ldc + 2] = alpha * c22 + beta * C[2 * ldc + 2];
        C[2 * ldc + 3] = alpha * c23 + beta * C[2 * ldc + 3];

        C[3 * ldc + 0] = alpha * c30 + beta * C[3 * ldc + 0];
        C[3 * ldc + 1] = alpha * c31 + beta * C[3 * ldc + 1];
        C[3 * ldc + 2] = alpha * c32 + beta * C[3 * ldc + 2];
        C[3 * ldc + 3] = alpha * c33 + beta * C[3 * ldc + 3];
    }
}

/**
 * Scalar GEMM: C = alpha * A @ B + beta * C
 */
template <>
void gemm<ScalarBackend>(const float* A, const float* B, float* C, int M, int N, int K, float alpha,
                         float beta, int lda, int ldb, int ldc) {
    // Default leading dimensions
    if (lda < 0)
        lda = K;
    if (ldb < 0)
        ldb = N;
    if (ldc < 0)
        ldc = N;

    // Get tuned blocking parameters
    auto tuning = tuning::GEMMTuning::get_for_size(M, N, K);
    const int MC = tuning.MC;
    const int NC = tuning.NC;
    const int KC = tuning.KC;
    const int MR = tuning.MR;
    const int NR = tuning.NR;

    // Cache-blocked GEMM
    // Note: beta is handled in the first k-block iteration
    for (int jj = 0; jj < N; jj += NC) {
        int j_block = std::min(NC, N - jj);

        for (int kk = 0; kk < K; kk += KC) {
            int k_block = std::min(KC, K - kk);

            for (int ii = 0; ii < M; ii += MC) {
                int i_block = std::min(MC, M - ii);

                // Block GEMM: C[ii:ii+i_block, jj:jj+j_block] += A[ii:ii+i_block, kk:kk+k_block] @
                // B[kk:kk+k_block, jj:jj+j_block]
                const float* A_block = A + ii * lda + kk;
                const float* B_block = B + kk * ldb + jj;
                float* C_block = C + ii * ldc + jj;

                // Use 4*4 microkernel for the bulk
                int i = 0;
                for (; i + MR <= i_block; i += MR) {
                    int j = 0;
                    for (; j + NR <= j_block; j += NR) {
                        gemm_microkernel_4x4(
                            A_block + i * lda, B_block + j, C_block + i * ldc + j, k_block, lda,
                            ldb, ldc, alpha,
                            (kk == 0) ? beta : 1.0f  // First k-block uses beta, rest accumulate
                        );
                    }

                    // Handle remaining columns (j < NR)
                    for (; j < j_block; j++) {
                        for (int i_inner = 0; i_inner < MR; i_inner++) {
                            float sum = 0.0f;
                            for (int k_inner = 0; k_inner < k_block; k_inner++) {
                                sum += A_block[(i + i_inner) * lda + k_inner] *
                                       B_block[k_inner * ldb + j];
                            }
                            if (kk == 0) {
                                C_block[(i + i_inner) * ldc + j] =
                                    alpha * sum + beta * C_block[(i + i_inner) * ldc + j];
                            } else {
                                C_block[(i + i_inner) * ldc + j] += alpha * sum;
                            }
                        }
                    }
                }

                // Handle remaining rows (i < MR)
                for (; i < i_block; i++) {
                    for (int j = 0; j < j_block; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < k_block; k++) {
                            sum += A_block[i * lda + k] * B_block[k * ldb + j];
                        }
                        if (kk == 0) {
                            C_block[i * ldc + j] = alpha * sum + beta * C_block[i * ldc + j];
                        } else {
                            C_block[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    }
}

/**
 * Scalar GEMV: y = alpha * A @ x + beta * y
 */
template <>
void gemv<ScalarBackend>(const float* A, const float* x, float* y, int M, int K, float alpha,
                         float beta, int lda) {
    if (lda < 0)
        lda = K;

    // Matrix-vector multiply: y = alpha * A @ x + beta * y
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * lda + k] * x[k];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

/**
 * Batched GEMM: Process multiple matrices
 */
template <>
void gemm_batch<ScalarBackend>(const float** batch_A, const float** batch_B, float** batch_C,
                               int batch_size, int M, int N, int K, float alpha, float beta) {
    // Simple sequential processing (no parallelism in scalar version)
    for (int b = 0; b < batch_size; b++) {
        gemm<ScalarBackend>(batch_A[b], batch_B[b], batch_C[b], M, N, K, alpha, beta, K, N,
                            N  // Default leading dimensions
        );
    }
}

}  // namespace gemm
}  // namespace pfalign
