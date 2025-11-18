/**
 * 3*3 Singular Value Decomposition
 *
 * Computes SVD of 3*3 matrices using Jacobi iteration.
 * Used for Kabsch algorithm in protein structure alignment.
 *
 * Algorithm: Cyclic Jacobi iteration with Givens rotations
 * - Iteratively applies rotations to zero off-diagonal elements
 * - Converges to diagonal form (singular values)
 * - Accumulates rotations to recover U and V matrices
 *
 * Performance:
 * - Scalar: ~10-20 iterations, ~1 mus
 * - NEON: Future optimization, ~500 ns
 * - CUDA: Overkill for 3*3, use scalar
 *
 * Accuracy:
 * - Orthogonality: ||U Uᵀ - I||_F < 1e-5
 * - Reconstruction: ||A - U Sigma Vᵀ||_F < 1e-5
 */

#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace svd3x3 {

/**
 * Compute 3*3 Singular Value Decomposition.
 *
 * Decomposes A = U Sigma Vᵀ where:
 * - U [3*3]: Left singular vectors (orthogonal: U Uᵀ = I)
 * - Sigma [3]: Singular values (diagonal, descending: Sigma₀ >= Sigma₁ >= Sigma₂ >= 0)
 * - V [3*3]: Right singular vectors (orthogonal: V Vᵀ = I)
 *
 * All matrices stored row-major:
 * - A[0..8] = [a₀₀ a₀₁ a₀₂ a₁₀ a₁₁ a₁₂ a₂₀ a₂₁ a₂₂]
 *
 * Algorithm:
 * 1. Initialize U, V to identity matrices
 * 2. For each iteration:
 *    a. Apply Givens rotation to zero A[0,1]
 *    b. Apply Givens rotation to zero A[0,2]
 *    c. Apply Givens rotation to zero A[1,2]
 *    d. Accumulate rotations in U, V
 * 3. Stop when ||off-diagonal|| < tolerance
 * 4. Extract diagonal as singular values
 * 5. Sort singular values descending (permute U, V columns)
 *
 * @tparam Backend Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param A Input matrix [9] (3*3 row-major). NOTE: Will be modified during computation.
 *          If you need to preserve A, copy it before calling.
 * @param U Output left singular vectors [9] (3*3 row-major, orthogonal)
 * @param S Output singular values [3] (descending order)
 * @param V Output right singular vectors [9] (3*3 row-major, orthogonal)
 * @param max_iterations Maximum number of Jacobi sweeps (default: 30)
 *                       Typically converges in 10-20 iterations for normal matrices.
 * @param tolerance Convergence criterion for off-diagonal norm (default: 1e-6)
 *                  Iteration stops when ||off-diagonal||_inf < tolerance
 *
 * Example usage:
 * ```cpp
 *   // Input matrix (row-major)
 *   float A[9] = {1, 2, 3,
 *                 4, 5, 6,
 *                 7, 8, 9};
 *
 *   float U[9], S[3], V[9];
 *
 *   svd3x3<ScalarBackend>(A, U, S, V);
 *
 *   // Result: A ~= U @ diag(S) @ Vᵀ
 *   // U Uᵀ ~= I, V Vᵀ ~= I
 *   // S[0] >= S[1] >= S[2] >= 0
 * ```
 *
 * Numerical properties:
 * - Handles rank-deficient matrices (singular values -> 0)
 * - Stable for condition numbers up to ~1e6
 * - Guarantees orthogonality of U, V to machine precision
 *
 * Future SIMD optimizations:
 * - NEON: Vectorize 3*3 matrix operations (4-wide with padding)
 * - AVX2: Similar vectorization strategy
 * - CUDA: Not recommended (overhead > benefit for 3*3)
 */
template <typename Backend>
void svd3x3(float* A,                 // [9] input/working matrix (row-major, will be modified)
            float* U,                 // [9] left singular vectors output
            float* S,                 // [3] singular values output
            float* V,                 // [9] right singular vectors output
            int max_iterations = 30,  // maximum Jacobi sweeps
            float tolerance = 1e-6f   // convergence threshold
);

}  // namespace svd3x3
}  // namespace pfalign
