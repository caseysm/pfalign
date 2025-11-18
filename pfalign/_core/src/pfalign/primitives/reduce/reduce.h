/**
 * Reduce operations for matrix/array computations.
 *
 * Provides fundamental reduction operations used in alignment scoring:
 * - Element-wise multiplication
 * - Matrix sum (reduce to scalar)
 * - Frobenius norm (L2 norm)
 *
 * All operations support backend dispatch (Scalar, NEON, CUDA).
 */

#pragma once

#include <cmath>

namespace pfalign {
namespace reduce {

/**
 * Element-wise multiplication: C[i] = A[i] * B[i]
 *
 * Multiplies two arrays element-wise and stores result in C.
 *
 * @tparam Backend Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param A First input array [size]
 * @param B Second input array [size]
 * @param C Output array [size] (caller allocated)
 * @param size Number of elements
 *
 * Example:
 *   float A[4] = {1, 2, 3, 4};
 *   float B[4] = {2, 3, 4, 5};
 *   float C[4];
 *   elementwise_multiply<ScalarBackend>(A, B, C, 4);
 *   // C = {2, 6, 12, 20}
 */
template <typename Backend>
void elementwise_multiply(const float* A, const float* B, float* C, int size);

/**
 * Sum all elements in array: returns sum(A[i])
 *
 * Computes the sum of all elements in the input array.
 *
 * @tparam Backend Computation backend
 * @param A Input array [size]
 * @param size Number of elements
 * @return Sum of all elements
 *
 * Example:
 *   float A[4] = {1, 2, 3, 4};
 *   float sum = matrix_sum<ScalarBackend>(A, 4);
 *   // sum = 10.0
 */
template <typename Backend>
float matrix_sum(const float* A, int size);

/**
 * Frobenius norm: sqrt(sum(A[i]^2))
 *
 * Computes the Frobenius (L2) norm of an array.
 * This is the square root of the sum of squared elements.
 *
 * @tparam Backend Computation backend
 * @param A Input array [size]
 * @param size Number of elements
 * @return Frobenius norm
 *
 * Example:
 *   float A[3] = {3, 4, 0};
 *   float norm = frobenius_norm<ScalarBackend>(A, 3);
 *   // norm = sqrt(9 + 16 + 0) = sqrt(25) = 5.0
 *
 * Note: For matrices, treat as flattened 1D array.
 *       ||M||_F = sqrt(sum_{i,j} M[i,j]^2)
 */
template <typename Backend>
float frobenius_norm(const float* A, int size);

}  // namespace reduce
}  // namespace pfalign
