/**
 * Scalar backend implementation for reduce operations.
 */

#include "reduce.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>

namespace pfalign {
namespace reduce {

//==============================================================================
// Element-wise Multiplication
//==============================================================================

template <>
void elementwise_multiply<ScalarBackend>(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] * B[i];
    }
}

//==============================================================================
// Matrix Sum
//==============================================================================

template <>
float matrix_sum<ScalarBackend>(const float* A, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += A[i];
    }
    return sum;
}

//==============================================================================
// Frobenius Norm
//==============================================================================

template <>
float frobenius_norm<ScalarBackend>(const float* A, int size) {
    float sum_squares = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_squares += A[i] * A[i];
    }
    return std::sqrt(sum_squares);
}

}  // namespace reduce
}  // namespace pfalign
