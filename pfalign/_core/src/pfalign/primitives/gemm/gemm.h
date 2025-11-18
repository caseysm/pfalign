#pragma once

#include "pfalign/dispatch/execution_policy.h"

namespace pfalign {

/**
 * Public GEMM API with runtime backend dispatch
 *
 * Matrix multiply: C = alpha * A @ B + beta * C
 *
 * @param A     Input matrix [M x K]
 * @param B     Input matrix [K x N]
 * @param C     Output matrix [M x N] (in/out)
 * @param M     Number of rows in A and C
 * @param N     Number of columns in B and C
 * @param K     Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A @ B
 * @param beta  Scalar multiplier for existing C
 * @param policy Backend selection (Auto = fastest available)
 */
void gemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha = 1.0f,
          float beta = 0.0f, ExecutionPolicy policy = ExecutionPolicy::Auto);

}  // namespace pfalign
