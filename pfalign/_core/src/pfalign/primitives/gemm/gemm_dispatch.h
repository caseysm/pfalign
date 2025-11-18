#pragma once

#include "pfalign/dispatch/runtime_dispatch.h"

namespace pfalign {
namespace gemm {

/**
 * Runtime-dispatched GEMM functions.
 *
 * These wrap the template versions with runtime backend selection.
 * Use these from Python bindings or when backend isn't known at compile time.
 */

/**
 * Runtime GEMM: C = alpha * A @ B + beta * C
 *
 * Backend selected automatically or via set_backend().
 */
void gemm_dispatch(const float* A, const float* B, float* C, int M, int N, int K,
                   float alpha = 1.0f, float beta = 0.0f,
                   BackendType backend = BackendType::SCALAR  // Default to scalar for now
);

/**
 * Runtime GEMV: y = alpha * A @ x + beta * y
 */
void gemv_dispatch(const float* A, const float* x, float* y, int M, int K, float alpha = 1.0f,
                   float beta = 0.0f, BackendType backend = BackendType::SCALAR);

/**
 * Runtime batched GEMM
 */
void gemm_batch_dispatch(const float** batch_A, const float** batch_B, float** batch_C,
                         int batch_size, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                         BackendType backend = BackendType::SCALAR);

}  // namespace gemm
}  // namespace pfalign
