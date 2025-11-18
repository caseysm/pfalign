#include "gemm_dispatch.h"
#include "gemm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <stdexcept>

namespace pfalign {
namespace gemm {

void gemm_dispatch(const float* A, const float* B, float* C, int M, int N, int K, float alpha,
                   float beta, [[maybe_unused]] BackendType backend) {
    // Scalar-only branch: backend must be SCALAR
    gemm<ScalarBackend>(A, B, C, M, N, K, alpha, beta);
}

void gemv_dispatch(const float* A, const float* x, float* y, int M, int K, float alpha, float beta,
                   [[maybe_unused]] BackendType backend) {
    // Scalar-only branch: backend must be SCALAR
    gemv<ScalarBackend>(A, x, y, M, K, alpha, beta);
}

void gemm_batch_dispatch(const float** batch_A, const float** batch_B, float** batch_C,
                         int batch_size, int M, int N, int K, float alpha, float beta,
                         [[maybe_unused]] BackendType backend) {
    // Scalar-only branch: backend must be SCALAR
    gemm_batch<ScalarBackend>(batch_A, batch_B, batch_C, batch_size, M, N, K, alpha, beta);
}

}  // namespace gemm
}  // namespace pfalign
