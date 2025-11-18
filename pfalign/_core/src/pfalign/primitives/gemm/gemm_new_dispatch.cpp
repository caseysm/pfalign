#include "pfalign/primitives/gemm/gemm.h"
#include "pfalign/dispatch/execution_policy.h"
#include "pfalign/dispatch/backend_traits.h"
#include "gemm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <stdexcept>

namespace pfalign {

// Forward declare scalar implementation (from gemm_scalar.cpp)
// We'll need to create a wrapper or use the existing pfalign::gemm::gemm<ScalarBackend>

void gemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta,
          ExecutionPolicy policy) {
    // Scalar-only branch: policy is always Scalar
    // (Auto resolves to Scalar automatically)
    pfalign::gemm::gemm<ScalarBackend>(A, B, C, M, N, K, alpha, beta);
}

}  // namespace pfalign
