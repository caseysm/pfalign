/**
 * Scalar implementation of similarity computation.
 */

#include "similarity.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/primitives/gemm/gemm_impl.h"

namespace pfalign {
namespace similarity {

/**
 * Compute similarity matrix as S = emb1 * emb2^T
 *
 * GEMM formulation:
 * - A = emb1 [L1 * D]
 * - B = emb2 [L2 * D]
 * - C = similarity [L1 * L2]
 * - Operation: C = A * B^T
 *
 * This requires transposing B, which we handle by swapping dimensions
 * in the GEMM call.
 */
template <>
void compute_similarity<ScalarBackend>(const float* emb1, const float* emb2, float* similarity,
                                       int L1, int L2, int D) {
    // Compute S = emb1 * emb2^T using GEMM
    //
    // Standard GEMM: C = A * B where A is [M * K], B is [K * N], C is [M * N]
    // We want: S = emb1 * emb2^T
    //   emb1 is [L1 * D] -> treat as A with M=L1, K=D
    //   emb2^T is [D * L2] -> we have emb2 as [L2 * D], need to transpose
    //   similarity is [L1 * L2] -> C with M=L1, N=L2
    //
    // Our GEMM signature handles this by treating emb2 as row-major [L2 * D]
    // and computing the transpose implicitly.

    // For simplicity, let's do a naive implementation first
    // (Can optimize to use GEMM with transpose later)

    for (int i = 0; i < L1; i++) {
        for (int j = 0; j < L2; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += emb1[i * D + d] * emb2[j * D + d];
            }
            similarity[i * L2 + j] = dot;
        }
    }
}

}  // namespace similarity
}  // namespace pfalign
