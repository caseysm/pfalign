#include "gather_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cstring>

namespace pfalign {
namespace gather {

/**
 * Scalar Gather implementation.
 *
 * Simple index-based memory copy operations.
 * Cache-friendly for small k (typical: k=30).
 *
 * Performance:
 *   - gather: ~5 ns per element (D=128)
 *   - gather_edges: ~10 ns per element (2D=256)
 *   - scatter_add: ~8 ns per element (atomic not needed for scalar)
 */

/**
 * Gather embeddings from indices.
 */
template <>
void gather<ScalarBackend>(const float* embeddings, const int* indices, float* output, int M, int k,
                           int D) {
    // For each query point
    for (int i = 0; i < M; i++) {
        // For each neighbor
        for (int j = 0; j < k; j++) {
            int idx = indices[i * k + j];

            // Skip invalid indices (-1 marker for k > N)
            if (idx < 0) {
                // Fill with zeros
                std::memset(output + (i * k + j) * D, 0, static_cast<size_t>(D) * sizeof(float));
                continue;
            }

            // Copy embedding: output[i,j,:] = embeddings[idx,:]
            std::memcpy(output + (i * k + j) * D, embeddings + idx * D,
                        static_cast<size_t>(D) * sizeof(float));
        }
    }
}

/**
 * Gather edges (concatenate query and neighbor embeddings).
 */
template <>
void gather_edges<ScalarBackend>(const float* embeddings, const int* indices, float* output, int M,
                                 int k, int D) {
    // For each query point
    for (int i = 0; i < M; i++) {
        const float* query_emb = embeddings + i * D;

        // For each neighbor
        for (int j = 0; j < k; j++) {
            int neighbor_idx = indices[i * k + j];
            float* edge_emb = output + (i * k + j) * (2 * D);

            // First half: query embedding
            std::memcpy(edge_emb, query_emb, static_cast<size_t>(D) * sizeof(float));

            // Second half: neighbor embedding
            if (neighbor_idx >= 0) {
                const float* neighbor_emb = embeddings + neighbor_idx * D;
                std::memcpy(edge_emb + D, neighbor_emb, static_cast<size_t>(D) * sizeof(float));
            } else {
                // Invalid index: fill with zeros
                std::memset(edge_emb + D, 0, static_cast<size_t>(D) * sizeof(float));
            }
        }
    }
}

/**
 * Scatter-add (accumulate into indexed positions).
 */
template <>
void scatter_add<ScalarBackend>(const float* input, const int* indices, float* output, int M, int k,
                                int D, int N) {
    // For each source point
    for (int i = 0; i < M; i++) {
        // For each value to scatter
        for (int j = 0; j < k; j++) {
            int target_idx = indices[i * k + j];

            // Skip invalid indices
            if (target_idx < 0 || target_idx >= N) {
                continue;
            }

            const float* src = input + (i * k + j) * D;
            float* dst = output + target_idx * D;

            // Accumulate: dst += src
            for (int d = 0; d < D; d++) {
                dst[d] += src[d];
            }
        }
    }
}

}  // namespace gather
}  // namespace pfalign
