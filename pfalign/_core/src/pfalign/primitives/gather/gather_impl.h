#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace gather {

/**
 * Gather operations for MPNN message passing.
 *
 * Collects features from specified indices - used to gather neighbor
 * embeddings based on KNN results.
 *
 * Example workflow:
 *   1. KNN finds k nearest neighbors for each point -> indices [N * k]
 *   2. Gather collects neighbor embeddings -> [N * k * D]
 *   3. MPNN processes neighbor features via message passing
 *
 * Reference:
 *   align/_jax_reference/mpnn.py::gather_edges()
 */

/**
 * Gather embeddings from specified indices.
 *
 * For each query point i and each of its k neighbors j:
 *   output[i, j, :] = embeddings[indices[i, j], :]
 *
 * @param embeddings    Source embeddings [N * D] (row-major)
 * @param indices       Indices to gather [M * k] (row-major)
 * @param output        Gathered embeddings [M * k * D] (row-major)
 * @param M             Number of query points
 * @param k             Number of neighbors per query
 * @param D             Embedding dimension
 *
 * Example:
 *   float embeddings[100 * 128];  // 100 residues, 128-dim
 *   int indices[100 * 30];        // 30 neighbors each (from KNN)
 *   float output[100 * 30 * 128]; // Neighbor embeddings
 *   gather<ScalarBackend>(embeddings, indices, output, 100, 30, 128);
 */
template <typename Backend>
void gather(const float* embeddings, const int* indices, float* output, int M, int k, int D);

/**
 * Gather edges (pairwise features).
 *
 * For each query point i and each neighbor j, concatenates:
 *   output[i, j, :] = [query_emb[i], neighbor_emb[indices[i,j]]]
 *
 * This creates pairwise (edge) features for message passing.
 *
 * @param embeddings    Node embeddings [N * D]
 * @param indices       Neighbor indices [M * k]
 * @param output        Edge embeddings [M * k * 2D]
 * @param M             Number of query points
 * @param k             Number of neighbors
 * @param D             Embedding dimension
 *
 * Example:
 *   float embeddings[100 * 128];
 *   int indices[100 * 30];
 *   float output[100 * 30 * 256];  // 2D = 256
 *   gather_edges<ScalarBackend>(embeddings, indices, output, 100, 30, 128);
 */
template <typename Backend>
void gather_edges(const float* embeddings, const int* indices, float* output, int M, int k, int D);

/**
 * Scatter-add operation (reverse of gather).
 *
 * Accumulates values from multiple sources into indexed positions:
 *   output[indices[i, j]] += input[i, j, :]
 *
 * Used in message passing to aggregate messages from neighbors.
 *
 * @param input         Input features [M * k * D]
 * @param indices       Target indices [M * k]
 * @param output        Output accumulator [N * D] (must be zero-initialized!)
 * @param M             Number of source points
 * @param k             Number of values per source
 * @param D             Feature dimension
 * @param N             Size of output (max index + 1)
 *
 * Note: output must be initialized to zero before calling.
 */
template <typename Backend>
void scatter_add(const float* input, const int* indices, float* output, int M, int k, int D, int N);

}  // namespace gather
}  // namespace pfalign
