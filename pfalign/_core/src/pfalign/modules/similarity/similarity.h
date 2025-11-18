/**
 * Similarity computation for protein alignment.
 *
 * Computes pairwise similarity scores between residue embeddings
 * using dot product: S[i,j] = dot(emb1[i], emb2[j])
 *
 * This is implemented as a matrix multiply: S = emb1 * emb2^T
 */

#pragma once

namespace pfalign {
namespace similarity {

/**
 * Compute pairwise similarity matrix between two protein embeddings.
 *
 * @param emb1          First protein embeddings [L1 * D]
 * @param emb2          Second protein embeddings [L2 * D]
 * @param similarity    Output similarity matrix [L1 * L2]
 * @param L1            Length of first protein
 * @param L2            Length of second protein
 * @param D             Embedding dimension
 *
 * Computes: similarity[i,j] = dot(emb1[i,:], emb2[j,:])
 *
 * Example:
 *   float emb1[100 * 128];  // Protein 1: 100 residues, 128D embeddings
 *   float emb2[150 * 128];  // Protein 2: 150 residues, 128D embeddings
 *   float sim[100 * 150];   // Output similarity matrix
 *   compute_similarity<ScalarBackend>(emb1, emb2, sim, 100, 150, 128);
 */
template <typename Backend>
void compute_similarity(const float* emb1, const float* emb2, float* similarity, int L1, int L2,
                        int D);

}  // namespace similarity
}  // namespace pfalign
