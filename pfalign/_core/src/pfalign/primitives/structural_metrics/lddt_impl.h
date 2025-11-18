/**
 * LDDT (Local Distance Difference Test) Scoring
 *
 * Computes local structural similarity without superposition.
 * Evaluates preservation of local distance patterns.
 *
 * Reference:
 *   Mariani et al. (2013). "LDDT: a local superposition-free score
 *   for comparing protein structures and models using distance difference tests."
 *   Bioinformatics 29.21: 2722-2728.
 *
 * Three variants implemented:
 * 1. lddt_pairwise: Standard LDDT for model vs reference (lddt_original.py)
 * 2. lddt_msa_allvsall: MSA-based all-vs-all average (lddt_mu.py - CORRECT)
 * 3. lddt_msa_column: MSA-based column-centric (lddt_foldmason.py - for comparison)
 */

#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace structural_metrics {

/**
 * LDDT default parameters.
 */
struct LDDTParams {
    float R0 = 15.0f;                                // Inclusion radius (Å)
    float thresholds[4] = {0.5f, 1.0f, 2.0f, 4.0f};  // Distance thresholds (Å)
    const char* symmetry = "first";                  // "first", "both", "either"
};

/**
 * Compute LDDT score for pairwise comparison (original LDDT).
 *
 * This is the standard LDDT metric for comparing a model structure against
 * a reference structure. It measures the fraction of local distances that
 * are preserved within specified thresholds.
 *
 * Algorithm:
 * 1. For each aligned residue pair (i, j):
 *    - Consider all other aligned pairs (i', j') within R0 of i
 *    - Check if |dist(i,i') - dist(j,j')| < each threshold
 * 2. Score = (preserved distances) / (considered distances)
 * 3. Average over 4 thresholds: [0.5, 1, 2, 4] Å
 *
 * Symmetry modes:
 * - "first": Only reference distances within R0 are considered (standard LDDT)
 * - "both": Both reference AND model distances must be within R0
 * - "either": Either reference OR model distance within R0
 *
 * @tparam Backend Computation backend
 * @param dist_mx1 Distance matrix for reference structure [L1 * L1]
 * @param dist_mx2 Distance matrix for model structure [L2 * L2]
 * @param alignment Aligned position pairs [aligned_length * 2]
 *                  alignment[i][0] = position in structure 1
 *                  alignment[i][1] = position in structure 2
 * @param aligned_length Number of aligned residue pairs
 * @param params LDDT parameters (R0, thresholds, symmetry)
 * @param per_residue_scores Optional output: LDDT score per residue [aligned_length]
 * @return LDDT score in [0, 1], where 1 = perfect local similarity
 *
 * Example:
 * ```cpp
 *   // Compute distance matrices
 *   std::vector<float> dist1(L1 * L1), dist2(L2 * L2);
 *   compute_distance_matrix<ScalarBackend>(ca1, L1, dist1.data());
 *   compute_distance_matrix<ScalarBackend>(ca2, L2, dist2.data());
 *
 *   // Define alignment (example: perfect alignment of first N residues)
 *   std::vector<int> alignment(N * 2);
 *   for (int i = 0; i < N; i++) {
 *       alignment[i*2 + 0] = i;  // position in structure 1
 *       alignment[i*2 + 1] = i;  // position in structure 2
 *   }
 *
 *   // Compute LDDT
 *   LDDTParams params;
 *   float lddt = lddt_pairwise<ScalarBackend>(
 *       dist1.data(), dist2.data(), alignment.data(), N, params
 *   );
 * ```
 *
 * Properties:
 * - Range: [0, 1]
 * - Superposition-free (no Kabsch alignment needed)
 * - Sensitive to local geometry
 * - Robust to domain movements
 * - >0.8: High quality model
 * - 0.6-0.8: Good model
 * - <0.6: Poor model
 *
 * Performance:
 * - Scalar: O(N^2) ~5ms for N=200
 */
template <typename Backend>
float lddt_pairwise(const float* dist_mx1,  // [L1 * L1] reference distance matrix
                    const float* dist_mx2,  // [L2 * L2] model distance matrix
                    const int* alignment,   // [aligned_length * 2] position pairs
                    int aligned_length,     // number of aligned pairs
                    const LDDTParams& params = LDDTParams(),
                    float* per_residue_scores = nullptr  // optional per-residue output
);

/**
 * Compute LDDT for MSA using all-vs-all averaging (lddt_mu - CORRECT method).
 *
 * For each pair of sequences in the MSA, compute pairwise LDDT, then average.
 * This is the mathematically correct way to compute MSA-based LDDT.
 *
 * Formula:
 *   LDDT = average over all pairs (i,j): lddt_pairwise(struct_i, struct_j)
 *
 * @tparam Backend Computation backend
 * @param dist_mxs Array of distance matrices, one per sequence [num_sequences][L*L]
 * @param col2pos Column-to-position mapping per sequence [num_sequences][num_columns]
 *                col2pos[seq_idx][col] = position in ungapped sequence (-1 for gap)
 * @param num_sequences Number of sequences in MSA
 * @param num_columns Number of columns in MSA alignment
 * @param params LDDT parameters
 * @return Average LDDT score over all sequence pairs
 *
 * Example:
 * ```cpp
 *   // dist_mxs[0] = distance matrix for sequence 0
 *   // col2pos[0][5] = position in ungapped sequence 0 corresponding to column 5
 *   //                 (-1 if gap at that column)
 *
 *   float lddt = lddt_msa_allvsall<ScalarBackend>(
 *       dist_mxs, col2pos, num_seqs, num_cols, params
 *   );
 * ```
 */
template <typename Backend>
float lddt_msa_allvsall(const float** dist_mxs,  // [num_sequences][L*L] distance matrices
                        const int** col2pos,     // [num_sequences][num_columns] col->pos mapping
                        int num_sequences, int num_columns,
                        const LDDTParams& params = LDDTParams());

/**
 * Compute LDDT for MSA using column-centric averaging (lddt_foldmason).
 *
 * For each column, compute average LDDT over all pairs, then average columns.
 * This is the FoldMason method - known to have anomalies with gap handling.
 *
 * Formula:
 *   LDDT = average over columns: average over pairs at column
 *
 * NOTE: This method is included for comparison/validation purposes only.
 * It is known to give incorrect scores for misaligned sequences.
 * Use lddt_msa_allvsall for correct MSA LDDT scoring.
 *
 * @tparam Backend Computation backend
 * @param dist_mxs Array of distance matrices
 * @param col2pos Column-to-position mapping
 * @param num_sequences Number of sequences
 * @param num_columns Number of columns
 * @param params LDDT parameters
 * @param col_scores Optional output: LDDT score per column [num_columns]
 * @return Average LDDT score (column-centric method)
 */
template <typename Backend>
float lddt_msa_column(const float** dist_mxs, const int** col2pos, int num_sequences,
                      int num_columns, const LDDTParams& params = LDDTParams(),
                      float* col_scores = nullptr);

}  // namespace structural_metrics
}  // namespace pfalign
