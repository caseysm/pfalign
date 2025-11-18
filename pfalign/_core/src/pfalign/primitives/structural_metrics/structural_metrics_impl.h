/**
 * Structural Alignment Metrics
 *
 * Computes standard protein structure comparison metrics:
 * - TM-score: Template Modeling score (fold similarity)
 * - GDT-TS: Global Distance Test - Total Score
 * - GDT-HA: Global Distance Test - High Accuracy
 *
 * These metrics are computed AFTER Kabsch alignment and evaluate
 * the quality of structural similarity.
 *
 * References:
 * - Zhang & Skolnick (2004). "Scoring function for automated assessment..."
 * - Zemla (2003). "LGA: a method for finding 3D similarities in protein structures"
 */

#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace structural_metrics {

/**
 * Compute TM-score (Template Modeling score).
 *
 * TM-score measures the global fold similarity between two structures.
 * It is more sensitive than RMSD for detecting correct folds.
 *
 * Formula:
 *   TM-score = (1/L_target) * Sigmaᵢ 1/(1 + (dᵢ/d₀)^2)
 *
 * where:
 *   d₀ = 1.24 * (L_target - 15)^(1/3) - 1.8  (normalization distance)
 *   dᵢ = ||P_aligned[i] - Q[i]||           (distance after alignment)
 *   L_target = normalization length (typically L1 or L2)
 *
 * Properties:
 * - Range: [0, 1], where 1 = perfect match
 * - >0.5: Same fold (high confidence)
 * - 0.3-0.5: Possible homology
 * - <0.3: Different fold
 * - Length-normalized: fair comparison across sizes
 * - **Asymmetric**: TM(A->B) != TM(B->A) due to normalization
 *
 * Note: Typically report both TM(A->B, norm by L_A) and TM(A->B, norm by L_B)
 *
 * @tparam Backend Computation backend
 * @param P_aligned Source coords after Kabsch alignment [N * 3]
 * @param Q Target coords [N * 3]
 * @param N Number of aligned point pairs
 * @param L_target Normalization length (for TM-score formula)
 *                 Typically set to L1 (source length) or L2 (target length)
 * @return TM-score in [0, 1]
 *
 * Example:
 * ```cpp
 *   // After Kabsch alignment
 *   float tm1 = compute_tm_score<ScalarBackend>(ca1_aligned, ca2, N, L1);  // norm by L1
 *   float tm2 = compute_tm_score<ScalarBackend>(ca1_aligned, ca2, N, L2);  // norm by L2
 *
 *   // Convention: report higher score, or both
 *   std::cout << "TM-score (norm by L1=" << L1 << "): " << tm1 << "\n";
 *   std::cout << "TM-score (norm by L2=" << L2 << "): " << tm2 << "\n";
 * ```
 *
 * Performance:
 * - Scalar: O(N) ~2 mus for N=200
 * - NEON: Vectorize distance computation ~1 mus for N=200
 */
template <typename Backend>
float compute_tm_score(const float* P_aligned,  // [N * 3] aligned source coords
                       const float* Q,          // [N * 3] target coords
                       int N,                   // number of aligned pairs
                       int L_target             // normalization length
);

/**
 * Compute GDT scores (Global Distance Test).
 *
 * GDT measures the percentage of residues aligned within specific distance cutoffs.
 * Two variants:
 * - GDT-TS (Total Score): Cutoffs [1, 2, 4, 8] Å (standard)
 * - GDT-HA (High Accuracy): Cutoffs [0.5, 1, 2, 4] Å (stricter)
 *
 * Formula:
 *   GDT-TS = (P₁ + P₂ + P₄ + P₈) / 4
 *   GDT-HA = (P₀.₅ + P₁ + P₂ + P₄) / 4
 *
 * where Pₓ = percentage of residues with distance < x Å
 *
 * Properties:
 * - Range: [0, 1], where 1 = all residues within cutoffs
 * - >0.7: High quality alignment
 * - 0.5-0.7: Good alignment
 * - <0.5: Poor alignment
 * - Symmetric (same result regardless of which structure is source/target)
 *
 * @tparam Backend Computation backend
 * @param P_aligned Source coords after Kabsch [N * 3]
 * @param Q Target coords [N * 3]
 * @param N Number of aligned pairs
 * @param gdt_ts Output GDT-TS score [0, 1]
 * @param gdt_ha Output GDT-HA score [0, 1]
 * @param p1 Optional output: percentage under 1Å (can be nullptr)
 * @param p2 Optional output: percentage under 2Å (can be nullptr)
 * @param p4 Optional output: percentage under 4Å (can be nullptr)
 * @param p8 Optional output: percentage under 8Å (can be nullptr)
 * @param p0_5 Optional output: percentage under 0.5Å (can be nullptr)
 *
 * Example:
 * ```cpp
 *   float gdt_ts, gdt_ha;
 *   compute_gdt<ScalarBackend>(ca1_aligned, ca2, N, &gdt_ts, &gdt_ha);
 *
 *   std::cout << "GDT-TS: " << gdt_ts << "\n";
 *   std::cout << "GDT-HA: " << gdt_ha << "\n";
 * ```
 *
 * Performance:
 * - Scalar: O(N) ~3 mus for N=200
 * - NEON: Vectorize distance + comparison ~1.5 mus for N=200
 */
template <typename Backend>
void compute_gdt(const float* P_aligned,  // [N * 3] aligned source coords
                 const float* Q,          // [N * 3] target coords
                 int N,                   // number of aligned pairs
                 float* gdt_ts,           // output GDT-TS
                 float* gdt_ha,           // output GDT-HA
                 float* p1 = nullptr,     // optional: % under 1Å
                 float* p2 = nullptr,     // optional: % under 2Å
                 float* p4 = nullptr,     // optional: % under 4Å
                 float* p8 = nullptr,     // optional: % under 8Å
                 float* p0_5 = nullptr    // optional: % under 0.5Å
);

/**
 * Compute average RMSD for MSA using all-vs-all pairwise comparisons.
 *
 * For each pair of sequences in the MSA:
 * 1. Extract aligned CA coordinates using col2pos mapping
 * 2. Run Kabsch alignment
 * 3. Compute RMSD
 * 4. Average over all pairs
 *
 * Formula:
 *   RMSD_MSA = average over all pairs (i,j): RMSD(structure_i, structure_j)
 *
 * Properties:
 * - Range: [0, inf) Ångströms
 * - 0Å: Identical structures
 * - <2Å: Very similar structures
 * - 2-4Å: Moderately similar
 * - >4Å: Different structures
 * - Symmetric: RMSD(i,j) = RMSD(j,i)
 *
 * @tparam Backend Computation backend
 * @param ca_coords Array of CA coordinates per sequence [num_sequences][L*3]
 * @param col2pos Column-to-position mapping [num_sequences][num_columns]
 *                col2pos[seq_idx][col] = position in ungapped sequence (-1 for gap)
 * @param num_sequences Number of sequences in MSA (must be >= 2)
 * @param num_columns Number of columns in MSA alignment
 * @return Average RMSD over all sequence pairs
 *
 * Example:
 * ```cpp
 *   // ca_coords[0] = CA coordinates for sequence 0 [L0 * 3]
 *   // col2pos[0][5] = position in sequence 0 at column 5 (-1 if gap)
 *
 *   float avg_rmsd = rmsd_msa_allvsall<ScalarBackend>(
 *       ca_coords, col2pos, num_seqs, num_cols
 *   );
 * ```
 *
 * Performance:
 * - Complexity: O(M^2 * N) where M=num_sequences, N=avg aligned length
 * - Scalar: ~0.5ms per pair for N=200
 * - For M=10 sequences: ~50 pairs, ~25ms total
 */
template <typename Backend>
float rmsd_msa_allvsall(const float** ca_coords,  // [num_sequences][L*3] CA coordinates
                        const int** col2pos,      // [num_sequences][num_columns] col->pos mapping
                        int num_sequences, int num_columns);

/**
 * Compute average TM-score for MSA using all-vs-all pairwise comparisons.
 *
 * For each pair of sequences in the MSA:
 * 1. Extract aligned CA coordinates using col2pos mapping
 * 2. Run Kabsch alignment
 * 3. Compute TM-score (normalized by source length L_i)
 * 4. Average over all pairs
 *
 * Formula:
 *   TM_MSA = average over all pairs (i,j): TM(i->j, normalized by L_i)
 *
 * Note: TM-score is asymmetric. This function computes TM(i->j) normalized
 * by the source length L_i for each directed pair, then averages.
 *
 * Alternative: Can also compute symmetric version using max(L_i, L_j) or
 * average of TM(i->j, L_i) and TM(j->i, L_j). This implementation uses
 * directed pairs for consistency with standard TM-score practice.
 *
 * Properties:
 * - Range: [0, 1]
 * - >0.5: Similar fold (high confidence)
 * - 0.3-0.5: Possible homology
 * - <0.3: Different fold
 *
 * @tparam Backend Computation backend
 * @param ca_coords Array of CA coordinates per sequence [num_sequences][L*3]
 * @param col2pos Column-to-position mapping [num_sequences][num_columns]
 * @param num_sequences Number of sequences in MSA (must be >= 2)
 * @param num_columns Number of columns in MSA alignment
 * @param seq_lengths Array of sequence lengths [num_sequences]
 *                    Used for TM-score normalization
 * @return Average TM-score over all sequence pairs
 *
 * Example:
 * ```cpp
 *   float avg_tm = tm_score_msa_allvsall<ScalarBackend>(
 *       ca_coords, col2pos, num_seqs, num_cols, seq_lengths
 *   );
 * ```
 *
 * Performance:
 * - Complexity: O(M^2 * N) where M=num_sequences, N=avg aligned length
 * - Scalar: ~0.5ms per pair for N=200
 * - For M=10 sequences: ~50 pairs (directed), ~25ms total
 */
template <typename Backend>
float tm_score_msa_allvsall(
    const float** ca_coords,  // [num_sequences][L*3] CA coordinates
    const int** col2pos,      // [num_sequences][num_columns] col->pos mapping
    int num_sequences, int num_columns,
    const int* seq_lengths  // [num_sequences] sequence lengths for normalization
);

/**
 * Compute MSA metrics bundle (RMSD, TM-score, both directions).
 *
 * Efficiently computes both RMSD and TM-score in a single pass,
 * amortizing the cost of Kabsch alignment across both metrics.
 *
 * Computes:
 * - Average RMSD over all pairs
 * - Average TM-score (i->j, normalized by L_i)
 * - Optional: Average TM-score (j->i, normalized by L_j)
 *
 * @tparam Backend Computation backend
 * @param ca_coords Array of CA coordinates per sequence [num_sequences][L*3]
 * @param col2pos Column-to-position mapping [num_sequences][num_columns]
 * @param num_sequences Number of sequences in MSA
 * @param num_columns Number of columns in MSA alignment
 * @param seq_lengths Array of sequence lengths [num_sequences]
 * @param avg_rmsd Output: average RMSD
 * @param avg_tm Output: average TM-score
 * @param avg_tm_symmetric Optional output: symmetric TM (both directions) (can be nullptr)
 *
 * Example:
 * ```cpp
 *   float avg_rmsd, avg_tm, avg_tm_sym;
 *   msa_metrics_allvsall<ScalarBackend>(
 *       ca_coords, col2pos, num_seqs, num_cols, seq_lengths,
 *       &avg_rmsd, &avg_tm, &avg_tm_sym
 *   );
 * ```
 */
template <typename Backend>
void msa_metrics_allvsall(const float** ca_coords, const int** col2pos, int num_sequences,
                          int num_columns, const int* seq_lengths, float* avg_rmsd, float* avg_tm,
                          float* avg_tm_symmetric = nullptr);

}  // namespace structural_metrics
}  // namespace pfalign
