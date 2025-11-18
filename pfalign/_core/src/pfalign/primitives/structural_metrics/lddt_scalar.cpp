/**
 * Scalar implementation of LDDT scoring.
 */

#include "lddt_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace pfalign {
namespace structural_metrics {

/**
 * Helper: Check if distance pair should be included based on R0 and symmetry mode.
 */
static inline bool should_include_distance_pair(float d1, float d2, float R0,
                                                const char* symmetry) {
    if (std::strcmp(symmetry, "first") == 0) {
        // Standard LDDT: only reference (d1) within R0
        return d1 <= R0;
    } else if (std::strcmp(symmetry, "both") == 0) {
        // Both distances must be within R0
        return (d1 <= R0) && (d2 <= R0);
    } else if (std::strcmp(symmetry, "either") == 0) {
        // Either distance within R0
        return (d1 <= R0) || (d2 <= R0);
    }
    // Default to "first"
    return d1 <= R0;
}

/**
 * Helper: Compute LDDT score for distance difference.
 */
static inline float compute_lddt_term(float dist_diff,
                                      const float* thresholds  // [0.5, 1, 2, 4]
) {
    // Count how many thresholds are satisfied
    int count = 0;
    for (int t = 0; t < 4; t++) {
        if (dist_diff < thresholds[t]) {
            count++;
        }
    }
    // Return average over 4 thresholds
    return count / 4.0f;
}

/**
 * LDDT pairwise implementation (Scalar backend).
 */
template <>
float lddt_pairwise<ScalarBackend>(const float* dist_mx1, const float* dist_mx2,
                                   const int* alignment, int aligned_length,
                                   const LDDTParams& params, float* per_residue_scores) {
    if (aligned_length == 0)
        return 0.0f;

    // Compute L1 and L2 from alignment (max position + 1)
    int L1 = 0, L2 = 0;
    for (int k = 0; k < aligned_length; k++) {
        if (alignment[k * 2 + 0] >= L1)
            L1 = alignment[k * 2 + 0] + 1;
        if (alignment[k * 2 + 1] >= L2)
            L2 = alignment[k * 2 + 1] + 1;
    }

    float total_score = 0.0f;
    int total_residues_scored = 0;

    // For each aligned residue pair (coli)
    for (int coli = 0; coli < aligned_length; coli++) {
        int pos1i = alignment[coli * 2 + 0];
        int pos2i = alignment[coli * 2 + 1];

        // Skip if gap (position -1)
        if (pos1i < 0 || pos2i < 0) {
            if (per_residue_scores != nullptr) {
                per_residue_scores[coli] = 0.0f;
            }
            continue;
        }

        int num_considered = 0;
        float residue_score = 0.0f;

        // For each other aligned pair (colj)
        for (int colj = 0; colj < aligned_length; colj++) {
            if (colj == coli)
                continue;

            int pos1j = alignment[colj * 2 + 0];
            int pos2j = alignment[colj * 2 + 1];

            // Skip if gap
            if (pos1j < 0 || pos2j < 0)
                continue;

            // Get distances from precomputed distance matrices
            // dist_mx1 is [L1 * L1], dist_mx2 is [L2 * L2] (row-major)
            float d1 = dist_mx1[pos1i * L1 + pos1j];
            float d2 = dist_mx2[pos2i * L2 + pos2j];

            // Check if this pair should be included
            if (!should_include_distance_pair(d1, d2, params.R0, params.symmetry)) {
                continue;
            }

            num_considered++;

            // Compute distance difference
            float dist_diff = std::abs(d1 - d2);

            // Compute LDDT term (average over 4 thresholds)
            residue_score += compute_lddt_term(dist_diff, params.thresholds);
        }

        // Compute average for this residue
        if (num_considered > 0) {
            float avg_score = residue_score / num_considered;
            if (per_residue_scores != nullptr) {
                per_residue_scores[coli] = avg_score;
            }
            total_score += avg_score;
            total_residues_scored++;
        } else {
            if (per_residue_scores != nullptr) {
                per_residue_scores[coli] = 0.0f;
            }
        }
    }

    // Return average over all scored residues
    if (total_residues_scored > 0) {
        return total_score / total_residues_scored;
    }
    return 0.0f;
}

/**
 * LDDT MSA all-vs-all implementation (Scalar backend).
 */
template <>
float lddt_msa_allvsall<ScalarBackend>(const float** dist_mxs, const int** col2pos,
                                       int num_sequences, int num_columns,
                                       const LDDTParams& params) {
    if (num_sequences < 2)
        return 0.0f;

    float total_lddt = 0.0f;
    int num_pairs = 0;

    // For each pair of sequences (i, j)
    for (int i = 0; i < num_sequences; i++) {
        for (int j = i + 1; j < num_sequences; j++) {
            // Build alignment between sequences i and j from col2pos
            std::vector<int> alignment;
            alignment.reserve(
                static_cast<size_t>(num_columns * 2));  // Pre-allocate to avoid reallocations
            for (int col = 0; col < num_columns; col++) {
                int posi = col2pos[i][col];
                int posj = col2pos[j][col];
                if (posi >= 0 && posj >= 0) {
                    alignment.push_back(posi);
                    alignment.push_back(posj);
                }
            }

            int aligned_length = static_cast<int>(alignment.size() / 2);
            if (aligned_length == 0)
                continue;

            // Compute pairwise LDDT
            float lddt = lddt_pairwise<ScalarBackend>(dist_mxs[i], dist_mxs[j], alignment.data(),
                                                      aligned_length, params, nullptr);

            total_lddt += lddt;
            num_pairs++;
        }
    }

    // Return average over all pairs
    if (num_pairs > 0) {
        return total_lddt / num_pairs;
    }
    return 0.0f;
}

/**
 * LDDT MSA column-centric implementation (Scalar backend).
 */
template <>
float lddt_msa_column<ScalarBackend>(const float** dist_mxs, const int** col2pos, int num_sequences,
                                     int num_columns, const LDDTParams& params, float* col_scores) {
    if (num_sequences < 2)
        return 0.0f;

    // Precompute sequence lengths (max position + 1 for each sequence)
    std::vector<int> seq_lengths(static_cast<size_t>(num_sequences), 0);
    for (int seq = 0; seq < num_sequences; seq++) {
        for (int col = 0; col < num_columns; col++) {
            if (col2pos[seq][col] >= seq_lengths[seq]) {
                seq_lengths[seq] = col2pos[seq][col] + 1;
            }
        }
    }

    float total_col_scores = 0.0f;
    int num_cols_scored = 0;

    // For each column
    for (int col = 0; col < num_columns; col++) {
        float col_score = 0.0f;
        int num_pairs_in_col = 0;

        // For each pair of sequences at this column
        for (int seqi = 0; seqi < num_sequences; seqi++) {
            int posi = col2pos[seqi][col];
            if (posi < 0)
                continue;  // Gap

            int Li = seq_lengths[seqi];

            for (int seqj = seqi + 1; seqj < num_sequences; seqj++) {
                int posj = col2pos[seqj][col];
                if (posj < 0)
                    continue;  // Gap

                int Lj = seq_lengths[seqj];

                // Compute score for this pair at this column
                int num_considered = 0;
                float pair_score = 0.0f;

                // For each other column
                for (int col2 = 0; col2 < num_columns; col2++) {
                    if (col2 == col)
                        continue;

                    int posi2 = col2pos[seqi][col2];
                    int posj2 = col2pos[seqj][col2];
                    if (posi2 < 0 || posj2 < 0)
                        continue;

                    // Get distances from precomputed distance matrices
                    float di = dist_mxs[seqi][posi * Li + posi2];
                    float dj = dist_mxs[seqj][posj * Lj + posj2];

                    // Check inclusion
                    if (!should_include_distance_pair(di, dj, params.R0, params.symmetry)) {
                        continue;
                    }

                    num_considered++;

                    float dist_diff = std::abs(di - dj);
                    pair_score += compute_lddt_term(dist_diff, params.thresholds);
                }

                // Average for this pair
                if (num_considered > 0) {
                    col_score += pair_score / num_considered;
                    num_pairs_in_col++;
                }
            }
        }

        // Average for this column over all pairs
        if (num_pairs_in_col > 0) {
            float avg_col_score = col_score / num_pairs_in_col;
            if (col_scores != nullptr) {
                col_scores[col] = avg_col_score;
            }
            total_col_scores += avg_col_score;
            num_cols_scored++;
        } else {
            if (col_scores != nullptr) {
                col_scores[col] = 0.0f;
            }
        }
    }

    // Return average over all columns
    if (num_cols_scored > 0) {
        return total_col_scores / num_cols_scored;
    }
    return 0.0f;
}

// ============================================================================
}  // namespace structural_metrics
}  // namespace pfalign
