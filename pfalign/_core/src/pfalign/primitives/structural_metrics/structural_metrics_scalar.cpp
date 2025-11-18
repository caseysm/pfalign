/**
 * Scalar implementation of structural alignment metrics.
 */

#include "structural_metrics_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include <cmath>
#include <vector>

namespace pfalign {
namespace structural_metrics {

/**
 * Compute TM-score (Template Modeling score).
 */
template <>
float compute_tm_score<ScalarBackend>(const float* P_aligned, const float* Q, int N, int L_target) {
    // Step 1: Compute normalization distance d₀
    // d₀ = 1.24 * (L_target - 15)^(1/3) - 1.8
    float L_minus_15 = static_cast<float>(L_target - 15);
    if (L_minus_15 < 1.0f)
        L_minus_15 = 1.0f;  // Avoid issues for very short proteins

    float d0 = 1.24f * std::cbrt(L_minus_15) - 1.8f;
    if (d0 < 0.5f)
        d0 = 0.5f;  // Floor at 0.5Å (safety for small proteins)

    float d0_sq = d0 * d0;

    // Step 2: Compute TM-score = (1/L_target) * Sigma 1/(1 + (dᵢ/d₀)^2)
    float tm_sum = 0.0f;

    for (int i = 0; i < N; i++) {
        float dx = P_aligned[i * 3 + 0] - Q[i * 3 + 0];
        float dy = P_aligned[i * 3 + 1] - Q[i * 3 + 1];
        float dz = P_aligned[i * 3 + 2] - Q[i * 3 + 2];

        float dist_sq = dx * dx + dy * dy + dz * dz;

        // TM-score term: 1/(1 + (d/d₀)^2) = 1/(1 + d^2/d₀^2) = d₀^2/(d₀^2 + d^2)
        tm_sum += d0_sq / (d0_sq + dist_sq);
    }

    float tm_score = tm_sum / static_cast<float>(L_target);
    return tm_score;
}

/**
 * Compute GDT-TS and GDT-HA scores.
 */
template <>
void compute_gdt<ScalarBackend>(const float* P_aligned, const float* Q, int N, float* gdt_ts,
                                float* gdt_ha, float* p1, float* p2, float* p4, float* p8,
                                float* p0_5) {
    // Count residues within each distance cutoff
    int count_0_5 = 0;
    int count_1 = 0;
    int count_2 = 0;
    int count_4 = 0;
    int count_8 = 0;

    for (int i = 0; i < N; i++) {
        float dx = P_aligned[i * 3 + 0] - Q[i * 3 + 0];
        float dy = P_aligned[i * 3 + 1] - Q[i * 3 + 1];
        float dz = P_aligned[i * 3 + 2] - Q[i * 3 + 2];

        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < 0.5f)
            count_0_5++;
        if (dist < 1.0f)
            count_1++;
        if (dist < 2.0f)
            count_2++;
        if (dist < 4.0f)
            count_4++;
        if (dist < 8.0f)
            count_8++;
    }

    float inv_N = 1.0f / static_cast<float>(N);

    // Compute percentages
    float percent_0_5 = count_0_5 * inv_N;
    float percent_1 = count_1 * inv_N;
    float percent_2 = count_2 * inv_N;
    float percent_4 = count_4 * inv_N;
    float percent_8 = count_8 * inv_N;

    // GDT-TS: average of [1, 2, 4, 8] Å cutoffs
    *gdt_ts = (percent_1 + percent_2 + percent_4 + percent_8) / 4.0f;

    // GDT-HA: average of [0.5, 1, 2, 4] Å cutoffs
    *gdt_ha = (percent_0_5 + percent_1 + percent_2 + percent_4) / 4.0f;

    // Optionally return individual percentages
    if (p0_5 != nullptr)
        *p0_5 = percent_0_5;
    if (p1 != nullptr)
        *p1 = percent_1;
    if (p2 != nullptr)
        *p2 = percent_2;
    if (p4 != nullptr)
        *p4 = percent_4;
    if (p8 != nullptr)
        *p8 = percent_8;
}

/**
 * Compute average RMSD for MSA (all-vs-all).
 */
template <>
float rmsd_msa_allvsall<ScalarBackend>(const float** ca_coords, const int** col2pos,
                                       int num_sequences, int num_columns) {
    if (num_sequences < 2)
        return 0.0f;

    float total_rmsd = 0.0f;
    int num_pairs = 0;

    // For each pair of sequences (i, j)
    for (int i = 0; i < num_sequences; i++) {
        for (int j = i + 1; j < num_sequences; j++) {
            // Extract aligned CA coordinates for this pair
            std::vector<float> ca_i_aligned, ca_j_aligned;
            ca_i_aligned.reserve(num_columns * 3);
            ca_j_aligned.reserve(num_columns * 3);

            for (int col = 0; col < num_columns; col++) {
                int posi = col2pos[i][col];
                int posj = col2pos[j][col];

                // Skip if either sequence has a gap at this column
                if (posi < 0 || posj < 0)
                    continue;

                // Add CA coordinates to aligned arrays
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 0]);
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 1]);
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 2]);

                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 0]);
                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 1]);
                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 2]);
            }

            int N_aligned = ca_i_aligned.size() / 3;
            if (N_aligned < 3)
                continue;  // Need at least 3 points for Kabsch

            // Run Kabsch alignment to get RMSD
            float R[9], t[3], rmsd;
            kabsch::kabsch_align<ScalarBackend>(ca_i_aligned.data(), ca_j_aligned.data(), N_aligned,
                                                R, t, &rmsd);

            total_rmsd += rmsd;
            num_pairs++;
        }
    }

    // Return average RMSD over all pairs
    if (num_pairs > 0) {
        return total_rmsd / num_pairs;
    }
    return 0.0f;
}

/**
 * Compute average TM-score for MSA (all-vs-all).
 */
template <>
float tm_score_msa_allvsall<ScalarBackend>(const float** ca_coords, const int** col2pos,
                                           int num_sequences, int num_columns,
                                           const int* seq_lengths) {
    if (num_sequences < 2)
        return 0.0f;

    float total_tm = 0.0f;
    int num_pairs = 0;

    // For each directed pair of sequences (i -> j)
    for (int i = 0; i < num_sequences; i++) {
        for (int j = 0; j < num_sequences; j++) {
            if (i == j)
                continue;  // Skip self-comparison

            // Extract aligned CA coordinates for this pair
            std::vector<float> ca_i_aligned, ca_j_aligned;
            ca_i_aligned.reserve(num_columns * 3);
            ca_j_aligned.reserve(num_columns * 3);

            for (int col = 0; col < num_columns; col++) {
                int posi = col2pos[i][col];
                int posj = col2pos[j][col];

                if (posi < 0 || posj < 0)
                    continue;

                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 0]);
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 1]);
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 2]);

                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 0]);
                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 1]);
                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 2]);
            }

            int N_aligned = ca_i_aligned.size() / 3;
            if (N_aligned < 3)
                continue;

            // Run Kabsch alignment
            float R[9], t[3], rmsd;
            kabsch::kabsch_align<ScalarBackend>(ca_i_aligned.data(), ca_j_aligned.data(), N_aligned,
                                                R, t, &rmsd);

            // Apply transformation to get aligned coordinates
            std::vector<float> ca_i_transformed(N_aligned * 3);
            for (int k = 0; k < N_aligned; k++) {
                float x = ca_i_aligned[k * 3 + 0];
                float y = ca_i_aligned[k * 3 + 1];
                float z = ca_i_aligned[k * 3 + 2];

                ca_i_transformed[k * 3 + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
                ca_i_transformed[k * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
                ca_i_transformed[k * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
            }

            // Compute TM-score (normalized by source length L_i)
            float tm = compute_tm_score<ScalarBackend>(ca_i_transformed.data(), ca_j_aligned.data(),
                                                       N_aligned, seq_lengths[i]);

            total_tm += tm;
            num_pairs++;
        }
    }

    // Return average TM-score over all directed pairs
    if (num_pairs > 0) {
        return total_tm / num_pairs;
    }
    return 0.0f;
}

/**
 * Compute MSA metrics bundle (RMSD + TM-score) efficiently.
 */
template <>
void msa_metrics_allvsall<ScalarBackend>(const float** ca_coords, const int** col2pos,
                                         int num_sequences, int num_columns, const int* seq_lengths,
                                         float* avg_rmsd, float* avg_tm, float* avg_tm_symmetric) {
    if (num_sequences < 2) {
        *avg_rmsd = 0.0f;
        *avg_tm = 0.0f;
        if (avg_tm_symmetric != nullptr)
            *avg_tm_symmetric = 0.0f;
        return;
    }

    float total_rmsd = 0.0f;
    float total_tm_forward = 0.0f;
    float total_tm_backward = 0.0f;
    int num_undirected_pairs = 0;
    int num_directed_pairs = 0;

    // For each undirected pair of sequences (i, j) where i < j
    for (int i = 0; i < num_sequences; i++) {
        for (int j = i + 1; j < num_sequences; j++) {
            // Extract aligned CA coordinates
            std::vector<float> ca_i_aligned, ca_j_aligned;
            ca_i_aligned.reserve(num_columns * 3);
            ca_j_aligned.reserve(num_columns * 3);

            for (int col = 0; col < num_columns; col++) {
                int posi = col2pos[i][col];
                int posj = col2pos[j][col];

                if (posi < 0 || posj < 0)
                    continue;

                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 0]);
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 1]);
                ca_i_aligned.push_back(ca_coords[i][posi * 3 + 2]);

                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 0]);
                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 1]);
                ca_j_aligned.push_back(ca_coords[j][posj * 3 + 2]);
            }

            int N_aligned = ca_i_aligned.size() / 3;
            if (N_aligned < 3)
                continue;

            // === Compute i->j (forward direction) ===
            float R_ij[9], t_ij[3], rmsd_ij;
            kabsch::kabsch_align<ScalarBackend>(ca_i_aligned.data(), ca_j_aligned.data(), N_aligned,
                                                R_ij, t_ij, &rmsd_ij);

            // Transform i to align with j
            std::vector<float> ca_i_transformed(N_aligned * 3);
            for (int k = 0; k < N_aligned; k++) {
                float x = ca_i_aligned[k * 3 + 0];
                float y = ca_i_aligned[k * 3 + 1];
                float z = ca_i_aligned[k * 3 + 2];

                ca_i_transformed[k * 3 + 0] = R_ij[0] * x + R_ij[1] * y + R_ij[2] * z + t_ij[0];
                ca_i_transformed[k * 3 + 1] = R_ij[3] * x + R_ij[4] * y + R_ij[5] * z + t_ij[1];
                ca_i_transformed[k * 3 + 2] = R_ij[6] * x + R_ij[7] * y + R_ij[8] * z + t_ij[2];
            }

            // TM(i->j) normalized by L_i
            float tm_ij = compute_tm_score<ScalarBackend>(
                ca_i_transformed.data(), ca_j_aligned.data(), N_aligned, seq_lengths[i]);

            // Accumulate
            total_rmsd += rmsd_ij;
            total_tm_forward += tm_ij;
            num_undirected_pairs++;
            num_directed_pairs++;

            // === Optionally compute j->i (backward direction) for symmetric TM ===
            if (avg_tm_symmetric != nullptr) {
                float R_ji[9], t_ji[3], rmsd_ji;
                kabsch::kabsch_align<ScalarBackend>(ca_j_aligned.data(), ca_i_aligned.data(),
                                                    N_aligned, R_ji, t_ji, &rmsd_ji);

                std::vector<float> ca_j_transformed(N_aligned * 3);
                for (int k = 0; k < N_aligned; k++) {
                    float x = ca_j_aligned[k * 3 + 0];
                    float y = ca_j_aligned[k * 3 + 1];
                    float z = ca_j_aligned[k * 3 + 2];

                    ca_j_transformed[k * 3 + 0] = R_ji[0] * x + R_ji[1] * y + R_ji[2] * z + t_ji[0];
                    ca_j_transformed[k * 3 + 1] = R_ji[3] * x + R_ji[4] * y + R_ji[5] * z + t_ji[1];
                    ca_j_transformed[k * 3 + 2] = R_ji[6] * x + R_ji[7] * y + R_ji[8] * z + t_ji[2];
                }

                // TM(j->i) normalized by L_j
                float tm_ji = compute_tm_score<ScalarBackend>(
                    ca_j_transformed.data(), ca_i_aligned.data(), N_aligned, seq_lengths[j]);

                total_tm_backward += tm_ji;
                num_directed_pairs++;
            }
        }
    }

    // Compute averages
    if (num_undirected_pairs > 0) {
        *avg_rmsd = total_rmsd / num_undirected_pairs;
        *avg_tm = total_tm_forward / num_undirected_pairs;

        if (avg_tm_symmetric != nullptr && num_directed_pairs > 0) {
            // Average over both directions
            *avg_tm_symmetric = (total_tm_forward + total_tm_backward) / num_directed_pairs;
        }
    } else {
        *avg_rmsd = 0.0f;
        *avg_tm = 0.0f;
        if (avg_tm_symmetric != nullptr)
            *avg_tm_symmetric = 0.0f;
    }
}

}  // namespace structural_metrics
}  // namespace pfalign
