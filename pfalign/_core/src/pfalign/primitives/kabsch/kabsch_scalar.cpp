/**
 * Scalar implementation of Kabsch algorithm.
 */

#include "kabsch_impl.h"
#include "pfalign/primitives/svd3x3/svd3x3_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace pfalign {
namespace kabsch {

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute 3x3 matrix determinant.
 */
static inline float det3x3(const float* M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) +
           M[2] * (M[3] * M[7] - M[4] * M[6]);
}

/**
 * Compute 3x3 matrix multiplication: C = A * B
 */
static inline void matmul3x3(const float* A, const float* B, float* C) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += A[i * 3 + k] * B[k * 3 + j];
            }
            C[i * 3 + j] = sum;
        }
    }
}

/**
 * Compute 3x3 matrix transpose.
 */
static inline void transpose3x3(const float* A, float* AT) {
    AT[0] = A[0];
    AT[1] = A[3];
    AT[2] = A[6];
    AT[3] = A[1];
    AT[4] = A[4];
    AT[5] = A[7];
    AT[6] = A[2];
    AT[7] = A[5];
    AT[8] = A[8];
}

// ============================================================================
// Main Kabsch Implementation
// ============================================================================

/**
 * Scalar Kabsch optimal superposition.
 */
template <>
void kabsch_align<ScalarBackend>(const float* P, const float* Q, int N, float* R, float* t,
                                 float* rmsd, float* centroid_P, float* centroid_Q) {
    // Step 1: Compute centroids
    float cP[3] = {0, 0, 0};
    float cQ[3] = {0, 0, 0};

    for (int i = 0; i < N; i++) {
        cP[0] += P[i * 3 + 0];
        cP[1] += P[i * 3 + 1];
        cP[2] += P[i * 3 + 2];

        cQ[0] += Q[i * 3 + 0];
        cQ[1] += Q[i * 3 + 1];
        cQ[2] += Q[i * 3 + 2];
    }

    float inv_N = 1.0f / N;
    cP[0] *= inv_N;
    cP[1] *= inv_N;
    cP[2] *= inv_N;

    cQ[0] *= inv_N;
    cQ[1] *= inv_N;
    cQ[2] *= inv_N;

    // Optionally return centroids
    if (centroid_P != nullptr) {
        centroid_P[0] = cP[0];
        centroid_P[1] = cP[1];
        centroid_P[2] = cP[2];
    }
    if (centroid_Q != nullptr) {
        centroid_Q[0] = cQ[0];
        centroid_Q[1] = cQ[1];
        centroid_Q[2] = cQ[2];
    }

    // Step 2: Compute covariance matrix H = sum[(P_i - cP) (Q_i - cQ)^T]
    float H[9] = {0};  // 3x3 covariance matrix

    for (int i = 0; i < N; i++) {
        float Px = P[i * 3 + 0] - cP[0];
        float Py = P[i * 3 + 1] - cP[1];
        float Pz = P[i * 3 + 2] - cP[2];

        float Qx = Q[i * 3 + 0] - cQ[0];
        float Qy = Q[i * 3 + 1] - cQ[1];
        float Qz = Q[i * 3 + 2] - cQ[2];

        // H += P_centered @ Q_centered^T (outer product)
        H[0] += Px * Qx;
        H[1] += Px * Qy;
        H[2] += Px * Qz;
        H[3] += Py * Qx;
        H[4] += Py * Qy;
        H[5] += Py * Qz;
        H[6] += Pz * Qx;
        H[7] += Pz * Qy;
        H[8] += Pz * Qz;
    }

    // Step 3: SVD of H: H = U * V^T
    // Check if H is symmetric (happens when P and Q have same shape)
    // Use relative symmetry threshold based on matrix magnitude (ChatGPT fix)
    float H_symmetry = 0.0f;
    float H_abs_sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            H_symmetry = std::max(H_symmetry, std::abs(H[i * 3 + j] - H[j * 3 + i]));
        }
    }
    for (int k = 0; k < 9; k++) {
        H_abs_sum += std::abs(H[k]);
    }
    float symmetry_threshold = std::max(1e-6f, H_abs_sum * 1e-5f);

    // Make a copy of H since SVD may modify it
    float H_copy[9];
    std::memcpy(H_copy, H, 9 * sizeof(float));

    float U[9], S[3], V[9];
    svd3x3::svd3x3<ScalarBackend>(H_copy, U, S, V);

    // For symmetric H (or near-symmetric), U should equal V
    // This happens when P and Q have the same shape (e.g., identity or pure translation)
    if (H_symmetry < symmetry_threshold) {
        // Use V for both (symmetric case)
        std::memcpy(U, V, 9 * sizeof(float));
    }

    // Step 4: Compute optimal rotation R = V U@
    float UT[9];
    transpose3x3(U, UT);
    matmul3x3(V, UT, R);

    // Step 5: Handle reflection case (det(R) < 0)
    float d = det3x3(R);
    if (d < 0.0f) {
        // Flip sign of third column of V (smallest singular value)
        V[2] = -V[2];
        V[5] = -V[5];
        V[8] = -V[8];

        // Recompute R = V U@
        matmul3x3(V, UT, R);
    }

    // Step 6: Compute translation t = cQ - R @ cP
    float R_cP[3];
    R_cP[0] = R[0] * cP[0] + R[1] * cP[1] + R[2] * cP[2];
    R_cP[1] = R[3] * cP[0] + R[4] * cP[1] + R[5] * cP[2];
    R_cP[2] = R[6] * cP[0] + R[7] * cP[1] + R[8] * cP[2];

    t[0] = cQ[0] - R_cP[0];
    t[1] = cQ[1] - R_cP[1];
    t[2] = cQ[2] - R_cP[2];

    // Step 7: Compute RMSD (if requested)
    if (rmsd != nullptr) {
        float sum_sq_dist = 0.0f;

        for (int i = 0; i < N; i++) {
            // Compute R @ P[i] + t
            float Px = P[i * 3 + 0];
            float Py = P[i * 3 + 1];
            float Pz = P[i * 3 + 2];

            float transformed_x = R[0] * Px + R[1] * Py + R[2] * Pz + t[0];
            float transformed_y = R[3] * Px + R[4] * Py + R[5] * Pz + t[1];
            float transformed_z = R[6] * Px + R[7] * Py + R[8] * Pz + t[2];

            // Compute squared distance to Q[i]
            float dx = transformed_x - Q[i * 3 + 0];
            float dy = transformed_y - Q[i * 3 + 1];
            float dz = transformed_z - Q[i * 3 + 2];

            sum_sq_dist += dx * dx + dy * dy + dz * dz;
        }

        *rmsd = std::sqrt(sum_sq_dist * inv_N);
    }
}

/**
 * Apply transformation to full structure.
 */
template <>
void apply_transformation<ScalarBackend>(const float* R, const float* t, const float* coords_in,
                                         float* coords_out, int L) {
    // Transform each atom: coords_out = R @ coords_in + t
    // Layout: [L residues] x [4 atoms] x [3 coords] = L x 4 x 3

    for (int i = 0; i < L * 4; i++) {
        float x = coords_in[i * 3 + 0];
        float y = coords_in[i * 3 + 1];
        float z = coords_in[i * 3 + 2];

        coords_out[i * 3 + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
        coords_out[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
        coords_out[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
    }
}

}  // namespace kabsch
}  // namespace pfalign
