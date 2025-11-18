/**
 * Kabsch Algorithm: Optimal Rigid-Body Superposition
 *
 * Finds the optimal rotation R and translation t that minimizes RMSD
 * between two sets of 3D points.
 *
 * Application: Protein structure alignment
 * - Input: Aligned Calpha atom coordinates from two proteins
 * - Output: Rotation matrix, translation vector, RMSD
 *
 * Algorithm:
 * 1. Compute centroids: c_P = (1/N) Sigma P_i, c_Q = (1/N) Sigma Q_i
 * 2. Center point sets: P' = P - c_P, Q' = Q - c_Q
 * 3. Compute covariance matrix: H = Sigma P'_i Q'_iᵀ (3*3)
 * 4. SVD of H: H = U Sigma Vᵀ
 * 5. Construct rotation: R = V U^T, handling reflection if det(R) < 0
 * 6. Compute translation: t = c_Q - R c_P
 * 7. Compute RMSD: sqrt((1/N) Sigma ||R P_i + t - Q_i||^2)
 *
 * References:
 * - Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors"
 * - Kabsch, W. (1978). "A discussion of the solution for the best rotation..."
 */

#pragma once

#include "pfalign/dispatch/backend_traits.h"

namespace pfalign {
namespace kabsch {

/**
 * Kabsch optimal superposition of 3D point sets.
 *
 * Finds rotation R and translation t minimizing:
 *   RMSD = sqrt((1/N) Sigmaᵢ ||R Pᵢ + t - Qᵢ||^2)
 *
 * Point sets:
 * - P [N * 3]: Source coordinates (row-major)
 * - Q [N * 3]: Target coordinates (row-major)
 * - N: Number of point pairs (must be >= 3 for unique solution)
 *
 * Outputs:
 * - R [9]: Rotation matrix (3*3 row-major, orthogonal, det(R) = +1)
 * - t [3]: Translation vector
 * - rmsd: Root-mean-square deviation after alignment
 *
 * Properties:
 * - Guarantees det(R) = +1 (proper rotation, not reflection)
 * - Minimizes RMSD globally (optimal solution)
 * - Stable for well-conditioned point sets
 *
 * @tparam Backend Computation backend (ScalarBackend, NEONBackend, CUDABackend)
 * @param P Source coordinates [N * 3] (row-major: [x₀,y₀,z₀, x₁,y₁,z₁, ...])
 * @param Q Target coordinates [N * 3] (row-major)
 * @param N Number of point pairs (>= 3 required)
 * @param R Output rotation matrix [9] (3*3 row-major)
 * @param t Output translation vector [3]
 * @param rmsd Output RMSD after optimal alignment
 * @param centroid_P Optional output: centroid of P [3] (can be nullptr)
 * @param centroid_Q Optional output: centroid of Q [3] (can be nullptr)
 *
 * Example usage:
 * ```cpp
 *   // Aligned Calpha coordinates from two proteins
 *   float P[N*3] = {...};  // Protein 1 CA coords
 *   float Q[N*3] = {...};  // Protein 2 CA coords
 *
 *   float R[9], t[3], rmsd;
 *   kabsch_align<ScalarBackend>(P, Q, N, R, t, &rmsd);
 *
 *   // Apply transformation: P_aligned = R @ P + t
 *   // RMSD = sqrt((1/N) Sigma ||P_aligned - Q||^2)
 * ```
 *
 * Performance:
 * - Scalar: O(N) for centering + O(1) for 3*3 SVD ~10-20 mus for N=200
 * - NEON: Vectorize centroid computation ~5-10 mus for N=200
 * - CUDA: Parallel reduction for centroid, rest on CPU ~same as scalar
 *
 * Future SIMD optimizations:
 * - Vectorize centroid computation (4-wide NEON/AVX)
 * - Vectorize covariance matrix accumulation
 * - SVD remains scalar (3*3 too small for SIMD)
 */
template <typename Backend>
void kabsch_align(const float* P,               // [N * 3] source coordinates
                  const float* Q,               // [N * 3] target coordinates
                  int N,                        // number of point pairs
                  float* R,                     // [9] rotation matrix output (3*3 row-major)
                  float* t,                     // [3] translation vector output
                  float* rmsd,                  // RMSD output (can be nullptr if not needed)
                  float* centroid_P = nullptr,  // Optional: centroid of P output
                  float* centroid_Q = nullptr   // Optional: centroid of Q output
);

/**
 * Apply Kabsch transformation to full protein structure.
 *
 * Transforms all atoms using rotation and translation from kabsch_align:
 *   coords_out = R @ coords_in + t
 *
 * Used to transform entire protein structure after computing alignment
 * from Calpha atoms only.
 *
 * Coordinate layout: [L * 4 * 3] = [L residues] * [4 atoms per residue] * [3 coords]
 * Atoms per residue: N, CA, C, O (backbone atoms)
 *
 * @tparam Backend Computation backend
 * @param R Rotation matrix [9] from kabsch_align
 * @param t Translation vector [3] from kabsch_align
 * @param coords_in Input coordinates [L * 4 * 3] (row-major)
 * @param coords_out Output transformed coordinates [L * 4 * 3] (row-major)
 * @param L Number of residues
 *
 * Example:
 * ```cpp
 *   // Align CA atoms to get R, t
 *   kabsch_align<ScalarBackend>(ca1, ca2, N_aligned, R, t, &rmsd);
 *
 *   // Apply transformation to full structure (all backbone atoms)
 *   float coords_aligned[L1 * 4 * 3];
 *   apply_transformation<ScalarBackend>(
 *       R, t, coords1, coords_aligned, L1
 *   );
 * ```
 *
 * Performance:
 * - Scalar: O(4L) ~1 mus per 100 residues
 * - NEON: Vectorize 3D rotation ~500 ns per 100 residues
 */
template <typename Backend>
void apply_transformation(const float* R,          // [9] rotation matrix (3*3 row-major)
                          const float* t,          // [3] translation vector
                          const float* coords_in,  // [L * 4 * 3] input coordinates
                          float* coords_out,       // [L * 4 * 3] output coordinates
                          int L                    // number of residues
);

}  // namespace kabsch
}  // namespace pfalign
