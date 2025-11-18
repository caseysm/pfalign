/**
 * Distance Matrix Computation for Protein Structures
 *
 * Utilities for computing pairwise distance matrices between C-alpha atoms.
 * Used as foundation for LDDT and DALI scoring.
 *
 * Reference:
 * - Mariani et al. (2013). "LDDT: a local superposition-free score..."
 * - Holm & Sander (1993). "Protein structure comparison by alignment of distance matrices"
 */

#pragma once

#include "pfalign/dispatch/backend_traits.h"
#include <cmath>
#include <vector>

namespace pfalign {
namespace structural_metrics {

/**
 * Extract C-alpha (CA) atom coordinates from backbone coordinates.
 *
 * Backbone coords are stored as [L × 4 × 3] where the 4 atoms are N, CA, C, O.
 * This function extracts just the CA atoms (index 1) into [L × 3] format.
 *
 * @param backbone_coords Backbone coordinates [L × 4 × 3] from PDB parser
 * @param L Number of residues
 * @param ca_coords Output buffer [L × 3] for CA coordinates
 *
 * Example:
 * ```cpp
 *   PDBParser parser;
 *   auto protein = parser.parse_file("1crn.pdb");
 *   auto backbone = protein.get_backbone_coords(0);  // [L × 4 × 3]
 *
 *   int L = backbone.size() / 12;  // L residues × 4 atoms × 3 dims
 *   std::vector<float> ca_coords(L * 3);
 *   extract_ca_atoms(backbone.data(), L, ca_coords.data());
 * ```
 */
inline void extract_ca_atoms(const float* backbone_coords, int L, float* ca_coords) {
    for (int i = 0; i < L; i++) {
        // CA is atom index 1 (N=0, CA=1, C=2, O=3)
        const float* ca = &backbone_coords[(i * 4 + 1) * 3];
        ca_coords[i * 3 + 0] = ca[0];  // x
        ca_coords[i * 3 + 1] = ca[1];  // y
        ca_coords[i * 3 + 2] = ca[2];  // z
    }
}

/**
 * Compute symmetric pairwise distance matrix for C-alpha atoms.
 *
 * Computes Euclidean distances between all pairs of CA atoms:
 *   dist_mx[i][j] = ||ca[i] - ca[j]||₂
 *
 * The matrix is symmetric (dist[i][j] = dist[j][i]) and diagonal is zero.
 * Only the upper triangle (j > i) is computed, then copied to lower triangle.
 *
 * @tparam Backend Computation backend (ScalarBackend, etc.)
 * @param ca_coords C-alpha coordinates [L × 3]
 * @param L Number of residues
 * @param dist_mx Output distance matrix [L × L] stored in row-major order
 *
 * Memory layout:
 *   dist_mx[i * L + j] = distance between residue i and residue j
 *
 * Example:
 * ```cpp
 *   std::vector<float> ca_coords = {...};  // [L × 3]
 *   std::vector<float> dist_mx(L * L);
 *   compute_distance_matrix<ScalarBackend>(ca_coords.data(), L, dist_mx.data());
 *
 *   // Access distance between residue 5 and 10
 *   float d = dist_mx[5 * L + 10];
 * ```
 *
 * Properties:
 * - Symmetric: dist[i][j] = dist[j][i]
 * - Diagonal: dist[i][i] = 0
 * - Non-negative: dist[i][j] ≥ 0
 *
 * Performance:
 * - Scalar: O(L²) ~2ms for L=200, ~30ms for L=500
 * - SIMD: Future optimization ~50% faster
 *
 * Complexity:
 * - Time: O(L²)
 * - Space: O(L²) for output matrix
 */
template <typename Backend>
void compute_distance_matrix(const float* ca_coords,  // [L × 3] C-alpha coordinates
                             int L,                   // number of residues
                             float* dist_mx           // Output: [L × L] distance matrix
);

/**
 * Compute distance between two 3D points.
 *
 * Helper function for distance calculations.
 *
 * @param p1 First point [x, y, z]
 * @param p2 Second point [x, y, z]
 * @return Euclidean distance ||p1 - p2||₂
 */
inline float compute_distance_3d(const float* p1, const float* p2) {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    float dz = p1[2] - p2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace structural_metrics
}  // namespace pfalign
