/**
 * Scalar implementation of 3*3 SVD using Jacobi iteration.
 */

#include "svd3x3_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace pfalign {
namespace svd3x3 {

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Compute 3*3 matrix transpose.
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

/**
 * Compute 3*3 matrix multiplication: C = A * B
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
 * Apply Givens rotation to zero out A[p][q].
 *
 * Computes rotation angle theta and applies:
 * - Left: A <- G(p,q,theta)ᵀ A
 * - Right: A <- A G(p,q,theta)
 * - Accumulate: U <- U G(p,q,theta), V <- V G(p,q,theta)
 *
 * Givens rotation matrix G(p,q,theta):
 *   [1  ...      0      ...      0     ]
 *   [.   c      ...     s         .    ]
 *   [0  ...      1      ...      0     ]
 *   [.  -s      ...     c         .    ]
 *   [0  ...      0      ...      1     ]
 *
 * where c = cos(theta), s = sin(theta), and (p,q) are the rotation axes.
 */
static void jacobi_rotation(float* A,  // 3*3 matrix to diagonalize
                            float* U,  // 3*3 left accumulator
                            float* V,  // 3*3 right accumulator
                            int p,     // first rotation axis (0, 1, or 2)
                            int q      // second rotation axis (0, 1, or 2)
) {
    // Check if already zero (skip rotation)
    const float apq = A[p * 3 + q];
    if (std::abs(apq) < 1e-10f) {
        return;
    }

    const float app = A[p * 3 + p];
    const float aqq = A[q * 3 + q];

    // Compute rotation angle using Jacobi formula
    // For symmetric matrices: tan(2theta) = 2*A[p,q] / (A[q,q] - A[p,p])
    // For general matrices (SVD): work with AᵀA
    // Simplified: use atan2 for stability
    const float theta = 0.5f * std::atan2(2.0f * apq, aqq - app);
    const float c = std::cos(theta);
    const float s = std::sin(theta);

    // Apply rotation to A: A <- Gᵀ A G
    // This is the core of Jacobi: zeros out A[p,q] and A[q,p]

    // Update A (only affected rows/cols: p and q)
    // Note: temp storage declared but not needed for this implementation

    // Update rows p and q
    for (int j = 0; j < 3; j++) {
        const float ap = A[p * 3 + j];
        const float aq = A[q * 3 + j];
        A[p * 3 + j] = c * ap - s * aq;
        A[q * 3 + j] = s * ap + c * aq;
    }

    // Update columns p and q
    for (int i = 0; i < 3; i++) {
        const float ap = A[i * 3 + p];
        const float aq = A[i * 3 + q];
        A[i * 3 + p] = c * ap - s * aq;
        A[i * 3 + q] = s * ap + c * aq;
    }

    // Accumulate rotation in V (right singular vectors)
    // V <- V @ G(p,q,theta)
    for (int i = 0; i < 3; i++) {
        const float vp = V[i * 3 + p];
        const float vq = V[i * 3 + q];
        V[i * 3 + p] = c * vp - s * vq;
        V[i * 3 + q] = s * vp + c * vq;
    }

    // Accumulate rotation in U (left singular vectors)
    // U <- U @ G(p,q,theta)
    for (int i = 0; i < 3; i++) {
        const float up = U[i * 3 + p];
        const float uq = U[i * 3 + q];
        U[i * 3 + p] = c * up - s * uq;
        U[i * 3 + q] = s * up + c * uq;
    }
}

/**
 * Sort singular values in descending order and permute U, V accordingly.
 */
static void sort_singular_values(float* S, float* U, float* V) {
    // Simple bubble sort for 3 elements (optimal for small n)
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (S[j] > S[i]) {
                // Swap singular values
                std::swap(S[i], S[j]);

                // Swap corresponding columns in U
                for (int k = 0; k < 3; k++) {
                    std::swap(U[k * 3 + i], U[k * 3 + j]);
                }

                // Swap corresponding columns in V
                for (int k = 0; k < 3; k++) {
                    std::swap(V[k * 3 + i], V[k * 3 + j]);
                }
            }
        }
    }

    // Handle negative singular values (take absolute value, flip sign in U)
    for (int i = 0; i < 3; i++) {
        if (S[i] < 0.0f) {
            S[i] = -S[i];
            // Flip sign of corresponding column in U
            for (int k = 0; k < 3; k++) {
                U[k * 3 + i] = -U[k * 3 + i];
            }
        }
    }
}

// ============================================================================
// Main SVD Implementation
// ============================================================================

/**
 * Scalar 3*3 SVD using Jacobi iteration on AᵀA.
 */
template <>
void svd3x3<ScalarBackend>(float* A, float* U, float* S, float* V, int max_iterations,
                           float tolerance) {
    // Step 1: Form B = AᵀA (symmetric 3*3 matrix)
    float AT[9], B[9];
    transpose3x3(A, AT);
    matmul3x3(AT, A, B);  // B = AᵀA

    // Initialize V to identity (will accumulate rotations)
    // Initialize U to identity (will be computed later)
    for (int i = 0; i < 9; i++) {
        V[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        U[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    // Step 2: Diagonalize B using Jacobi iteration to get V and eigenvalues
    // Each sweep applies 3 rotations to zero off-diagonal elements
    for (int iter = 0; iter < max_iterations; iter++) {
        // Cyclic Jacobi: rotate pairs in order (0,1), (0,2), (1,2)
        jacobi_rotation(B, U, V, 0, 1);  // U is unused, but kept for API
        jacobi_rotation(B, U, V, 0, 2);
        jacobi_rotation(B, U, V, 1, 2);

        // Check convergence: sum of absolute values of off-diagonal elements
        const float off_norm = std::abs(B[0 * 3 + 1]) + std::abs(B[0 * 3 + 2]) +
                               std::abs(B[1 * 3 + 0]) + std::abs(B[1 * 3 + 2]) +
                               std::abs(B[2 * 3 + 0]) + std::abs(B[2 * 3 + 1]);

        if (off_norm < tolerance) {
            break;  // Converged
        }
    }

    // Step 3: Extract singular values from diagonal of B (eigenvalues of AᵀA)
    // Singular values = sqrt(eigenvalues)
    S[0] = std::sqrt(std::max(0.0f, B[0 * 3 + 0]));
    S[1] = std::sqrt(std::max(0.0f, B[1 * 3 + 1]));
    S[2] = std::sqrt(std::max(0.0f, B[2 * 3 + 2]));

    // Step 4: Sort singular values in descending order (and permute V)
    // Temporarily use U for sorting, then compute properly
    float S_sorted[3] = {S[0], S[1], S[2]};
    sort_singular_values(S_sorted, U, V);
    S[0] = S_sorted[0];
    S[1] = S_sorted[1];
    S[2] = S_sorted[2];

    // Step 5: Compute U from U = A V S⁻^1
    // For each column i of U: u_i = (A v_i) / s_i
    float max_sigma = std::max({S[0], S[1], S[2]});
    float zero_threshold = max_sigma * 1e-6f;

    int num_nonzero = 0;
    for (int i = 0; i < 3; i++) {
        if (S[i] > zero_threshold) {
            // Extract column i of V
            float v_col[3] = {V[0 * 3 + i], V[1 * 3 + i], V[2 * 3 + i]};

            // Compute A @ v_col
            float temp[3];
            temp[0] = A[0 * 3 + 0] * v_col[0] + A[0 * 3 + 1] * v_col[1] + A[0 * 3 + 2] * v_col[2];
            temp[1] = A[1 * 3 + 0] * v_col[0] + A[1 * 3 + 1] * v_col[1] + A[1 * 3 + 2] * v_col[2];
            temp[2] = A[2 * 3 + 0] * v_col[0] + A[2 * 3 + 1] * v_col[1] + A[2 * 3 + 2] * v_col[2];

            // Normalize: u_i = (A v_i) / s_i
            U[0 * 3 + i] = temp[0] / S[i];
            U[1 * 3 + i] = temp[1] / S[i];
            U[2 * 3 + i] = temp[2] / S[i];
            num_nonzero++;
        } else {
            // Zero singular value - initialize to zero, will complete basis later
            U[0 * 3 + i] = 0.0f;
            U[1 * 3 + i] = 0.0f;
            U[2 * 3 + i] = 0.0f;
        }
    }

    // Complete orthonormal basis for zero singular values using Gram-Schmidt
    if (num_nonzero < 3) {
        // Generate candidate vectors and orthogonalize
        float candidates[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        for (int i = 0; i < 3; i++) {
            if (S[i] > zero_threshold)
                continue;  // Skip non-zero singular values

            // Try each candidate vector
            float best_norm = 0.0f;
            float best_vec[3] = {0, 0, 0};

            for (int c = 0; c < 3; c++) {
                float vec[3] = {candidates[c][0], candidates[c][1], candidates[c][2]};

                // Orthogonalize against previous columns
                for (int j = 0; j < i; j++) {
                    float dot =
                        vec[0] * U[0 * 3 + j] + vec[1] * U[1 * 3 + j] + vec[2] * U[2 * 3 + j];
                    vec[0] -= dot * U[0 * 3 + j];
                    vec[1] -= dot * U[1 * 3 + j];
                    vec[2] -= dot * U[2 * 3 + j];
                }

                // Compute norm
                float norm = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);

                if (norm > best_norm) {
                    best_norm = norm;
                    best_vec[0] = vec[0];
                    best_vec[1] = vec[1];
                    best_vec[2] = vec[2];
                }
            }

            // Normalize and store
            if (best_norm > 1e-10f) {
                U[0 * 3 + i] = best_vec[0] / best_norm;
                U[1 * 3 + i] = best_vec[1] / best_norm;
                U[2 * 3 + i] = best_vec[2] / best_norm;
            }
        }
    }
}

// ============================================================================
}  // namespace svd3x3
}  // namespace pfalign
