/**
 * Data structures for alignment results and decoding.
 *
 * Defines types used throughout the alignment pipeline:
 * - AlignmentPair: Single alignment entry (residue pair or gap)
 * - TracebackDirection: DP traceback state
 *
 * These types are shared across primitives, modules, and I/O.
 */

#pragma once

#include <cstdint>

namespace pfalign {

/**
 * Single entry in an alignment path.
 *
 * Represents either a residue-to-residue match or a gap:
 * - Match: Both i and j are non-negative
 * - Gap in seq1: i = -1, j >= 0
 * - Gap in seq2: i >= 0, j = -1
 *
 * The posterior field stores the posterior probability from the alignment
 * matrix for matches, or 0.0 for gaps (which have no posterior).
 *
 * Example alignment path:
 *   [(0, 0, 0.8), (1, 1, 0.9), (2, -1, 0.0), (3, 2, 0.7)]
 *   Interpretation:
 *     0->0 (match, p=0.8)
 *     1->1 (match, p=0.9)
 *     2->gap (gap in seq2)
 *     3->2 (match after gap, p=0.7)
 */
struct AlignmentPair {
    int i;            ///< Position in sequence 1 (-1 for gap in seq1)
    int j;            ///< Position in sequence 2 (-1 for gap in seq2)
    float posterior;  ///< Posterior probability (0.0 for gaps)
};

/**
 * Traceback direction for Viterbi dynamic programming.
 *
 * Encoded as uint8_t for memory efficiency (201*201 matrix = 40KB).
 *
 * Direction meanings:
 * - MATCH: Came from diagonal (i-1, j-1) - residues aligned
 * - GAP_SEQ1: Came from left (i, j-1) - gap inserted in seq1
 * - GAP_SEQ2: Came from top (i-1, j) - gap inserted in seq2
 * - START: Boundary condition at (0, 0)
 */
enum TracebackDirection : uint8_t {
    MATCH = 0,     ///< From diagonal (i-1, j-1)
    GAP_SEQ1 = 1,  ///< From left (i, j-1) - gap in seq1
    GAP_SEQ2 = 2,  ///< From top (i-1, j) - gap in seq2
    START = 3      ///< Boundary condition
};

}  // namespace pfalign
