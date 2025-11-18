#include "alignment_adapters.h"
#include <stdexcept>
#include <sstream>

namespace pfalign {
namespace adapters {

void pairwise_to_columns(const AlignmentPair* path, int path_length, int seq1_idx, int seq2_idx,
                         msa::AlignmentColumn* columns) {
    // Each AlignmentPair in the path becomes one MSA column
    // Each column has positions for both sequences

    for (int col = 0; col < path_length; ++col) {
        const AlignmentPair& pair = path[col];

        // Initialize column with 2 positions (binary alignment)
        columns[col].positions.resize(2);

        // Position for sequence 1
        columns[col].positions[0].seq_idx = seq1_idx;
        columns[col].positions[0].pos = pair.i;  // -1 if gap

        // Position for sequence 2
        columns[col].positions[1].seq_idx = seq2_idx;
        columns[col].positions[1].pos = pair.j;  // -1 if gap
    }
}

msa::AlignmentColumn* alignment_result_to_columns(const pairwise::AlignmentResult& result,
                                                  int seq1_idx, int seq2_idx,
                                                  pfalign::memory::GrowableArena* arena,
                                                  int* out_aligned_length) {
    // Validate inputs
    if (result.alignment_path == nullptr) {
        throw std::invalid_argument("AlignmentResult has null alignment_path");
    }

    if (result.path_length == 0) {
        throw std::invalid_argument("AlignmentResult has zero path_length");
    }

    if (arena == nullptr) {
        throw std::invalid_argument("Arena pointer is null");
    }

    // Allocate columns array from arena
    int aligned_length = result.path_length;
    msa::AlignmentColumn* columns = arena->allocate<msa::AlignmentColumn>(aligned_length);

    // Placement new for each column (initialize std::vector members)
    for (int i = 0; i < aligned_length; ++i) {
        new (&columns[i]) msa::AlignmentColumn();
    }

    // Convert path to columns
    pairwise_to_columns(result.alignment_path, result.path_length, seq1_idx, seq2_idx, columns);

    // Output aligned length
    if (out_aligned_length != nullptr) {
        *out_aligned_length = aligned_length;
    }

    return columns;
}

void validate_alignment_path(const AlignmentPair* path, int path_length, int L1, int L2) {
    if (path == nullptr) {
        throw std::invalid_argument("Alignment path is null");
    }

    if (path_length < 0) {
        throw std::invalid_argument("Path length is negative");
    }

    if (L1 < 0 || L2 < 0) {
        throw std::invalid_argument("Sequence lengths must be non-negative");
    }

    for (int k = 0; k < path_length; ++k) {
        const AlignmentPair& pair = path[k];

        // Check for invalid indices
        if (pair.i < -1 || pair.j < -1) {
            std::ostringstream msg;
            msg << "Invalid index at position " << k << ": i=" << pair.i << ", j=" << pair.j
                << " (indices must be >= -1)";
            throw std::invalid_argument(msg.str());
        }

        // Check for empty column (both gaps)
        if (pair.i == -1 && pair.j == -1) {
            std::ostringstream msg;
            msg << "Empty column at position " << k << " (both i and j are -1)";
            throw std::invalid_argument(msg.str());
        }

        // Check bounds
        if (pair.i >= L1) {
            std::ostringstream msg;
            msg << "Index i out of bounds at position " << k << ": i=" << pair.i << " >= L1=" << L1;
            throw std::invalid_argument(msg.str());
        }

        if (pair.j >= L2) {
            std::ostringstream msg;
            msg << "Index j out of bounds at position " << k << ": j=" << pair.j << " >= L2=" << L2;
            throw std::invalid_argument(msg.str());
        }

        // Check monotonic increase (allowing gaps)
        if (k > 0) {
            const AlignmentPair& prev = path[k - 1];

            // If not a gap, index should increase or stay the same
            if (pair.i != -1 && prev.i != -1 && pair.i < prev.i) {
                std::ostringstream msg;
                msg << "Non-monotonic i indices at position " << k << ": prev.i=" << prev.i
                    << ", curr.i=" << pair.i;
                throw std::invalid_argument(msg.str());
            }

            if (pair.j != -1 && prev.j != -1 && pair.j < prev.j) {
                std::ostringstream msg;
                msg << "Non-monotonic j indices at position " << k << ": prev.j=" << prev.j
                    << ", curr.j=" << pair.j;
                throw std::invalid_argument(msg.str());
            }
        }
    }
}

}  // namespace adapters
}  // namespace pfalign
