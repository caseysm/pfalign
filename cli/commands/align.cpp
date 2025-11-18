#include "commands.h"
#include "npy_utils.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/modules/decode/decode.h"

namespace pfalign {
namespace commands {

using pfalign::ScalarBackend;
namespace sw = pfalign::smith_waterman;

int alignment_forward(const std::string& similarity_path,
                     const std::string& output_path,
                     float gap_open,
                     float gap_extend,
                     float temperature) {
    try {
        // Validate inputs
        validation::validate_file_exists(similarity_path, "similarity matrix");

        // Parse NPY header to get dimensions
        std::cout << "[INFO] Loading similarity matrix from " << similarity_path << "\n";
        cli::NpyHeader header = cli::parse_npy_header(similarity_path);

        if (header.shape.size() != 2) {
            throw errors::ValidationError(
                "Invalid similarity matrix dimensions",
                "Expected 2D array (L1, L2), got " + std::to_string(header.shape.size()) + "D"
            );
        }

        if (header.dtype != "<f4") {
            throw errors::FormatError(
                "Unsupported dtype: " + header.dtype,
                "Similarity matrix must be float32 ('<f4')"
            );
        }

        int L1 = static_cast<int>(header.shape[0]);
        int L2 = static_cast<int>(header.shape[1]);

        std::cout << "[INFO] Matrix dimensions: " << L1 << " x " << L2 << "\n";

        // Load similarity matrix data
        std::vector<float> sim_data(L1 * L2);
        if (!cli::load_npy_simple(similarity_path, sim_data.data(), L1 * L2)) {
            throw errors::FileNotFoundError(similarity_path, "Could not load similarity matrix");
        }

        // Configure Smith-Waterman
        sw::SWConfig config;
        config.gap_open = gap_open;
        config.gap_extend = gap_extend;
        config.temperature = temperature;

        // Allocate output arrays for forward scores (L1, L2, 3) and partition
        std::vector<float> forward_scores(L1 * L2 * 3);
        float partition = 0.0f;

        // Call forward algorithm
        std::cout << "[INFO] Computing forward scores (gap_open=" << gap_open
                  << ", gap_extend=" << gap_extend
                  << ", temperature=" << temperature << ")...\n";

        sw::smith_waterman_jax_affine_flexible<ScalarBackend>(
            sim_data.data(), L1, L2, config,
            forward_scores.data(), &partition
        );

        // Save forward scores as 3D NPY array (L1, L2, 3)
        std::cout << "[INFO] Saving forward scores to " << output_path << "\n";
        cli::save_npy_3d(output_path, forward_scores.data(), L1, L2, 3);

        std::cout << "[OK] Forward scores computed and saved\n";
        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int alignment_backward(const std::string& forward_path,
                      const std::string& similarity_path,
                      float partition,
                      const std::string& output_path,
                      float gap_open,
                      float gap_extend,
                      float temperature) {
    try {
        // Validate inputs
        validation::validate_file_exists(forward_path, "forward scores");
        validation::validate_file_exists(similarity_path, "similarity matrix");

        // Parse forward scores header
        std::cout << "[INFO] Loading forward scores from " << forward_path << "\n";
        cli::NpyHeader fwd_header = cli::parse_npy_header(forward_path);

        if (fwd_header.shape.size() != 3 || fwd_header.shape[2] != 3) {
            throw errors::ValidationError(
                "Invalid forward scores dimensions",
                "Expected 3D array (L1, L2, 3), got shape " + std::to_string(fwd_header.shape.size()) + "D"
            );
        }

        int L1 = static_cast<int>(fwd_header.shape[0]);
        int L2 = static_cast<int>(fwd_header.shape[1]);

        // Parse similarity matrix header
        std::cout << "[INFO] Loading similarity matrix from " << similarity_path << "\n";
        cli::NpyHeader sim_header = cli::parse_npy_header(similarity_path);

        if (sim_header.shape.size() != 2) {
            throw errors::ValidationError(
                "Invalid similarity matrix dimensions",
                "Expected 2D array (L1, L2)"
            );
        }

        if (sim_header.shape[0] != fwd_header.shape[0] || sim_header.shape[1] != fwd_header.shape[1]) {
            throw errors::ValidationError(
                "Dimension mismatch",
                "Forward scores and similarity matrix must have same dimensions"
            );
        }

        // Load forward scores
        std::vector<float> forward_scores(L1 * L2 * 3);
        if (!cli::load_npy_simple(forward_path, forward_scores.data(), L1 * L2 * 3)) {
            throw errors::FileNotFoundError(forward_path, "Could not load forward scores");
        }

        // Load similarity matrix
        std::vector<float> sim_data(L1 * L2);
        if (!cli::load_npy_simple(similarity_path, sim_data.data(), L1 * L2)) {
            throw errors::FileNotFoundError(similarity_path, "Could not load similarity matrix");
        }

        // Configure Smith-Waterman
        sw::SWConfig config;
        config.gap_open = gap_open;
        config.gap_extend = gap_extend;
        config.temperature = temperature;

        // Allocate output array for posterior probabilities (L1, L2)
        std::vector<float> posterior(L1 * L2);

        // Call backward algorithm
        std::cout << "[INFO] Computing backward scores (partition=" << partition << ")...\n";

        sw::smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
            forward_scores.data(), sim_data.data(), L1, L2, config, partition,
            posterior.data(), nullptr
        );

        // Save posterior probabilities as 2D NPY array (L1, L2)
        std::cout << "[INFO] Saving posterior probabilities to " << output_path << "\n";
        cli::save_npy_2d(output_path, posterior.data(), L1, L2);

        std::cout << "[OK] Posterior probabilities computed and saved\n";
        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int alignment_decode(const std::string& posterior_path,
                    const std::string& seq1,
                    const std::string& seq2,
                    const std::string& /* output_path */,
                    float /* gap_penalty */) {
    try {
        // Validate inputs
        validation::validate_file_exists(posterior_path, "posterior probabilities");

        if (seq1.empty() || seq2.empty()) {
            throw errors::ValidationError(
                "Sequences cannot be empty",
                "Provide valid amino acid sequences for seq1 and seq2"
            );
        }

        // Parse posterior probabilities header
        std::cout << "[INFO] Loading posterior probabilities from " << posterior_path << "\n";
        cli::NpyHeader post_header = cli::parse_npy_header(posterior_path);

        if (post_header.shape.size() != 2) {
            throw errors::ValidationError(
                "Invalid posterior dimensions",
                "Expected 2D array (L1, L2)"
            );
        }

        int L1 = static_cast<int>(post_header.shape[0]);
        int L2 = static_cast<int>(post_header.shape[1]);

        // Validate sequence lengths
        if (seq1.length() != static_cast<size_t>(L1)) {
            throw errors::ValidationError(
                "Sequence length mismatch",
                "seq1 length (" + std::to_string(seq1.length()) +
                ") doesn't match posterior dimension L1 (" + std::to_string(L1) + ")"
            );
        }

        if (seq2.length() != static_cast<size_t>(L2)) {
            throw errors::ValidationError(
                "Sequence length mismatch",
                "seq2 length (" + std::to_string(seq2.length()) +
                ") doesn't match posterior dimension L2 (" + std::to_string(L2) + ")"
            );
        }

        // Load posterior probabilities
        std::vector<float> posterior(L1 * L2);
        if (!cli::load_npy_simple(posterior_path, posterior.data(), L1 * L2)) {
            throw errors::FileNotFoundError(posterior_path, "Could not load posterior probabilities");
        }

        // Viterbi decoding not yet implemented in current API
        // TODO: Implement alignment decoding from posteriors
        std::cerr << "[ERROR] Viterbi decoding is not yet implemented in the current API\n";
        std::cerr << "[ERROR] This command will be re-enabled when the decode module is updated\n";
        return 1;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace commands
}  // namespace pfalign
