#include "commands.h"
#include "npy_utils.h"
#include <iostream>
#include <fstream>
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/types/guide_tree_types.h"
#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace commands {

int tree_build(const std::string& distances_path,
               const std::string& output_path,
               const std::string& method,
               const std::vector<std::string>& labels) {
    try {
        // Validate inputs
        validation::validate_file_exists(distances_path, "distance matrix");

        // Parse NPY header to get dimensions
        std::cout << "[INFO] Loading distance matrix from " << distances_path << "\n";
        cli::NpyHeader header = cli::parse_npy_header(distances_path);

        if (header.shape.size() != 2) {
            throw errors::ValidationError(
                "Invalid distance matrix dimensions",
                "Expected 2D array (N, N), got " + std::to_string(header.shape.size()) + "D"
            );
        }

        if (header.shape[0] != header.shape[1]) {
            throw errors::ValidationError(
                "Distance matrix must be square",
                "Got dimensions " + std::to_string(header.shape[0]) + " x " + std::to_string(header.shape[1])
            );
        }

        int N = static_cast<int>(header.shape[0]);
        std::cout << "[INFO] Matrix size: " << N << " x " << N << " (" << N << " sequences)\n";

        // Load distance matrix
        std::vector<float> distances(N * N);
        if (!cli::load_npy_simple(distances_path, distances.data(), N * N)) {
            throw errors::FileNotFoundError(distances_path, "Could not load distance matrix");
        }

        // Create arena for tree construction
        pfalign::memory::GrowableArena arena;

        // Build tree using selected method
        std::cout << "[INFO] Building tree using " << method << " algorithm...\n";
        types::GuideTree tree;

        if (method == "upgma") {
            tree = tree_builders::build_upgma_tree(distances.data(), N, &arena);
        } else if (method == "nj") {
            tree = tree_builders::build_nj_tree(distances.data(), N, &arena);
        } else if (method == "bionj") {
            tree = tree_builders::build_bionj_tree(distances.data(), N, &arena);
        } else if (method == "mst") {
            tree = tree_builders::build_mst_tree(distances.data(), N, &arena);
        } else {
            throw errors::ValidationError(
                "Unknown tree building method: " + method,
                "Valid methods: upgma, nj, bionj, mst"
            );
        }

        // Prepare sequence labels
        std::vector<const char*> label_ptrs;
        std::vector<std::string> default_labels;

        if (labels.empty()) {
            // Generate default labels: seq0, seq1, seq2, ...
            for (int i = 0; i < N; ++i) {
                default_labels.push_back("seq" + std::to_string(i));
            }
            for (const auto& label : default_labels) {
                label_ptrs.push_back(label.c_str());
            }
        } else {
            // Validate label count
            if (static_cast<int>(labels.size()) != N) {
                throw errors::ValidationError(
                    "Label count mismatch",
                    "Provided " + std::to_string(labels.size()) + " labels for " + std::to_string(N) + " sequences"
                );
            }
            for (const auto& label : labels) {
                label_ptrs.push_back(label.c_str());
            }
        }

        // Convert tree to Newick format
        std::cout << "[INFO] Converting tree to Newick format...\n";
        const char* newick = tree.to_newick(label_ptrs.data(), &arena);

        // Write to output file
        std::cout << "[INFO] Writing tree to " << output_path << "\n";
        std::ofstream outfile(output_path);
        if (!outfile) {
            throw errors::FileNotFoundError(output_path, "Could not open output file for writing");
        }

        outfile << newick << "\n";
        outfile.close();

        std::cout << "[OK] Tree built and saved\n";
        std::cout << "  Method: " << method << "\n";
        std::cout << "  Sequences: " << N << "\n";
        std::cout << "  Output: " << output_path << "\n";

        return 0;

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
