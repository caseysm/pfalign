#include "commands.h"
#include <iostream>
#include <memory>

// I/O
#include "pfalign/io/pdb_parser.h"
#include "pfalign/io/mmcif_parser.h"
#include "pfalign/io/protein_structure.h"

// MPNN
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"

// Save NPY
#include "pfalign/tools/weights/save_npy.h"

// Backend
#include "pfalign/dispatch/backend_traits.h"

// Error handling
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"

namespace pfalign {
namespace commands {

int encode(const std::string& pdb_path, const std::string& output_path, int k_neighbors,
           int chain) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign Encode Command\n";
    std::cout << "===========================================\n\n";
    std::cout << "Input:        " << pdb_path << "\n";
    std::cout << "Output:       " << output_path << "\n";
    std::cout << "k_neighbors:  " << k_neighbors << "\n";
    std::cout << "Chain:        " << chain << "\n\n";

    // ========================================================================
    // Step 1: Load MPNN weights
    // ========================================================================
    std::cout << "Step 1: Loading MPNN weights...\n";

    mpnn::MPNNConfig config;
    mpnn::MPNNWeights weights(0);

    try {
        auto result = weights::load_embedded_mpnn_weights();
        weights = std::move(std::get<0>(result));
        config = std::get<1>(result);
    } catch (const std::exception& e) {
        auto error = errors::messages::weights_loading_failed(e.what());
        std::cerr << error.formatted() << "\n";
        return 1;
    }

    // Override k_neighbors if specified
    config.k_neighbors = k_neighbors;

    std::cout << "✓ Loaded weights\n";
    std::cout << "  Hidden dim:   " << config.hidden_dim << "\n";
    std::cout << "  Num layers:   " << config.num_layers << "\n";
    std::cout << "  k_neighbors:  " << config.k_neighbors << "\n\n";

    // ========================================================================
    // Step 2: Parse PDB file (auto-detect PDB vs mmCIF)
    // ========================================================================
    std::cout << "Step 2: Parsing structure file...\n";

    io::Protein protein;
    bool is_cif = false;

    // Auto-detect file format by checking extension
    if (pdb_path.find(".cif") != std::string::npos ||
        pdb_path.find(".mmcif") != std::string::npos) {
        is_cif = true;
    }

    try {
        if (is_cif) {
            io::mmCIFParser parser;
            protein = parser.parse_file(pdb_path);
            std::cout << "✓ Parsed mmCIF file\n";
        } else {
            io::PDBParser parser;
            protein = parser.parse_file(pdb_path);
            std::cout << "✓ Parsed PDB file\n";
        }
    } catch (const std::exception& e) {
        // Try the other format
        try {
            if (is_cif) {
                io::PDBParser parser;
                protein = parser.parse_file(pdb_path);
                std::cout << "✓ Parsed as PDB file (mmCIF failed)\n";
            } else {
                io::mmCIFParser parser;
                protein = parser.parse_file(pdb_path);
                std::cout << "✓ Parsed as mmCIF file (PDB failed)\n";
            }
        } catch (const std::exception& e2) {
            auto error = errors::messages::file_parse_error(
                pdb_path,
                is_cif ? "mmCIF/PDB" : "PDB/mmCIF",
                e2.what()
            );
            std::cerr << error.formatted() << "\n";
            return 1;
        }
    }

    std::cout << "  Chains: " << protein.num_chains() << "\n";

    // Validate chain index
    try {
        validation::validate_chain_index(
            static_cast<int>(protein.num_chains()),
            chain,
            pdb_path
        );
    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    }

    // Get backbone coordinates
    int L = protein.get_chain(chain).size();
    std::cout << "  Chain " << chain << ": " << L << " residues\n\n";

    auto coords = protein.get_backbone_coords(chain);

    // ========================================================================
    // Step 3: Compute MPNN embeddings
    // ========================================================================
    std::cout << "Step 3: Computing MPNN embeddings...\n";

    // Allocate embedding buffer
    std::vector<float> embeddings(L * config.hidden_dim);

    // Create workspace
    mpnn::MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim, config.num_rbf);

    // Initialize residue indices and chain labels
    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    // Run MPNN forward pass
    try {
        mpnn::mpnn_forward<ScalarBackend>(coords.data(), L, weights, config, embeddings.data(),
                                          &workspace);
    } catch (const std::exception& e) {
        auto error = errors::messages::mpnn_encoding_failed(pdb_path, e.what());
        std::cerr << error.formatted() << "\n";
        return 1;
    }

    std::cout << "✓ Computed embeddings\n";
    std::cout << "  Shape: (" << L << ", " << config.hidden_dim << ")\n\n";

    // ========================================================================
    // Step 4: Save embeddings to NPY file
    // ========================================================================
    std::cout << "Step 4: Saving embeddings to " << output_path << "...\n";

    try {
        save_npy_2d(output_path, embeddings.data(), L, config.hidden_dim);
    } catch (const std::exception& e) {
        auto error = errors::messages::file_write_error(output_path, e.what());
        std::cerr << error.formatted() << "\n";
        return 1;
    }

    std::cout << "✓ Saved embeddings\n\n";

    std::cout << "===========================================\n";
    std::cout << "  Encode Complete\n";
    std::cout << "===========================================\n";

    return 0;
}

int encode_batch(const std::vector<std::string>& input_paths, const std::string& output_dir,
                 int k_neighbors, int chain) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign Encode Batch Command\n";
    std::cout << "===========================================\n\n";
    std::cout << "Input files:  " << input_paths.size() << "\n";
    std::cout << "Output dir:   " << output_dir << "\n";
    std::cout << "k_neighbors:  " << k_neighbors << "\n";
    std::cout << "Chain:        " << chain << "\n\n";

    // ========================================================================
    // Step 1: Load MPNN weights (once for all structures)
    // ========================================================================
    std::cout << "Step 1: Loading MPNN weights...\n";

    mpnn::MPNNConfig config;
    mpnn::MPNNWeights weights(0);

    try {
        auto result = weights::load_embedded_mpnn_weights();
        weights = std::move(std::get<0>(result));
        config = std::get<1>(result);
    } catch (const std::exception& e) {
        auto error = errors::messages::weights_loading_failed(e.what());
        std::cerr << error.formatted() << "\n";
        return 1;
    }

    config.k_neighbors = k_neighbors;
    std::cout << "✓ Loaded weights\n\n";

    // ========================================================================
    // Step 2: Process each structure
    // ========================================================================
    int success_count = 0;
    int failure_count = 0;

    for (size_t i = 0; i < input_paths.size(); i++) {
        const std::string& pdb_path = input_paths[i];
        std::cout << "Processing [" << (i + 1) << "/" << input_paths.size() << "]: " << pdb_path
                  << "\n";

        // Determine output filename
        size_t last_slash = pdb_path.find_last_of("/\\");
        std::string filename = (last_slash != std::string::npos)
                                 ? pdb_path.substr(last_slash + 1)
                                 : pdb_path;
        size_t last_dot = filename.find_last_of(".");
        std::string base_name = (last_dot != std::string::npos)
                                  ? filename.substr(0, last_dot)
                                  : filename;
        std::string output_path = output_dir + "/" + base_name + ".npy";

        try {
            // Parse structure
            io::Protein protein;
            bool is_cif = (pdb_path.find(".cif") != std::string::npos ||
                          pdb_path.find(".mmcif") != std::string::npos);

            try {
                if (is_cif) {
                    io::mmCIFParser parser;
                    protein = parser.parse_file(pdb_path);
                } else {
                    io::PDBParser parser;
                    protein = parser.parse_file(pdb_path);
                }
            } catch (const std::exception&) {
                // Try the other format
                if (is_cif) {
                    io::PDBParser parser;
                    protein = parser.parse_file(pdb_path);
                } else {
                    io::mmCIFParser parser;
                    protein = parser.parse_file(pdb_path);
                }
            }

            // Validate chain
            validation::validate_chain_index(
                static_cast<int>(protein.num_chains()),
                chain,
                pdb_path
            );

            // Get coordinates
            int L = protein.get_chain(chain).size();
            auto coords = protein.get_backbone_coords(chain);

            // Compute embeddings
            std::vector<float> embeddings(L * config.hidden_dim);
            mpnn::MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim,
                                         config.num_rbf);

            for (int j = 0; j < L; j++) {
                workspace.residue_idx[j] = j;
                workspace.chain_labels[j] = 0;
            }

            mpnn::mpnn_forward<ScalarBackend>(coords.data(), L, weights, config,
                                             embeddings.data(), &workspace);

            // Save embeddings
            try {
                save_npy_2d(output_path, embeddings.data(), L, config.hidden_dim);
            } catch (const std::exception& save_error) {
                throw std::runtime_error("Failed to save embeddings: " + std::string(save_error.what()));
            }

            std::cout << "  ✓ Saved to " << output_path << " (L=" << L << ")\n";
            success_count++;

        } catch (const std::exception& e) {
            std::cerr << "  ✗ Failed: " << e.what() << "\n";
            failure_count++;
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n===========================================\n";
    std::cout << "  Encode Batch Complete\n";
    std::cout << "===========================================\n";
    std::cout << "Success: " << success_count << "\n";
    std::cout << "Failure: " << failure_count << "\n";

    return (failure_count > 0) ? 1 : 0;
}

}  // namespace commands
}  // namespace pfalign
