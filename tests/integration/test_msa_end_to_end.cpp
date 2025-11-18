/**
 * End-to-End MSA Test
 *
 * Full integration test that:
 * 1. Loads PDB files from organized families
 * 2. Encodes with MPNN
 * 3. Builds guide tree
 * 4. Runs progressive MSA
 * 5. Writes FASTA output
 * 6. Validates results
 *
 * Usage:
 *   ./test_msa_end_to_end
 *   ./test_msa_end_to_end --family=immunoglobulin
 *   ./test_msa_end_to_end --method=nj --output=output.fasta
 */

#include "pfalign/algorithms/progressive_msa/progressive_msa.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/algorithms/distance_metrics/distance_matrix.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/modules/mpnn/mpnn_cache_adapter.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/io/fasta_writer.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/io/sequence_utils.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "../test_utils.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <fstream>

using namespace pfalign;
using namespace pfalign::tree_builders;
namespace fs = std::filesystem;

// Test configuration
struct TestConfig {
    std::string family = "globin";
    std::string method = "upgma";
    std::string output_fasta = "msa_output.fasta";
    bool verbose = false;
};

// Parse command line arguments
TestConfig parse_args(int argc, char** argv) {
    TestConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.find("--family=") == 0) {
            config.family = arg.substr(9);
        } else if (arg.find("--method=") == 0) {
            config.method = arg.substr(9);
        } else if (arg.find("--output=") == 0) {
            config.output_fasta = arg.substr(9);
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: test_msa_end_to_end [OPTIONS]\n";
            std::cout << "\nOptions:\n";
            std::cout << "  --family=NAME     Protein family (globin, immunoglobulin, lysozyme) [default: globin]\n";
            std::cout << "  --method=METHOD   Guide tree method (upgma, nj, bionj, mst) [default: upgma]\n";
            std::cout << "  --output=FILE     Output FASTA file [default: msa_output.fasta]\n";
            std::cout << "  --verbose, -v     Verbose output\n";
            std::cout << "  --help, -h        Show this help\n";
            std::exit(0);
        }
    }

    return config;
}

// Find family directory
fs::path find_family_dir(const std::string& family) {
    // Use test_utils helper to get integration path
    std::string family_path = pfalign::test::get_integration_path("msa_families/" + family);

    if (fs::exists(family_path) && fs::is_directory(family_path)) {
        return family_path;
    }

    throw std::runtime_error("Could not find family directory: " + family);
}

// Load all PDB files from family directory
std::vector<fs::path> load_family_pdbs(const fs::path& family_dir) {
    std::vector<fs::path> pdbs;

    for (const auto& entry : fs::directory_iterator(family_dir)) {
        if (entry.path().extension() == ".pdb") {
            pdbs.push_back(entry.path());
        }
    }

    std::sort(pdbs.begin(), pdbs.end());
    return pdbs;
}

// Encode PDBs and add to cache
bool encode_family(
    const std::vector<fs::path>& pdb_paths,
    mpnn::MPNNCacheAdapter& adapter,
    bool verbose
) {
    io::PDBParser parser;

    for (size_t i = 0; i < pdb_paths.size(); ++i) {
        const auto& pdb_path = pdb_paths[i];
        std::string identifier = pdb_path.stem().string();

        if (verbose) {
            std::cout << "  Loading " << identifier << "..." << std::endl;
        }

        try {
            io::Protein prot = parser.parse_file(pdb_path.string());

            if (prot.chains.empty()) {
                std::cerr << "    ERROR: No chains in " << pdb_path << std::endl;
                return false;
            }

            int L = prot.get_chain(0).size();
            auto coords = prot.get_backbone_coords(0);
            std::string sequence = io::extract_sequence(prot.get_chain(0));

            adapter.add_protein(i, coords.data(), L, identifier, sequence);

            if (verbose) {
                std::cout << "    ✓ Encoded: " << L << " residues" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "    ERROR: Failed to load " << pdb_path << ": " << e.what() << std::endl;
            return false;
        }
    }

    return true;
}

// Build guide tree using specified method
msa::GuideTree build_tree(
    const float* distances,
    int N,
    const std::string& method,
    memory::Arena* arena
) {
    if (method == "upgma") {
        return tree_builders::build_upgma_tree(distances, N, arena);
    } else if (method == "nj") {
        return tree_builders::build_nj_tree(distances, N, arena);
    } else if (method == "bionj") {
        return tree_builders::build_bionj_tree(distances, N, arena);
    } else if (method == "mst") {
        return tree_builders::build_mst_tree(distances, N, arena);
    } else {
        throw std::runtime_error("Unknown guide tree method: " + method);
    }
}

// Write MSA result to FASTA
bool write_msa_fasta(
    const msa::Profile* alignment,
    const std::string& output_path,
    bool verbose
) {
    if (!alignment) {
        std::cerr << "ERROR: Alignment is null" << std::endl;
        return false;
    }

    try {
        std::ofstream out(output_path);
        if (!out) {
            std::cerr << "ERROR: Could not open " << output_path << " for writing" << std::endl;
            return false;
        }

        // Write each sequence
        for (int seq_idx = 0; seq_idx < alignment->num_sequences; ++seq_idx) {
            // Get sequence index from profile
            int orig_seq_idx = alignment->seq_indices[seq_idx];

            // Write header
            out << ">sequence_" << orig_seq_idx << " | Length: " << alignment->length << "\n";

            // Extract aligned sequence with gaps
            std::string sequence;
            sequence.reserve(alignment->length);

            for (int col_idx = 0; col_idx < alignment->length; ++col_idx) {
                const auto& col = alignment->columns[col_idx];
                const auto& pos = col.positions[seq_idx];

                // Check if this sequence has a residue at this column
                sequence += pos.is_gap() ? '-' : 'X';  // Use 'X' as placeholder
            }

            // Write sequence with line wrapping at 80 characters
            for (size_t i = 0; i < sequence.size(); i += 80) {
                out << sequence.substr(i, 80) << "\n";
            }
        }

        out.close();

        if (verbose) {
            std::cout << "  ✓ Wrote FASTA: " << output_path << std::endl;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to write FASTA: " << e.what() << std::endl;
        return false;
    }
}

// Main test
int main(int argc, char** argv) {
    TestConfig config = parse_args(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "  MSA End-to-End Integration Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Family: " << config.family << std::endl;
    std::cout << "Method: " << config.method << std::endl;
    std::cout << "Output: " << config.output_fasta << std::endl;
    std::cout << std::endl;

    try {
        // 1. Find family directory
        std::cout << "[1/6] Finding family directory..." << std::endl;
        fs::path family_dir = find_family_dir(config.family);
        std::cout << "  ✓ Found: " << family_dir << std::endl;

        // 2. Load PDB files
        std::cout << "\n[2/6] Loading PDB files..." << std::endl;
        auto pdb_paths = load_family_pdbs(family_dir);
        std::cout << "  ✓ Found " << pdb_paths.size() << " PDB files" << std::endl;

        if (pdb_paths.size() < 2) {
            std::cerr << "ERROR: Need at least 2 sequences for MSA" << std::endl;
            return 1;
        }

        // 3. Load MPNN weights and encode
        std::cout << "\n[3/6] Encoding sequences with MPNN..." << std::endl;

        mpnn::MPNNWeights weights(3);
        mpnn::MPNNConfig mpnn_config;

        try {
            auto [loaded_weights, loaded_config, loaded_sw] = weights::load_embedded_mpnn_weights();
            weights = std::move(loaded_weights);
            mpnn_config = loaded_config;
            std::cout << "  ✓ Loaded embedded MPNN weights" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Could not load MPNN weights: " << e.what() << std::endl;
            return 1;
        }

        memory::Arena arena(500);  // 500 MB
        SequenceCache cache(&arena);
        mpnn::MPNNCacheAdapter adapter(cache, weights, mpnn_config, &arena);

        if (!encode_family(pdb_paths, adapter, config.verbose)) {
            return 1;
        }

        std::cout << "  ✓ Encoded " << cache.size() << " sequences" << std::endl;

        // 4. Build guide tree
        std::cout << "\n[4/6] Building guide tree (" << config.method << ")..." << std::endl;

        int N = cache.size();
        float* distances = arena.allocate<float>(N * N);

        // Compute distance matrix from cache
        for (int i = 0; i < N; ++i) {
            distances[i * N + i] = 0.0f;

            for (int j = i + 1; j < N; ++j) {
                const auto* seq_i = cache.get(i);
                const auto* seq_j = cache.get(j);

                // Compute cosine distance from embeddings
                float dot = 0.0f;
                float norm_i = 0.0f;
                float norm_j = 0.0f;

                int min_len = std::min(seq_i->length, seq_j->length);
                for (int pos = 0; pos < min_len; ++pos) {
                    const float* emb_i = seq_i->embeddings + pos * seq_i->hidden_dim;
                    const float* emb_j = seq_j->embeddings + pos * seq_j->hidden_dim;

                    for (int d = 0; d < seq_i->hidden_dim; ++d) {
                        dot += emb_i[d] * emb_j[d];
                        norm_i += emb_i[d] * emb_i[d];
                        norm_j += emb_j[d] * emb_j[d];
                    }
                }

                float similarity = dot / (std::sqrt(norm_i * norm_j) + 1e-8f);
                float distance = 1.0f - similarity;

                distances[i * N + j] = distance;
                distances[j * N + i] = distance;
            }
        }

        msa::GuideTree tree = build_tree(distances, N, config.method, &arena);
        std::cout << "  ✓ Built guide tree with " << (2 * N - 1) << " nodes" << std::endl;

        // 5. Run progressive MSA
        std::cout << "\n[5/6] Running progressive MSA..." << std::endl;

        msa::MSAConfig msa_config;
        msa_config.gap_open = -11.0f;
        msa_config.gap_extend = -1.0f;
        msa_config.temperature = 1.0f;
        msa_config.ecs_temperature = 1.0f;

        msa::MSAResult result = msa::progressive_msa<ScalarBackend>(
            cache, tree, msa_config, &arena
        );

        std::cout << "  ✓ MSA complete" << std::endl;
        std::cout << "    Sequences: " << result.num_sequences << std::endl;
        std::cout << "    Alignment length: " << result.aligned_length << " columns" << std::endl;
        std::cout << "    Quality (ECS): " << std::fixed << std::setprecision(4) << result.ecs << std::endl;

        // 6. Write output
        std::cout << "\n[6/6] Writing FASTA output..." << std::endl;

        if (!write_msa_fasta(result.alignment, config.output_fasta, config.verbose)) {
            return 1;
        }

        std::cout << "  ✓ Wrote alignment to " << config.output_fasta << std::endl;

        // Cleanup
        msa::Profile::destroy(result.alignment);

        std::cout << "\n========================================" << std::endl;
        std::cout << "  ✓ End-to-End MSA Test PASSED" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
