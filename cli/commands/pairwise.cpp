#include "commands/commands.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands/input_utils.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/adapters/alignment_types.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/tools/weights/save_npy.h"

namespace pfalign {
namespace commands {

namespace {

std::string Basename(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}

std::vector<float> ExtractBackbone(const io::Protein& protein, int chain_index) {
    return protein.get_backbone_coords(static_cast<size_t>(chain_index));
}

std::string ExtractSequence(const io::Protein& protein, int chain_index) {
    return protein.get_sequence(static_cast<size_t>(chain_index));
}

struct PreparedInputs {
    std::vector<float> coords1;
    std::vector<float> coords2;
    std::vector<float> emb1;
    std::vector<float> emb2;
    int L1 = 0;
    int L2 = 0;
    int hidden_dim = 0;
    int hidden_dim1 = 0;
    int hidden_dim2 = 0;
    std::string seq1;
    std::string seq2;
    bool has_seq1 = false;
    bool has_seq2 = false;
    InputType type1 = InputType::kStructure;
    InputType type2 = InputType::kStructure;
};

struct AlignmentStats {
    int matches = 0;
    int gaps_in_first = 0;
    int gaps_in_second = 0;
};

AlignmentStats ComputeAlignmentStats(const pairwise::AlignmentResult& result) {
    AlignmentStats stats;
    for (int idx = 0; idx < result.path_length; ++idx) {
        const AlignmentPair& pair = result.alignment_path[idx];
        if (pair.i >= 0 && pair.j >= 0) {
            stats.matches++;
        } else if (pair.i >= 0) {
            stats.gaps_in_second++;
        } else if (pair.j >= 0) {
            stats.gaps_in_first++;
        }
    }
    return stats;
}

void WritePairwiseMetrics(const std::string& metrics_path, const std::string& input1,
                          const std::string& input2, const pairwise::AlignmentResult& result,
                          const PreparedInputs& prep, const std::string& poster_path,
                          const std::string& fasta_path, const std::string& superpose_path,
                          float gap_open, float gap_extend) {
    if (metrics_path.empty()) {
        return;
    }

    std::ofstream out(metrics_path);
    if (!out) {
        throw errors::FileWriteError(metrics_path);
    }

    AlignmentStats stats = ComputeAlignmentStats(result);
    const int max_len = std::max(result.L1, result.L2);
    const double coverage =
        max_len > 0 ? static_cast<double>(stats.matches) / static_cast<double>(max_len) : 0.0;

    out << "# PFalign Pairwise Metrics\n";
    out << "input1: " << input1 << "\n";
    out << "input2: " << input2 << "\n";
    out << "score: " << result.score << "\n";
    out << "partition: " << result.partition << "\n";
    out << "L1: " << result.L1 << "\n";
    out << "L2: " << result.L2 << "\n";
    out << "path_length: " << result.path_length << "\n";
    out << "matches: " << stats.matches << "\n";
    out << "gaps_in_first: " << stats.gaps_in_first << "\n";
    out << "gaps_in_second: " << stats.gaps_in_second << "\n";
    out << "coverage: " << coverage << "\n";
    out << "gap_open: " << std::fixed << std::setprecision(4) << gap_open << "\n";
    out << "gap_extend: " << std::fixed << std::setprecision(4) << gap_extend << "\n";
    out << "fastas_written: " << (fasta_path.empty() ? "no" : fasta_path) << "\n";
    out << "posterior_written: " << (poster_path.empty() ? "no" : poster_path) << "\n";
    out << "superpose_written: " << (superpose_path.empty() ? "no" : superpose_path) << "\n";

    if (prep.type1 == InputType::kStructure && prep.type2 == InputType::kStructure) {
        out << "structure_metrics_note: RMSD/TM available via pfalign Python API\n";
    } else {
        out << "structure_metrics_note: N/A (embedding inputs)\n";
    }
}

PreparedInputs LoadInputs(const std::string& input1, const std::string& input2, int chain1,
                          int chain2) {
    PreparedInputs prep;
    prep.type1 = DetectInputType(input1);
    prep.type2 = DetectInputType(input2);

    if (prep.type1 == InputType::kStructure) {
        io::Protein protein = LoadStructureFile(input1);
        validation::validate_chain_index(static_cast<int>(protein.num_chains()), chain1, input1);
        prep.coords1 = ExtractBackbone(protein, chain1);
        prep.seq1 = ExtractSequence(protein, chain1);
        prep.has_seq1 = true;
        prep.L1 = static_cast<int>(protein.get_chain(chain1).size());
    } else {
        EmbeddingArray emb = LoadEmbeddingFile(input1);
        prep.emb1 = std::move(emb.values);
        prep.L1 = emb.rows;
        prep.hidden_dim1 = emb.cols;
        prep.hidden_dim = emb.cols;
    }

    if (prep.type2 == InputType::kStructure) {
        io::Protein protein = LoadStructureFile(input2);
        validation::validate_chain_index(static_cast<int>(protein.num_chains()), chain2, input2);
        prep.coords2 = ExtractBackbone(protein, chain2);
        prep.seq2 = ExtractSequence(protein, chain2);
        prep.has_seq2 = true;
        prep.L2 = static_cast<int>(protein.get_chain(chain2).size());
    } else {
        EmbeddingArray emb = LoadEmbeddingFile(input2);
        prep.emb2 = std::move(emb.values);
        prep.L2 = emb.rows;
        prep.hidden_dim2 = emb.cols;
        prep.hidden_dim = (prep.hidden_dim == 0) ? emb.cols : prep.hidden_dim;
    }

    if (prep.hidden_dim1 > 0 && prep.hidden_dim2 > 0 && prep.hidden_dim1 != prep.hidden_dim2) {
        throw errors::messages::embedding_dimension_mismatch(
            prep.hidden_dim1, prep.hidden_dim2, input1, input2);
    }

    if (prep.L1 <= 0 || prep.L2 <= 0) {
        throw errors::ValidationError(
            "Inputs must contain at least one residue/row",
            "Provide valid structure or embedding files with non-zero length");
    }

    return prep;
}

void EncodeStructureToEmbeddings(const std::vector<float>& coords, int length,
                                 const mpnn::MPNNWeights& weights, const mpnn::MPNNConfig& config,
                                 std::vector<float>* out_embeddings) {
    if (out_embeddings->size() != static_cast<size_t>(length * config.hidden_dim)) {
        out_embeddings->assign(static_cast<size_t>(length) * config.hidden_dim, 0.0f);
    }

    mpnn::MPNNWorkspace workspace(length, config.k_neighbors, config.hidden_dim, config.num_rbf);
    mpnn::mpnn_forward<ScalarBackend>(coords.data(), length, weights, config,
                                      out_embeddings->data(), &workspace);
}

pairwise::PairwiseConfig BuildPairwiseConfig(const mpnn::MPNNConfig& mpnn_template, float gap_open,
                                             float gap_extend, float temperature, int k_neighbors,
                                             int hidden_dim_override, bool parallel_mpnn) {
    pairwise::PairwiseConfig config;
    config.mpnn_config = mpnn_template;
    if (hidden_dim_override > 0) {
        config.mpnn_config.hidden_dim = hidden_dim_override;
    }
    config.mpnn_config.k_neighbors = k_neighbors;
    config.sw_mode = pairwise::PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;
    config.sw_config.affine = true;
    config.sw_config.temperature = temperature;
    config.sw_config.gap_open = gap_open;
    config.sw_config.gap_extend = gap_extend;
    config.sw_config.gap = gap_extend;
    config.parallel_mpnn = parallel_mpnn;
    return config;
}

}  // namespace

int pairwise(const std::string& input1, const std::string& input2,
             const std::string& posterior_path, const std::string& fasta_path,
             const std::string& superpose_path, const std::string& metrics_path, float gap_open,
             float gap_extend, float temperature, int k_neighbors, int chain1, int chain2,
             bool disable_parallel_mpnn) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign Pairwise Command\n";
    std::cout << "===========================================\n\n";
    std::cout << "Input 1:      " << input1 << "\n";
    std::cout << "Input 2:      " << input2 << "\n";
    if (!posterior_path.empty()) {
        std::cout << "Posterior:    " << posterior_path << "\n";
    }
    if (!fasta_path.empty()) {
        std::cout << "FASTA:        " << fasta_path << "\n";
    }
    if (!superpose_path.empty()) {
        std::cout << "Superpose:    " << superpose_path << "\n";
    }
    if (!metrics_path.empty()) {
        std::cout << "Metrics:      " << metrics_path << "\n";
    }
    if (posterior_path.empty() && fasta_path.empty() && superpose_path.empty() &&
        metrics_path.empty()) {
        throw errors::ValidationError(
            "No output path provided",
            "Provide at least one of: --output (FASTA), --posterior (.npy), --superpose (.pdb), or --metrics");
    }
    std::cout << "gap_open:     " << gap_open << "\n";
    std::cout << "gap_extend:   " << gap_extend << "\n";
    std::cout << "temperature:  " << temperature << "\n";
    std::cout << "k_neighbors:  " << k_neighbors << "\n";
    std::cout << "chain1:       " << chain1 << "\n";
    std::cout << "chain2:       " << chain2 << "\n\n";

    try {
        PreparedInputs prep = LoadInputs(input1, input2, chain1, chain2);

        bool needs_weights =
            (prep.type1 == InputType::kStructure || prep.type2 == InputType::kStructure);

        mpnn::MPNNWeights weights(0);
        mpnn::MPNNConfig mpnn_config;

        if (needs_weights) {
            auto loaded = weights::load_embedded_mpnn_weights();
            weights = std::move(std::get<0>(loaded));
            mpnn_config = std::get<1>(loaded);
            int expected_dim = mpnn_config.hidden_dim;
            if ((prep.hidden_dim1 > 0 && prep.hidden_dim1 != expected_dim) ||
                (prep.hidden_dim2 > 0 && prep.hidden_dim2 != expected_dim)) {
                int actual_dim = prep.hidden_dim1 > 0 ? prep.hidden_dim1 : prep.hidden_dim2;
                throw errors::messages::embedding_dimension_mismatch(
                    actual_dim, expected_dim, "input embedding", "MPNN weights");
            }
        } else {
            mpnn_config.hidden_dim = prep.hidden_dim;
            mpnn_config.num_layers = 0;
            mpnn_config.k_neighbors = k_neighbors;
        }

        if (!needs_weights && prep.hidden_dim <= 0) {
            throw errors::ValidationError(
                "Failed to determine embedding dimension",
                "Check that embedding files are valid and non-empty");
        }

        int hidden_dim = needs_weights ? mpnn_config.hidden_dim : prep.hidden_dim;
        pairwise::PairwiseConfig config =
            BuildPairwiseConfig(mpnn_config, gap_open, gap_extend, temperature, k_neighbors,
                                hidden_dim, !disable_parallel_mpnn);
        const double matrix_size =
            std::max(1.0, static_cast<double>(prep.L1) * static_cast<double>(prep.L2));
        const float decode_gap_penalty =
            std::min(gap_open, static_cast<float>(std::log(1.0 / matrix_size)));

        std::vector<float> encoded1;
        std::vector<float> encoded2;

        if (prep.type1 == InputType::kStructure && prep.type2 != InputType::kStructure) {
            EncodeStructureToEmbeddings(prep.coords1, prep.L1, weights, config.mpnn_config,
                                        &encoded1);
        }
        if (prep.type2 == InputType::kStructure && prep.type1 != InputType::kStructure) {
            EncodeStructureToEmbeddings(prep.coords2, prep.L2, weights, config.mpnn_config,
                                        &encoded2);
        }

        pairwise::PairwiseWorkspace workspace(prep.L1, prep.L2, config);

        size_t arena_bytes = mpnn::compute_arena_size(
            std::max(prep.L1, prep.L2), config.mpnn_config.k_neighbors,
            config.mpnn_config.num_layers > 0 ? config.mpnn_config.num_layers : 3);
        size_t arena_mb = std::max<size_t>(1, arena_bytes / (1024 * 1024));
        pfalign::memory::GrowableArena arena(arena_mb);

        pairwise::AlignmentResult result;
        std::vector<float> posteriors(static_cast<size_t>(prep.L1) * prep.L2);
        std::vector<AlignmentPair> alignment(static_cast<size_t>(prep.L1 + prep.L2));
        result.posteriors = posteriors.data();
        result.alignment_path = alignment.data();
        result.max_path_length = static_cast<int>(alignment.size());
        result.id1 = Basename(input1);
        result.id2 = Basename(input2);

        if (prep.type1 == InputType::kStructure && prep.type2 == InputType::kStructure) {
            pairwise::pairwise_align_full<ScalarBackend>(
                prep.coords1.data(), prep.L1, prep.coords2.data(), prep.L2, config, weights,
                &workspace, &result, &arena, decode_gap_penalty);
        } else {
            const float* emb_ptr1 =
                (prep.type1 == InputType::kStructure) ? encoded1.data() : prep.emb1.data();
            const float* emb_ptr2 =
                (prep.type2 == InputType::kStructure) ? encoded2.data() : prep.emb2.data();

            if (prep.type1 != InputType::kStructure && prep.type2 != InputType::kStructure) {
                size_t expected1 = static_cast<size_t>(prep.L1 * hidden_dim);
                size_t expected2 = static_cast<size_t>(prep.L2 * hidden_dim);
                if (prep.emb1.size() != expected1 || prep.emb2.size() != expected2) {
                    throw errors::DimensionError(
                        "embedding arrays",
                        "(" + std::to_string(prep.emb1.size()) + ", " + std::to_string(prep.emb2.size()) + ")",
                        "(" + std::to_string(expected1) + ", " + std::to_string(expected2) + ")");
                }
            }

            pairwise::pairwise_align_from_embeddings_full<ScalarBackend>(
                emb_ptr1, prep.L1, emb_ptr2, prep.L2, hidden_dim, config, &workspace, &result,
                &arena, decode_gap_penalty);
        }

        if (result.path_length == 0) {
            throw errors::messages::alignment_failed("empty alignment path");
        }

        if (!posterior_path.empty()) {
            save_npy_2d(posterior_path, result.posteriors, prep.L1, prep.L2);
        }

        if (!fasta_path.empty()) {
            if (!prep.has_seq1 || !prep.has_seq2) {
                throw errors::ValidationError(
                    "FASTA output requires structure inputs",
                    "Both inputs must be structure files to generate FASTA alignment");
            }
            if (!result.write_fasta(fasta_path, prep.seq1, prep.seq2)) {
                throw errors::FileWriteError(fasta_path, "Failed to write alignment");
            }
        }

        if (!superpose_path.empty()) {
            if (prep.type1 != InputType::kStructure || prep.type2 != InputType::kStructure) {
                throw errors::ValidationError(
                    "Superposed PDB output requires structure inputs",
                    "Both inputs must be structure files to generate superposed PDB");
            }
            if (!result.write_superposed_pdb(superpose_path)) {
                throw errors::FileWriteError(superpose_path, "Failed to write superposed structure");
            }
        }

        if (!metrics_path.empty()) {
            WritePairwiseMetrics(metrics_path, input1, input2, result, prep, posterior_path,
                                 fasta_path, superpose_path, gap_open, gap_extend);
        }

        std::cout << "✓ Pairwise alignment complete\n";
        std::cout << "  Score:      " << result.score << "\n";
        std::cout << "  Partition:  " << result.partition << "\n";
        std::cout << "  Path len:   " << result.path_length << "\n";
        std::cout << "  Posterior:  (" << prep.L1 << ", " << prep.L2 << ")\n";
        if (!metrics_path.empty()) {
            std::cout << "  Metrics:    " << metrics_path << "\n";
        }
        return 0;
    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "✗ Pairwise command failed: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace commands
}  // namespace pfalign
