/**
 * PyBind11 entry point for PFalign v2.
 *
 * Exposes high-level functions (pairwise, msa, encode, similarity)
 * and NumPy-style result classes.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include "numpy_utils.h"  // Local to _bindings directory
#include "pfalign/common/growable_arena.h"
#include "pfalign/common/progress_bar.h"
#include "pfalign/common/result_types.h"
#include "pfalign/common/thread_pool.h"
#include "commands/input_utils.h"  // From CLI directory
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/modules/mpnn/mpnn_cache_adapter.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/modules/msa/profile.h"
#include "pfalign/algorithms/distance_metrics/distance_matrix.h"
#include "pfalign/algorithms/progressive_msa/progressive_msa.h"
#include "pfalign/algorithms/tree_builders/builders.h"
#include "pfalign/types/guide_tree_types.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/primitives/alignment_decode/alignment_decode.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"
#include "pfalign/primitives/structural_metrics/lddt_impl.h"
#include "pfalign/primitives/structural_metrics/dali_impl.h"
#include "pfalign/primitives/structural_metrics/distance_matrix.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/io/alignment_formats.h"
#include "pfalign/io/sequence_utils.h"
#include "pfalign/errors/pfalign_error.h"
#include "pfalign/errors/error_categories.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"

namespace py = pybind11;
namespace fs = std::filesystem;

using pfalign::PairwiseResult;
using pfalign::EmbeddingResult;
using pfalign::SimilarityResult;
using pfalign::MSAResult;
using pfalign::AlignmentPair;
using pfalign::pairwise::AlignmentResult;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;

namespace {

constexpr size_t kDefaultArenaMb = 200;

auto& EmbeddedWeightsTuple() {
    static auto embedded = pfalign::weights::load_embedded_mpnn_weights();
    return embedded;
}

const pfalign::mpnn::MPNNWeights& EmbeddedWeights() {
    return std::get<0>(EmbeddedWeightsTuple());
}

const pfalign::mpnn::MPNNConfig& EmbeddedConfig() {
    return std::get<1>(EmbeddedWeightsTuple());
}

const pfalign::weights::SWParams& EmbeddedSwParams() {
    return std::get<2>(EmbeddedWeightsTuple());
}

pfalign::mpnn::MPNNConfig BuildMpnnConfig(int k_neighbors) {
    pfalign::mpnn::MPNNConfig cfg = EmbeddedConfig();
    cfg.k_neighbors = k_neighbors;
    return cfg;
}

int ClampThreadCount(int requested_threads) {
    if (requested_threads <= 0) {
        return requested_threads;
    }
    int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hardware_threads <= 0) {
        hardware_threads = 1;
    }
    return std::min(requested_threads, hardware_threads);
}

struct StructureRecord {
    std::vector<float> coords;
    std::string sequence;
    int length = 0;
    std::string identifier;
};

// Holder for GuideTree that owns the arena memory
// (GuideTree contains pointers to arena-allocated nodes)
struct GuideTreeHolder {
    std::unique_ptr<pfalign::memory::GrowableArena> arena;
    pfalign::types::GuideTree tree;

    GuideTreeHolder(std::unique_ptr<pfalign::memory::GrowableArena> a, pfalign::types::GuideTree t)
        : arena(std::move(a)), tree(std::move(t)) {}

    // Delegate GuideTree methods
    int num_sequences() const { return tree.num_sequences(); }
    int num_nodes() const { return tree.num_nodes(); }
    int root_index() const { return tree.root_index(); }
    const pfalign::types::GuideTree& get_tree() const { return tree; }
};

// Get chain information from a structure file
std::vector<std::string> GetChainIds(const std::string& path) {
    pfalign::io::Protein protein = pfalign::commands::LoadStructureFile(path);
    std::vector<std::string> chain_ids;
    chain_ids.reserve(protein.num_chains());

    for (const auto& chain : protein.chains) {
        chain_ids.push_back(std::string(1, chain.chain_id));
    }

    return chain_ids;
}

// Resolve chain name or index to chain index
int ResolveChainIndex(const std::string& path, const std::string& chain_spec) {
    // If it's a single digit, treat as index
    if (chain_spec.size() == 1 && std::isdigit(chain_spec[0])) {
        return chain_spec[0] - '0';
    }

    // Otherwise treat as chain ID
    pfalign::io::Protein protein = pfalign::commands::LoadStructureFile(path);
    if (chain_spec.size() != 1) {
        throw pfalign::errors::ValidationError("chain", chain_spec, "single character chain ID");
    }

    int idx = protein.find_chain_index(chain_spec[0]);
    if (idx < 0) {
        std::vector<std::string> available;
        for (size_t i = 0; i < protein.num_chains(); ++i) {
            available.push_back(std::string(1, protein.chains[i].chain_id));
        }
        throw pfalign::errors::messages::chain_not_found(chain_spec, path, available);
    }

    return idx;
}

StructureRecord LoadStructureRecord(const std::string& path, int chain_index) {
    pfalign::io::Protein protein = pfalign::commands::LoadStructureFile(path);
    if (protein.num_chains() == 0) {
        throw pfalign::errors::messages::no_chains_in_structure(path);
    }
    if (chain_index < 0 || static_cast<size_t>(chain_index) >= protein.num_chains()) {
        throw pfalign::errors::messages::chain_index_out_of_range(
            chain_index,
            static_cast<int>(protein.num_chains()),
            path
        );
    }

    StructureRecord record;
    record.length = static_cast<int>(protein.get_chain(chain_index).size());
    record.sequence = protein.get_sequence(chain_index);
    record.coords = protein.get_backbone_coords(chain_index);
    record.identifier = fs::path(path).stem().string();
    if (record.sequence.empty()) {
        record.sequence.assign(record.length, 'X');
    }
    return record;
}

PairwiseConfig BuildPairwiseConfig(int k_neighbors,
                                   int hidden_dim,
                                   std::optional<float> gap_open,
                                   std::optional<float> gap_extend,
                                   std::optional<float> temperature,
                                   std::optional<bool> parallel_mpnn) {
    // Use SW parameters from embedded weights (trained model values) as defaults
    const auto& sw_params = EmbeddedSwParams();

    PairwiseConfig config;
    config.mpnn_config = BuildMpnnConfig(k_neighbors);
    config.mpnn_config.hidden_dim = hidden_dim > 0 ? hidden_dim : config.mpnn_config.hidden_dim;
    config.sw_mode = PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE;

    // Use embedded SW parameters from trained model, allow override
    config.sw_config.gap_open = gap_open.value_or(sw_params.gap_open);
    config.sw_config.gap_extend = gap_extend.value_or(sw_params.gap);
    config.sw_config.gap = gap_extend.value_or(sw_params.gap);
    config.sw_config.temperature = temperature.value_or(sw_params.temperature);
    config.sw_config.affine = true;
    config.parallel_mpnn = parallel_mpnn.value_or(true);
    return config;
}

PairwiseResult MakePairwiseResult(const AlignmentResult& result) {
    const size_t size = static_cast<size_t>(result.L1) * result.L2;
    float* posteriors = new float[size];
    std::copy(result.posteriors, result.posteriors + size, posteriors);

    std::vector<std::pair<int, int>> alignment;
    alignment.reserve(result.path_length);
    for (int i = 0; i < result.path_length; ++i) {
        alignment.emplace_back(result.alignment_path[i].i,
                               result.alignment_path[i].j);
    }

    return PairwiseResult(posteriors,
                          result.L1,
                          result.L2,
                          result.score,
                          result.partition,
                          std::move(alignment));
}

PairwiseResult PairwiseFromStructures(const std::string& input1,
                                      const std::string& input2,
                                      int chain1,
                                      int chain2,
                                      int k_neighbors,
                                      std::optional<float> gap_open,
                                      std::optional<float> gap_extend,
                                      std::optional<float> temperature,
                                      std::optional<bool> parallel_mpnn) {
    StructureRecord s1 = LoadStructureRecord(input1, chain1);
    StructureRecord s2 = LoadStructureRecord(input2, chain2);

    const int L1 = s1.length;
    const int L2 = s2.length;
    PairwiseConfig config = BuildPairwiseConfig(k_neighbors, EmbeddedConfig().hidden_dim,
                                                 gap_open, gap_extend, temperature, parallel_mpnn);

    // Use hard gap penalty for Viterbi decoding (default: -5.0)
    // This is separate from the soft gap parameters used in Smith-Waterman
    const float decode_gap_penalty = -5.0f;

    PairwiseWorkspace workspace(L1, L2, config);
    size_t arena_bytes = pfalign::mpnn::compute_arena_size(std::max(L1, L2),
                                          config.mpnn_config.k_neighbors,
                                          config.mpnn_config.num_layers);
    size_t arena_mb = std::max<size_t>(1, arena_bytes / (1024 * 1024));
    pfalign::memory::GrowableArena arena(arena_mb);

    std::vector<float> posterior_buffer(static_cast<size_t>(L1) * L2);
    std::vector<AlignmentPair> path_buffer(static_cast<size_t>(L1 + L2));

    AlignmentResult result;
    result.L1 = L1;
    result.L2 = L2;
    result.posteriors = posterior_buffer.data();
    result.alignment_path = path_buffer.data();
    result.max_path_length = static_cast<int>(path_buffer.size());
    result.coords1 = s1.coords.data();
    result.coords2 = s2.coords.data();
    result.id1 = s1.identifier;
    result.id2 = s2.identifier;

    pfalign::pairwise::pairwise_align_full<pfalign::ScalarBackend>(
        s1.coords.data(),
        L1,
        s2.coords.data(),
        L2,
        config,
        EmbeddedWeights(),
        &workspace,
        &result,
        &arena,
        decode_gap_penalty);

    return MakePairwiseResult(result);
}

PairwiseResult PairwiseFromEmbeddings(
    py::array_t<float, py::array::c_style | py::array::forcecast> emb1,
    py::array_t<float, py::array::c_style | py::array::forcecast> emb2,
    std::optional<float> gap_open,
    std::optional<float> gap_extend,
    std::optional<float> temperature,
    std::optional<bool> parallel_mpnn) {
    pfalign::bindings::validate_2d_array(emb1, "embeddings1");
    pfalign::bindings::validate_2d_array(emb2, "embeddings2");

    auto shape1 = emb1.shape();
    auto shape2 = emb2.shape();
    if (shape1[1] != shape2[1]) {
        throw pfalign::errors::messages::embedding_dimension_mismatch(
            static_cast<int>(shape1[1]),
            static_cast<int>(shape2[1]),
            "embeddings1",
            "embeddings2"
        );
    }

    const int L1 = static_cast<int>(shape1[0]);
    const int L2 = static_cast<int>(shape2[0]);
    const int hidden_dim = static_cast<int>(shape1[1]);

    PairwiseConfig config = BuildPairwiseConfig(EmbeddedConfig().k_neighbors, hidden_dim,
                                                 gap_open, gap_extend, temperature, parallel_mpnn);

    PairwiseWorkspace workspace(L1, L2, config);
    size_t arena_bytes = pfalign::mpnn::compute_arena_size(std::max(L1, L2),
                                          config.mpnn_config.k_neighbors,
                                          config.mpnn_config.num_layers);
    size_t arena_mb = std::max<size_t>(1, arena_bytes / (1024 * 1024));
    pfalign::memory::GrowableArena arena(arena_mb);

    std::vector<float> posterior_buffer(static_cast<size_t>(L1) * L2);
    std::vector<AlignmentPair> path_buffer(static_cast<size_t>(L1 + L2));

    AlignmentResult result;
    result.L1 = L1;
    result.L2 = L2;
    result.posteriors = posterior_buffer.data();
    result.alignment_path = path_buffer.data();
    result.max_path_length = static_cast<int>(path_buffer.size());

    // Use hard gap penalty for Viterbi decoding (default: -5.0)
    // This is separate from the soft gap parameters used in Smith-Waterman
    const float decode_gap_penalty = -5.0f;
    pfalign::pairwise::pairwise_align_from_embeddings_full<pfalign::ScalarBackend>(
        emb1.data(),
        L1,
        emb2.data(),
        L2,
        hidden_dim,
        config,
        &workspace,
        &result,
        &arena,
        decode_gap_penalty);

    return MakePairwiseResult(result);
}

EmbeddingResult EncodeStructure(const std::string& path,
                                int chain,
                                int k_neighbors) {
    StructureRecord record = LoadStructureRecord(path, chain);
    pfalign::mpnn::MPNNConfig config = BuildMpnnConfig(k_neighbors);

    pfalign::mpnn::MPNNWorkspace workspace(record.length,
                                           config.k_neighbors,
                                           config.hidden_dim,
                                           config.num_rbf);
    std::vector<float> embeddings(static_cast<size_t>(record.length) *
                                  config.hidden_dim);

    pfalign::mpnn::mpnn_forward<pfalign::ScalarBackend>(
        record.coords.data(),
        record.length,
        EmbeddedWeights(),
        config,
        embeddings.data(),
        &workspace);

    float* owned = new float[embeddings.size()];
    std::copy(embeddings.begin(), embeddings.end(), owned);
    return EmbeddingResult(owned, record.length, config.hidden_dim);
}

SimilarityResult SimilarityFromEmbeddings(
    py::array_t<float, py::array::c_style | py::array::forcecast> emb1,
    py::array_t<float, py::array::c_style | py::array::forcecast> emb2) {
    pfalign::bindings::validate_2d_array(emb1, "embeddings1");
    pfalign::bindings::validate_2d_array(emb2, "embeddings2");
    if (emb1.shape(1) != emb2.shape(1)) {
        throw std::runtime_error("Embedding dimensions do not match");
    }

    const int L1 = static_cast<int>(emb1.shape(0));
    const int L2 = static_cast<int>(emb2.shape(0));
    std::vector<float> similarity(static_cast<size_t>(L1) * L2);

    pfalign::similarity::compute_similarity<pfalign::ScalarBackend>(
        emb1.data(),
        emb2.data(),
        similarity.data(),
        L1,
        L2,
        static_cast<int>(emb1.shape(1)));

    float* owned = new float[similarity.size()];
    std::copy(similarity.begin(), similarity.end(), owned);
    return SimilarityResult(owned, L1, L2);
}

pfalign::smith_waterman::SWConfig BuildSwConfig(std::optional<float> gap_open,
                                                std::optional<float> gap_extend,
                                                std::optional<float> temperature) {
    // Use SW parameters from embedded weights (trained model values) as defaults
    const auto& sw_params = EmbeddedSwParams();

    pfalign::smith_waterman::SWConfig cfg;
    cfg.affine = true;
    cfg.gap_open = gap_open.value_or(sw_params.gap_open);
    cfg.gap_extend = gap_extend.value_or(sw_params.gap);
    cfg.gap = gap_extend.value_or(sw_params.gap);
    cfg.temperature = temperature.value_or(sw_params.temperature);
    return cfg;
}

pfalign::msa::GuideTree BuildGuideTree(const std::string& method,
                                       const float* distances,
                                       int N,
                                       pfalign::memory::GrowableArena* arena) {
    std::string lower = method;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "upgma") {
        return pfalign::tree_builders::build_upgma_tree(distances, N, arena);
    } else if (lower == "nj") {
        return pfalign::tree_builders::build_nj_tree(distances, N, arena);
    } else if (lower == "bionj") {
        return pfalign::tree_builders::build_bionj_tree(distances, N, arena);
    } else if (lower == "mst") {
        return pfalign::tree_builders::build_mst_tree(distances, N, arena);
    }
    throw pfalign::errors::messages::unknown_tree_method(method);
}

MSAResult ConvertProfileToResult(const pfalign::msa::MSAResult& src) {
    const pfalign::msa::Profile* profile = src.alignment;
    if (!profile || profile->length == 0) {
        throw pfalign::errors::messages::alignment_failed("MSA profile is empty");
    }

    const int num_sequences = profile->num_sequences;
    const int alignment_length = profile->length;

    std::vector<const pfalign::SequenceEmbeddings*> cache_entries(num_sequences, nullptr);
    if (src.cache) {
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
            int cache_id = profile->seq_indices[seq_idx];
            cache_entries[seq_idx] = src.cache->get(cache_id);
        }
    }

    std::vector<std::string> sequences(num_sequences);
    std::vector<std::string> identifiers(num_sequences);

    for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const auto* entry = cache_entries[seq_idx];
        if (entry && !entry->identifier.empty()) {
            identifiers[seq_idx] = entry->identifier;
        } else {
            identifiers[seq_idx] = "sequence_" + std::to_string(
                profile->seq_indices[seq_idx]);
        }
        sequences[seq_idx].reserve(alignment_length);
    }

    std::vector<int> column_positions(num_sequences);
    for (int col = 0; col < alignment_length; ++col) {
        std::fill(column_positions.begin(), column_positions.end(), -1);
        const auto& column = profile->columns[col].positions;
        for (const auto& pos : column) {
            if (pos.seq_idx >= 0 && pos.seq_idx < num_sequences) {
                column_positions[pos.seq_idx] = pos.pos;
            }
        }
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
            const int residue_idx = column_positions[seq_idx];
            if (residue_idx < 0) {
                sequences[seq_idx].push_back('-');
                continue;
            }
            const auto* entry = cache_entries[seq_idx];
            char residue = 'X';
            if (entry && residue_idx < static_cast<int>(entry->sequence.size())
                && !entry->sequence.empty()) {
                residue = entry->sequence[residue_idx];
            }
            sequences[seq_idx].push_back(residue);
        }
    }

    return MSAResult(num_sequences,
                     alignment_length,
                     src.ecs,
                     std::move(sequences),
                     std::move(identifiers));
}

MSAResult MsaFromStructures(const std::vector<std::string>& paths,
                            const std::string& method,
                            float ecs_temperature,
                            int k_neighbors,
                            int arena_size_mb,
                            std::optional<float> gap_open,
                            std::optional<float> gap_extend,
                            std::optional<float> temperature,
                            std::optional<int> thread_count,
                            py::object progress_callback = py::none()) {
    if (paths.size() < 2) {
        throw pfalign::errors::messages::insufficient_sequences_for_msa(static_cast<int>(paths.size()));
    }

    int threads = thread_count.value_or(0);
    if (threads < 0) {
        throw pfalign::errors::messages::parameter_must_be_non_negative("thread_count", std::to_string(threads));
    }
    threads = ClampThreadCount(threads);

    pfalign::memory::GrowableArena arena(static_cast<size_t>(arena_size_mb));
    size_t scratch_mb = std::max<size_t>(64, static_cast<size_t>(arena_size_mb) / 4);
    pfalign::memory::GrowableArena scratch(scratch_mb);
    pfalign::SequenceCache cache(&arena);
    pfalign::mpnn::MPNNConfig mpnn_config = BuildMpnnConfig(k_neighbors);

    // Load all structures first (I/O is sequential)
    std::vector<StructureRecord> structures;
    structures.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        structures.push_back(LoadStructureRecord(paths[i], 0));
        // Phase 1: Update progress for each structure loaded
        if (!progress_callback.is_none()) {
            progress_callback(static_cast<int>(i + 1), static_cast<int>(paths.size()), "Loading structures");
        }
    }

    // Parallel MPNN encoding (default behavior)
    const size_t requested_threads = threads > 0 ? static_cast<size_t>(threads) : 0;
    pfalign::threading::ThreadPool pool(requested_threads, arena_size_mb);

    // Phase 2 start: Encoding (parallel, can't track individual progress easily)
    if (!progress_callback.is_none()) {
        progress_callback(0, static_cast<int>(structures.size()), "Encoding structures");
    }

    pool.parallel_for(structures.size(), [&](int tid, size_t begin, size_t end, pfalign::memory::GrowableArena& thread_arena) {
        (void)tid;  // Unused

        // Create thread-local adapter using thread's arena
        pfalign::mpnn::MPNNCacheAdapter local_adapter(cache, EmbeddedWeights(), mpnn_config, &thread_arena);

        for (size_t idx = begin; idx < end; ++idx) {
            local_adapter.add_protein(static_cast<int>(idx),
                                      structures[idx].coords.data(),
                                      structures[idx].length,
                                      structures[idx].identifier,
                                      structures[idx].sequence);
        }
    });

    // Phase 2 complete: Encoding
    if (!progress_callback.is_none()) {
        progress_callback(static_cast<int>(structures.size()), static_cast<int>(structures.size()), "Encoding structures");
    }

    // Use embedded SW parameters for distance computation (allow override)
    pfalign::smith_waterman::SWConfig sw_cfg = BuildSwConfig(gap_open, gap_extend, temperature);
    const int N = static_cast<int>(paths.size());
    std::vector<float> distances(static_cast<size_t>(N) * N, 0.0f);

    // Phase 3: Computing distances (N*N pairwise alignments)
    const int total_distances = (N * (N - 1)) / 2;  // Upper triangle only
    if (!progress_callback.is_none()) {
        progress_callback(0, total_distances, "Computing distances");
    }

    {
        py::gil_scoped_release release;  // Release GIL before spawning worker threads
        pfalign::msa::compute_distance_matrix_alignment(
            cache,
            sw_cfg,
            &scratch,
            distances.data(),
            requested_threads,  // Pass user's thread count
            [&](int current, int total) {
                if (!progress_callback.is_none()) {
                    py::gil_scoped_acquire acquire;  // Acquire GIL in worker thread for callback
                    progress_callback(current, total, "Computing distances");
                }
            });
    }  // GIL reacquired here

    // Phase 3 complete
    if (!progress_callback.is_none()) {
        progress_callback(total_distances, total_distances, "Computing distances");
    }

    // Phase 4: Building tree
    if (!progress_callback.is_none()) {
        progress_callback(0, 1, "Building guide tree");
    }

    auto tree = BuildGuideTree(method, distances.data(), N, &arena);

    // Phase 4 complete
    if (!progress_callback.is_none()) {
        progress_callback(1, 1, "Building guide tree");
    }

    // Use embedded SW parameters from trained model (allow override)
    const auto& sw_params = EmbeddedSwParams();
    pfalign::msa::MSAConfig msa_cfg;
    msa_cfg.gap_open = gap_open.value_or(sw_params.gap_open);
    msa_cfg.gap_extend = gap_extend.value_or(sw_params.gap);
    msa_cfg.gap_penalty = gap_extend.value_or(sw_params.gap);
    msa_cfg.temperature = temperature.value_or(sw_params.temperature);
    msa_cfg.ecs_temperature = ecs_temperature;
    msa_cfg.thread_count = threads;

    // Set progress callback for progressive alignment
    msa_cfg.progress_callback = [&](int current, int total) {
        if (!progress_callback.is_none()) {
            py::gil_scoped_acquire acquire;  // Acquire GIL for Python callback from C++ thread
            progress_callback(current, total, "Progressive alignment");
        }
    };

    // Phase 5: Progressive alignment (N-1 merges)
    const int total_merges = N - 1;
    if (!progress_callback.is_none()) {
        progress_callback(0, total_merges, "Progressive alignment");
    }

    pfalign::msa::MSAResult msa_result;
    {
        py::gil_scoped_release release;  // Release GIL before spawning worker threads
        msa_result = pfalign::msa::progressive_msa<pfalign::ScalarBackend>(
            cache,
            tree,
            msa_cfg,
            &arena);
    }  // GIL reacquired here

    // Phase 5 complete
    if (!progress_callback.is_none()) {
        progress_callback(total_merges, total_merges, "Progressive alignment");
    }

    auto converted = ConvertProfileToResult(msa_result);
    pfalign::msa::Profile::destroy(msa_result.alignment);
    return converted;
}

MSAResult MsaFromEmbeddings(py::sequence embeddings,
                            const std::string& method,
                            float ecs_temperature,
                            int arena_size_mb,
                            std::optional<float> gap_open,
                            std::optional<float> gap_extend,
                            std::optional<float> temperature,
                            std::optional<int> thread_count,
                            py::object progress_callback) {
    if (embeddings.size() < 2) {
        throw pfalign::errors::messages::insufficient_sequences_for_msa(static_cast<int>(embeddings.size()));
    }

    int threads = thread_count.value_or(0);
    if (threads < 0) {
        throw pfalign::errors::messages::parameter_must_be_non_negative("thread_count", std::to_string(threads));
    }
    threads = ClampThreadCount(threads);

    pfalign::memory::GrowableArena arena(static_cast<size_t>(arena_size_mb));
    size_t scratch_mb = std::max<size_t>(64, static_cast<size_t>(arena_size_mb) / 4);
    pfalign::memory::GrowableArena scratch(scratch_mb);
    pfalign::SequenceCache cache(&arena);

    int hidden_dim = -1;
    for (ssize_t idx = 0; idx < static_cast<ssize_t>(embeddings.size()); ++idx) {
        py::array_t<float, py::array::c_style | py::array::forcecast> arr =
            py::cast<py::array_t<float>>(embeddings[idx]);
        pfalign::bindings::validate_2d_array(arr, "embeddings");
        if (hidden_dim < 0) {
            hidden_dim = static_cast<int>(arr.shape(1));
        } else if (hidden_dim != static_cast<int>(arr.shape(1))) {
            throw std::runtime_error("Embedding dimensions do not match");
        }

        const int length = static_cast<int>(arr.shape(0));
        const std::string identifier = "sequence_" + std::to_string(idx);
        std::string sequence(length, 'X');
        cache.add_precomputed(static_cast<int>(idx),
                              arr.data(),
                              length,
                              hidden_dim,
                              nullptr,
                              identifier,
                              sequence);
    }

    // Use embedded SW parameters for distance computation (allow override)
    pfalign::smith_waterman::SWConfig sw_cfg = BuildSwConfig(gap_open, gap_extend, temperature);
    const int N = static_cast<int>(embeddings.size());
    std::vector<float> distances(static_cast<size_t>(N) * N, 0.0f);

    // Phase 3: Computing distances with progress callback
    const size_t requested_threads = threads > 0 ? static_cast<size_t>(threads) : 0;
    {
        py::gil_scoped_release release;  // Release GIL before spawning worker threads
        pfalign::msa::compute_distance_matrix_alignment(
            cache,
            sw_cfg,
            &scratch,
            distances.data(),
            requested_threads,  // Pass user's thread count
            [&](int current, int total) {
                if (!progress_callback.is_none()) {
                    py::gil_scoped_acquire acquire;  // Acquire GIL in worker thread for callback
                    progress_callback(current, total, "Computing distances");
                }
            });
    }  // GIL reacquired here

    // Phase 4: Building guide tree
    if (!progress_callback.is_none()) {
        progress_callback(0, 1, "Building guide tree");
    }
    auto tree = BuildGuideTree(method, distances.data(), N, &arena);
    if (!progress_callback.is_none()) {
        progress_callback(1, 1, "Building guide tree");
    }

    // Use embedded SW parameters from trained model (allow override)
    const auto& sw_params = EmbeddedSwParams();
    pfalign::msa::MSAConfig msa_cfg;
    msa_cfg.gap_open = gap_open.value_or(sw_params.gap_open);
    msa_cfg.gap_extend = gap_extend.value_or(sw_params.gap);
    msa_cfg.gap_penalty = gap_extend.value_or(sw_params.gap);
    msa_cfg.temperature = temperature.value_or(sw_params.temperature);
    msa_cfg.ecs_temperature = ecs_temperature;
    msa_cfg.thread_count = threads;

    // Phase 5: Progressive alignment with progress callback
    msa_cfg.progress_callback = [&](int current, int total) {
        if (!progress_callback.is_none()) {
            py::gil_scoped_acquire acquire;  // Acquire GIL for Python callback from C++ thread
            progress_callback(current, total, "Progressive alignment");
        }
    };

    pfalign::msa::MSAResult msa_result;
    {
        py::gil_scoped_release release;  // Release GIL before spawning worker threads
        msa_result = pfalign::msa::progressive_msa<pfalign::ScalarBackend>(
            cache,
            tree,
            msa_cfg,
            &arena);
    }  // GIL reacquired here

    auto converted = ConvertProfileToResult(msa_result);
    pfalign::msa::Profile::destroy(msa_result.alignment);
    return converted;
}

}  // namespace

PYBIND11_MODULE(_align_cpp, m) {
    m.doc() = "PFalign v2.0 - High-performance protein alignment";
    m.attr("__version__") = "2.0.0";

    // ----------------------- Custom Exception Types -----------------------
    // Register custom exceptions that map to Python error types
    static py::exception<pfalign::errors::PFalignError> exc_pfalign(m, "PFalignError");
    static py::exception<pfalign::errors::FileNotFoundError> exc_file_not_found(m, "FileNotFoundError", PyExc_FileNotFoundError);
    static py::exception<pfalign::errors::FileWriteError> exc_file_write(m, "FileWriteError", PyExc_OSError);
    static py::exception<pfalign::errors::ValidationError> exc_validation(m, "ValidationError", PyExc_ValueError);
    static py::exception<pfalign::errors::FormatError> exc_format(m, "FormatError", PyExc_ValueError);
    static py::exception<pfalign::errors::ChainNotFoundError> exc_chain_not_found(m, "ChainNotFoundError", PyExc_KeyError);
    static py::exception<pfalign::errors::DimensionError> exc_dimension(m, "DimensionError", PyExc_ValueError);
    static py::exception<pfalign::errors::AlgorithmError> exc_algorithm(m, "AlgorithmError", PyExc_RuntimeError);

    // Register exception translators for automatic C++ -> Python exception conversion
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const pfalign::errors::FileNotFoundError& e) {
            py::set_error(exc_file_not_found, e.formatted().c_str());
        } catch (const pfalign::errors::FileWriteError& e) {
            py::set_error(exc_file_write, e.formatted().c_str());
        } catch (const pfalign::errors::ValidationError& e) {
            py::set_error(exc_validation, e.formatted().c_str());
        } catch (const pfalign::errors::FormatError& e) {
            py::set_error(exc_format, e.formatted().c_str());
        } catch (const pfalign::errors::ChainNotFoundError& e) {
            py::set_error(exc_chain_not_found, e.formatted().c_str());
        } catch (const pfalign::errors::DimensionError& e) {
            py::set_error(exc_dimension, e.formatted().c_str());
        } catch (const pfalign::errors::AlgorithmError& e) {
            py::set_error(exc_algorithm, e.formatted().c_str());
        } catch (const pfalign::errors::PFalignError& e) {
            py::set_error(exc_pfalign, e.formatted().c_str());
        }
    });

    // Expose ErrorCategory enum
    py::enum_<pfalign::errors::ErrorCategory>(m, "ErrorCategory")
        .value("FileIO", pfalign::errors::ErrorCategory::FileIO)
        .value("Validation", pfalign::errors::ErrorCategory::Validation)
        .value("Format", pfalign::errors::ErrorCategory::Format)
        .value("Algorithm", pfalign::errors::ErrorCategory::Algorithm)
        .value("Resource", pfalign::errors::ErrorCategory::Resource)
        .value("UserError", pfalign::errors::ErrorCategory::UserError)
        .export_values();

    // ----------------------- Pairwise Functions -----------------------
    m.def("_pairwise_from_structures",
          &PairwiseFromStructures,
          py::arg("input1"),
          py::arg("input2"),
          py::arg("chain1") = 0,
          py::arg("chain2") = 0,
          py::arg("k_neighbors") = 30,
          py::arg("gap_open") = py::none(),
          py::arg("gap_extend") = py::none(),
          py::arg("temperature") = py::none(),
          py::arg("parallel_mpnn") = true,
          "Pairwise alignment from structures. Gap parameters default to embedded trained model values. "
          "Set parallel_mpnn=False to disable parallel MPNN encoding.");

    m.def("_pairwise_from_embeddings",
          &PairwiseFromEmbeddings,
          py::arg("embeddings1"),
          py::arg("embeddings2"),
          py::arg("gap_open") = py::none(),
          py::arg("gap_extend") = py::none(),
          py::arg("temperature") = py::none(),
          py::arg("parallel_mpnn") = true,
          "Pairwise alignment from embeddings. Gap parameters default to embedded trained model values. "
          "Set parallel_mpnn=False to disable parallel MPNN encoding.");

    // ----------------------- Structure Info -----------------------
    m.def("_get_chain_ids",
          &GetChainIds,
          py::arg("path"),
          "Get list of chain IDs from a structure file");

    m.def("_resolve_chain_index",
          &ResolveChainIndex,
          py::arg("path"),
          py::arg("chain_spec"),
          "Resolve chain name or index to chain index");

    // ----------------------- Encode -----------------------
    m.def("_encode_structure",
          &EncodeStructure,
          py::arg("path"),
          py::arg("chain") = 0,
          py::arg("k_neighbors") = 30);

    // ----------------------- Similarity -----------------------
    m.def("_similarity_impl",
          &SimilarityFromEmbeddings,
          py::arg("embeddings1"),
          py::arg("embeddings2"));

    // ----------------------- MSA -----------------------
    m.def("_msa_from_structures",
          &MsaFromStructures,
          py::arg("paths"),
          py::arg("method") = "upgma",
          py::arg("ecs_temperature") = 5.0f,
          py::arg("k_neighbors") = 30,
          py::arg("arena_size_mb") = kDefaultArenaMb,
          py::arg("gap_open") = py::none(),
          py::arg("gap_extend") = py::none(),
          py::arg("temperature") = py::none(),
          py::arg("thread_count") = py::none(),
          py::arg("progress_callback") = py::none(),
          "MSA from structures. Gap parameters default to embedded trained model values.");

    m.def("_msa_from_embeddings",
          &MsaFromEmbeddings,
          py::arg("embeddings"),
          py::arg("method") = "upgma",
          py::arg("ecs_temperature") = 5.0f,
          py::arg("arena_size_mb") = kDefaultArenaMb,
          py::arg("gap_open") = py::none(),
          py::arg("gap_extend") = py::none(),
          py::arg("temperature") = py::none(),
          py::arg("thread_count") = py::none(),
          py::arg("progress_callback") = py::none(),
          "MSA from embeddings. Gap parameters default to embedded trained model values.");

    // ----------------------- Distance Matrix Computation -----------------------
    m.def("_compute_distances_from_embeddings",
          [](const std::vector<py::array_t<float>>& embeddings,
             std::optional<float> gap_open,
             std::optional<float> gap_extend,
             std::optional<float> temperature) -> py::array_t<float> {

              if (embeddings.empty()) {
                  throw std::runtime_error("Must provide at least one embedding");
              }

              const int N = static_cast<int>(embeddings.size());

              pfalign::memory::GrowableArena arena(static_cast<size_t>(kDefaultArenaMb));
              pfalign::SequenceCache cache(&arena);

              int hidden_dim = -1;
              for (int i = 0; i < N; i++) {
                  auto buf = embeddings[i].request();
                  if (buf.ndim != 2) {
                      throw std::runtime_error("Embeddings must be 2D arrays");
                  }
                  if (hidden_dim < 0) {
                      hidden_dim = static_cast<int>(buf.shape[1]);
                  } else if (hidden_dim != static_cast<int>(buf.shape[1])) {
                      throw std::runtime_error("Embedding dimensions do not match");
                  }

                  const int length = static_cast<int>(buf.shape[0]);
                  const std::string identifier = "seq_" + std::to_string(i);
                  std::string sequence(length, 'X');
                  cache.add_precomputed(i,
                                       static_cast<const float*>(buf.ptr),
                                       length,
                                       hidden_dim,
                                       nullptr,
                                       identifier,
                                       sequence);
              }

              pfalign::memory::GrowableArena scratch(10 * 1024 * 1024);
              pfalign::smith_waterman::SWConfig sw_cfg = BuildSwConfig(gap_open, gap_extend, temperature);

              std::vector<float> distances(static_cast<size_t>(N) * N, 0.0f);
              pfalign::msa::compute_distance_matrix_alignment(
                  cache,
                  sw_cfg,
                  &scratch,
                  distances.data());

              auto result = py::array_t<float>({N, N});
              auto result_buf = result.request();
              float* result_ptr = static_cast<float*>(result_buf.ptr);
              std::copy(distances.begin(), distances.end(), result_ptr);

              return result;
          },
          py::arg("embeddings"),
          py::arg("gap_open") = py::none(),
          py::arg("gap_extend") = py::none(),
          py::arg("temperature") = py::none(),
          "Compute pairwise distance matrix from embeddings. Gap parameters default to embedded trained model values.");

    // ----------------------- Progressive Alignment with Custom Tree -----------------------
    m.def("_progressive_align",
          [](const std::vector<py::array_t<float>>& embeddings,
             const GuideTreeHolder& tree_holder,
             std::optional<float> gap_open,
             std::optional<float> gap_extend,
             std::optional<float> temperature,
             float ecs_temperature,
             int arena_size_mb,
             std::optional<int> thread_count) -> MSAResult {

              if (embeddings.empty()) {
                  throw std::runtime_error("Must provide at least one embedding");
              }

              const int N = static_cast<int>(embeddings.size());
              if (N != tree_holder.num_sequences()) {
                  throw std::runtime_error(
                      "Number of embeddings (" + std::to_string(N) +
                      ") does not match tree size (" + std::to_string(tree_holder.num_sequences()) + ")");
              }

              int threads = thread_count.value_or(0);
              if (threads < 0) {
                  throw pfalign::errors::messages::parameter_must_be_non_negative("thread_count", std::to_string(threads));
              }
              threads = ClampThreadCount(threads);

              pfalign::memory::GrowableArena arena(static_cast<size_t>(arena_size_mb));
              pfalign::SequenceCache cache(&arena);

              int hidden_dim = -1;
              for (int i = 0; i < N; i++) {
                  auto buf = embeddings[i].request();
                  if (buf.ndim != 2) {
                      throw std::runtime_error("Embeddings must be 2D arrays");
                  }
                  if (hidden_dim < 0) {
                      hidden_dim = static_cast<int>(buf.shape[1]);
                  } else if (hidden_dim != static_cast<int>(buf.shape[1])) {
                      throw std::runtime_error("Embedding dimensions do not match");
                  }

                  const int length = static_cast<int>(buf.shape[0]);
                  const std::string identifier = "sequence_" + std::to_string(i);
                  std::string sequence(length, 'X');
                  cache.add_precomputed(i,
                                       static_cast<const float*>(buf.ptr),
                                       length,
                                       hidden_dim,
                                       nullptr,
                                       identifier,
                                       sequence);
              }

              // Use embedded SW parameters from trained model (allow override)
              const auto& sw_params = EmbeddedSwParams();
              pfalign::msa::MSAConfig msa_cfg;
              msa_cfg.gap_open = gap_open.value_or(sw_params.gap_open);
              msa_cfg.gap_extend = gap_extend.value_or(sw_params.gap);
              msa_cfg.gap_penalty = gap_extend.value_or(sw_params.gap);
              msa_cfg.temperature = temperature.value_or(sw_params.temperature);
              msa_cfg.ecs_temperature = ecs_temperature;
              msa_cfg.thread_count = threads;

              auto msa_result = pfalign::msa::progressive_msa<pfalign::ScalarBackend>(
                  cache,
                  tree_holder.get_tree(),
                  msa_cfg,
                  &arena);

              auto converted = ConvertProfileToResult(msa_result);
              pfalign::msa::Profile::destroy(msa_result.alignment);
              return converted;
          },
          py::arg("embeddings"),
          py::arg("tree"),
          py::arg("gap_open") = py::none(),
          py::arg("gap_extend") = py::none(),
          py::arg("temperature") = py::none(),
          py::arg("ecs_temperature") = 5.0f,
          py::arg("arena_size_mb") = kDefaultArenaMb,
          py::arg("thread_count") = py::none(),
          R"(
Perform progressive multiple sequence alignment using a custom guide tree.

Args:
    embeddings: List of embedding arrays (N, D) for each sequence
    tree: GuideTree object from pfalign.tree.upgma/nj/bionj/mst
    gap_open: Gap opening penalty (default: trained model value)
    gap_extend: Gap extension penalty (default: trained model value)
    temperature: Temperature for soft alignment (default: trained model value)
    ecs_temperature: ECS temperature for profile similarity (default: 5.0)
    arena_size_mb: Memory arena size in MB (default: 200)
    thread_count: Number of threads (default: auto)

Returns:
    MSAResult object with aligned sequences

Example:
    >>> emb1 = pfalign.encode("protein1.pdb")
    >>> emb2 = pfalign.encode("protein2.pdb")
    >>> emb3 = pfalign.encode("protein3.pdb")
    >>> distances = pfalign.compute_distances([emb1, emb2, emb3])
    >>> tree = pfalign.tree.upgma(distances)
    >>> msa = pfalign.progressive_align([emb1, emb2, emb3], tree)
    >>> msa.write_fasta("output.fasta")
          )");

    // ----------------------- Metrics Submodule -----------------------
    py::module_ metrics = m.def_submodule("metrics", "Structural and alignment quality metrics");

    // RMSD calculation
    metrics.def("rmsd",
        [](py::array_t<float> coords1, py::array_t<float> coords2, bool aligned = true) -> float {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();

            if (buf1.ndim != 2 || buf2.ndim != 2) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must have shape (N, 3)");
            }
            if (buf1.shape[0] != buf2.shape[0]) {
                throw std::runtime_error("coords1 and coords2 must have same number of points");
            }

            int N = static_cast<int>(buf1.shape[0]);
            const float* P = static_cast<const float*>(buf1.ptr);
            const float* Q = static_cast<const float*>(buf2.ptr);

            if (!aligned) {
                // Need to run Kabsch first
                std::vector<float> R(9);
                std::vector<float> t(3);
                std::vector<float> P_aligned(N * 3);
                float rmsd;

                pfalign::kabsch::kabsch_align<pfalign::ScalarBackend>(
                    P, Q, N, R.data(), t.data(), &rmsd, nullptr, nullptr
                );

                return rmsd;
            } else {
                // Coords already aligned, just compute RMSD
                float sum_sq = 0.0f;
                for (int i = 0; i < N; i++) {
                    float dx = P[i*3 + 0] - Q[i*3 + 0];
                    float dy = P[i*3 + 1] - Q[i*3 + 1];
                    float dz = P[i*3 + 2] - Q[i*3 + 2];
                    sum_sq += dx*dx + dy*dy + dz*dz;
                }
                return std::sqrt(sum_sq / N);
            }
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("aligned") = true,
        R"(
Compute RMSD (Root Mean Square Deviation) between two coordinate sets.

Args:
    coords1: Coordinates array (N, 3) - first structure
    coords2: Coordinates array (N, 3) - second structure
    aligned: If False, performs Kabsch alignment first (default: True)

Returns:
    RMSD value in Ångströms

Example:
    >>> coords1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    >>> coords2 = np.array([[0.1, 0, 0], [1.1, 0, 0], [2.1, 0, 0]])
    >>> rmsd = pfalign.metrics.rmsd(coords1, coords2)
    >>> print(f"RMSD: {rmsd:.2f} Å")
        )");

    // TM-score calculation
    metrics.def("tm_score",
        [](py::array_t<float> coords1, py::array_t<float> coords2,
           int len1, int /* len2 */, bool aligned = true) -> float {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();

            if (buf1.ndim != 2 || buf2.ndim != 2) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must have shape (N, 3)");
            }
            if (buf1.shape[0] != buf2.shape[0]) {
                throw std::runtime_error("coords1 and coords2 must have same number of aligned points");
            }

            int N = static_cast<int>(buf1.shape[0]);
            const float* P = static_cast<const float*>(buf1.ptr);
            const float* Q = static_cast<const float*>(buf2.ptr);

            if (!aligned) {
                // Need to run Kabsch first
                std::vector<float> R(9);
                std::vector<float> t(3);
                std::vector<float> P_aligned(N * 3);

                pfalign::kabsch::kabsch_align<pfalign::ScalarBackend>(
                    P, Q, N, R.data(), t.data(), nullptr, nullptr, nullptr
                );

                // Apply transformation to P
                for (int i = 0; i < N; i++) {
                    float x = P[i*3 + 0];
                    float y = P[i*3 + 1];
                    float z = P[i*3 + 2];
                    P_aligned[i*3 + 0] = R[0]*x + R[1]*y + R[2]*z + t[0];
                    P_aligned[i*3 + 1] = R[3]*x + R[4]*y + R[5]*z + t[1];
                    P_aligned[i*3 + 2] = R[6]*x + R[7]*y + R[8]*z + t[2];
                }

                return pfalign::structural_metrics::compute_tm_score<pfalign::ScalarBackend>(
                    P_aligned.data(), Q, N, len1
                );
            } else {
                return pfalign::structural_metrics::compute_tm_score<pfalign::ScalarBackend>(
                    P, Q, N, len1
                );
            }
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("len1"),
        py::arg("len2"),
        py::arg("aligned") = true,
        R"(
Compute TM-score (Template Modeling score) between two structures.

TM-score measures global fold similarity. Range: [0, 1], where 1 is perfect.
- > 0.5: Same fold (high confidence)
- 0.3-0.5: Possible homology
- < 0.3: Different fold

Args:
    coords1: Aligned coordinates (N, 3) - first structure
    coords2: Aligned coordinates (N, 3) - second structure
    len1: Full sequence length of structure 1 (for normalization)
    len2: Full sequence length of structure 2 (for normalization)
    aligned: If False, performs Kabsch alignment first (default: True)

Returns:
    TM-score value in [0, 1]

Example:
    >>> tm = pfalign.metrics.tm_score(coords1, coords2, len1=150, len2=145)
    >>> print(f"TM-score: {tm:.3f}")
        )");

    // GDT-TS and GDT-HA calculation
    metrics.def("gdt",
        [](py::array_t<float> coords1, py::array_t<float> coords2, bool aligned = true)
            -> std::tuple<float, float> {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();

            if (buf1.ndim != 2 || buf2.ndim != 2) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must have shape (N, 3)");
            }
            if (buf1.shape[0] != buf2.shape[0]) {
                throw std::runtime_error("coords1 and coords2 must have same number of points");
            }

            int N = static_cast<int>(buf1.shape[0]);
            const float* P = static_cast<const float*>(buf1.ptr);
            const float* Q = static_cast<const float*>(buf2.ptr);

            float gdt_ts, gdt_ha;

            if (!aligned) {
                // Need to run Kabsch first
                std::vector<float> R(9);
                std::vector<float> t(3);
                std::vector<float> P_aligned(N * 3);

                pfalign::kabsch::kabsch_align<pfalign::ScalarBackend>(
                    P, Q, N, R.data(), t.data(), nullptr, nullptr, nullptr
                );

                // Apply transformation to P
                for (int i = 0; i < N; i++) {
                    float x = P[i*3 + 0];
                    float y = P[i*3 + 1];
                    float z = P[i*3 + 2];
                    P_aligned[i*3 + 0] = R[0]*x + R[1]*y + R[2]*z + t[0];
                    P_aligned[i*3 + 1] = R[3]*x + R[4]*y + R[5]*z + t[1];
                    P_aligned[i*3 + 2] = R[6]*x + R[7]*y + R[8]*z + t[2];
                }

                pfalign::structural_metrics::compute_gdt<pfalign::ScalarBackend>(
                    P_aligned.data(), Q, N, &gdt_ts, &gdt_ha
                );
            } else {
                pfalign::structural_metrics::compute_gdt<pfalign::ScalarBackend>(
                    P, Q, N, &gdt_ts, &gdt_ha
                );
            }

            return std::make_tuple(gdt_ts, gdt_ha);
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("aligned") = true,
        R"(
Compute GDT-TS and GDT-HA scores.

GDT (Global Distance Test) measures the percentage of residues aligned
within distance cutoffs:
- GDT-TS: Cutoffs [1, 2, 4, 8] Å (standard)
- GDT-HA: Cutoffs [0.5, 1, 2, 4] Å (high accuracy)

Range: [0, 1] where 1 is perfect alignment
- > 0.7: High quality
- 0.5-0.7: Good quality
- < 0.5: Poor quality

Args:
    coords1: Coordinates (N, 3) - first structure
    coords2: Coordinates (N, 3) - second structure
    aligned: If False, performs Kabsch alignment first (default: True)

Returns:
    Tuple of (gdt_ts, gdt_ha) scores

Example:
    >>> gdt_ts, gdt_ha = pfalign.metrics.gdt(coords1, coords2)
    >>> print(f"GDT-TS: {gdt_ts:.3f}, GDT-HA: {gdt_ha:.3f}")
        )");

    // Individual GDT-TS
    metrics.def("gdt_ts",
        [](py::array_t<float> coords1, py::array_t<float> coords2, bool aligned = true) -> float {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();

            if (buf1.ndim != 2 || buf2.ndim != 2 || buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[0] != buf2.shape[0]) {
                throw std::runtime_error("coords1 and coords2 must have same number of points");
            }

            int N = static_cast<int>(buf1.shape[0]);
            const float* P = static_cast<const float*>(buf1.ptr);
            const float* Q = static_cast<const float*>(buf2.ptr);
            float gdt_ts, gdt_ha;

            if (!aligned) {
                std::vector<float> R(9), t(3), P_aligned(N * 3);
                pfalign::kabsch::kabsch_align<pfalign::ScalarBackend>(P, Q, N, R.data(), t.data(), nullptr, nullptr, nullptr);
                for (int i = 0; i < N; i++) {
                    float x = P[i*3], y = P[i*3+1], z = P[i*3+2];
                    P_aligned[i*3] = R[0]*x + R[1]*y + R[2]*z + t[0];
                    P_aligned[i*3+1] = R[3]*x + R[4]*y + R[5]*z + t[1];
                    P_aligned[i*3+2] = R[6]*x + R[7]*y + R[8]*z + t[2];
                }
                pfalign::structural_metrics::compute_gdt<pfalign::ScalarBackend>(P_aligned.data(), Q, N, &gdt_ts, &gdt_ha);
            } else {
                pfalign::structural_metrics::compute_gdt<pfalign::ScalarBackend>(P, Q, N, &gdt_ts, &gdt_ha);
            }
            return gdt_ts;
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("aligned") = true,
        "Compute GDT-TS score only. See gdt() for details.");

    // Individual GDT-HA
    metrics.def("gdt_ha",
        [](py::array_t<float> coords1, py::array_t<float> coords2, bool aligned = true) -> float {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();

            if (buf1.ndim != 2 || buf2.ndim != 2 || buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[0] != buf2.shape[0]) {
                throw std::runtime_error("coords1 and coords2 must have same number of points");
            }

            int N = static_cast<int>(buf1.shape[0]);
            const float* P = static_cast<const float*>(buf1.ptr);
            const float* Q = static_cast<const float*>(buf2.ptr);
            float gdt_ts, gdt_ha;

            if (!aligned) {
                std::vector<float> R(9), t(3), P_aligned(N * 3);
                pfalign::kabsch::kabsch_align<pfalign::ScalarBackend>(P, Q, N, R.data(), t.data(), nullptr, nullptr, nullptr);
                for (int i = 0; i < N; i++) {
                    float x = P[i*3], y = P[i*3+1], z = P[i*3+2];
                    P_aligned[i*3] = R[0]*x + R[1]*y + R[2]*z + t[0];
                    P_aligned[i*3+1] = R[3]*x + R[4]*y + R[5]*z + t[1];
                    P_aligned[i*3+2] = R[6]*x + R[7]*y + R[8]*z + t[2];
                }
                pfalign::structural_metrics::compute_gdt<pfalign::ScalarBackend>(P_aligned.data(), Q, N, &gdt_ts, &gdt_ha);
            } else {
                pfalign::structural_metrics::compute_gdt<pfalign::ScalarBackend>(P, Q, N, &gdt_ts, &gdt_ha);
            }
            return gdt_ha;
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("aligned") = true,
        "Compute GDT-HA score only. See gdt() for details.");

    // Sequence identity calculation
    metrics.def("identity",
        [](const std::string& seq1, const std::string& seq2, bool ignore_gaps = true) -> float {
            if (seq1.length() != seq2.length()) {
                throw std::runtime_error("Sequences must have equal length (aligned sequences)");
            }

            int matches = 0;
            int total = 0;

            for (size_t i = 0; i < seq1.length(); i++) {
                char c1 = seq1[i];
                char c2 = seq2[i];

                if (ignore_gaps && (c1 == '-' || c2 == '-')) {
                    continue;  // Skip gap positions
                }

                total++;
                if (c1 == c2) {
                    matches++;
                }
            }

            if (total == 0) {
                return 0.0f;
            }

            return static_cast<float>(matches) / static_cast<float>(total);
        },
        py::arg("seq1"),
        py::arg("seq2"),
        py::arg("ignore_gaps") = true,
        R"(
Compute sequence identity between two aligned sequences.

Args:
    seq1: First aligned sequence (with gaps)
    seq2: Second aligned sequence (with gaps)
    ignore_gaps: If True, only count non-gap positions (default: True)

Returns:
    Identity fraction in [0, 1]

Example:
    >>> identity = pfalign.metrics.identity("AC-DEF", "ACGDEF")
    >>> print(f"Identity: {identity:.1%}")
        )");

    // ECS score from MSAResult
    metrics.def("ecs",
        [](const MSAResult& msa, float /* temperature */ = 5.0f) -> float {
            // ECS is already computed in MSAResult, just return it
            return msa.ecs_score();
        },
        py::arg("msa"),
        py::arg("temperature") = 5.0f,
        R"(
Compute ECS (Expected Column Score) for MSA quality.

ECS measures the quality of column conservation in a multiple sequence alignment.
Higher scores indicate better-conserved columns.

Args:
    msa: MSAResult object
    temperature: ECS temperature parameter (default: 5.0)

Returns:
    ECS score (typically 0-1, higher is better)

Example:
    >>> msa = pfalign.msa(structures)
    >>> ecs = pfalign.metrics.ecs(msa)
    >>> print(f"ECS: {ecs:.3f}")
        )");

    // lDDT (Local Distance Difference Test)
    metrics.def("lddt",
        [](py::array_t<float> coords1, py::array_t<float> coords2,
           py::array_t<int> alignment, float radius = 15.0f) -> float {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();
            auto buf_aln = alignment.request();

            if (buf1.ndim != 2 || buf2.ndim != 2) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must have shape (N, 3)");
            }
            if (buf_aln.ndim != 2 || buf_aln.shape[1] != 2) {
                throw std::runtime_error("alignment must be 2D array (M, 2)");
            }

            int L1 = static_cast<int>(buf1.shape[0]);
            int L2 = static_cast<int>(buf2.shape[0]);
            int aligned_length = static_cast<int>(buf_aln.shape[0]);

            const float* ca1 = static_cast<const float*>(buf1.ptr);
            const float* ca2 = static_cast<const float*>(buf2.ptr);
            const int* aln = static_cast<const int*>(buf_aln.ptr);

            // Compute distance matrices
            std::vector<float> dist1(L1 * L1);
            std::vector<float> dist2(L2 * L2);

            pfalign::structural_metrics::compute_distance_matrix<pfalign::ScalarBackend>(
                ca1, L1, dist1.data()
            );
            pfalign::structural_metrics::compute_distance_matrix<pfalign::ScalarBackend>(
                ca2, L2, dist2.data()
            );

            // Set up lDDT parameters
            pfalign::structural_metrics::LDDTParams params;
            params.R0 = radius;

            // Compute lDDT
            return pfalign::structural_metrics::lddt_pairwise<pfalign::ScalarBackend>(
                dist1.data(), dist2.data(), aln, aligned_length, params
            );
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("alignment"),
        py::arg("radius") = 15.0f,
        R"DOC(
Compute lDDT (Local Distance Difference Test) score.

lDDT is a superposition-free metric that measures local structural similarity
by comparing distances between nearby residues. More sensitive to local geometry
than global metrics like RMSD.

Args:
    coords1: Coordinates for structure 1 (L1, 3)
    coords2: Coordinates for structure 2 (L2, 3)
    alignment: Aligned position pairs (M, 2) where alignment[i] = [pos1, pos2]
    radius: Inclusion radius in Ångströms (default: 15.0)

Returns:
    lDDT score in [0, 1] where 1 is perfect
    - > 0.8: High quality model
    - 0.6-0.8: Good model
    - < 0.6: Poor model

Example:
    >>> alignment = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    >>> lddt = pfalign.metrics.lddt(coords1, coords2, alignment, radius=15.0)
    >>> print(f"lDDT: {lddt:.3f}")
)DOC");

    // DALI score
    metrics.def("dali_score",
        [](py::array_t<float> coords1, py::array_t<float> coords2,
           py::array_t<int> alignment, int len1, int len2,
           float horizon = 20.0f) -> std::tuple<float, float> {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();
            auto buf_aln = alignment.request();

            if (buf1.ndim != 2 || buf2.ndim != 2) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must have shape (N, 3)");
            }
            if (buf_aln.ndim != 2 || buf_aln.shape[1] != 2) {
                throw std::runtime_error("alignment must be 2D array (M, 2)");
            }

            int L1 = static_cast<int>(buf1.shape[0]);
            int L2 = static_cast<int>(buf2.shape[0]);
            int aligned_length = static_cast<int>(buf_aln.shape[0]);

            const float* ca1 = static_cast<const float*>(buf1.ptr);
            const float* ca2 = static_cast<const float*>(buf2.ptr);
            const int* aln = static_cast<const int*>(buf_aln.ptr);

            // Compute distance matrices
            std::vector<float> dist1(L1 * L1);
            std::vector<float> dist2(L2 * L2);

            pfalign::structural_metrics::compute_distance_matrix<pfalign::ScalarBackend>(
                ca1, L1, dist1.data()
            );
            pfalign::structural_metrics::compute_distance_matrix<pfalign::ScalarBackend>(
                ca2, L2, dist2.data()
            );

            // Set up DALI parameters
            pfalign::structural_metrics::DALIParams params;
            params.horizon = horizon;

            // Compute DALI score
            auto result = pfalign::structural_metrics::dali_score<pfalign::ScalarBackend>(
                dist1.data(), dist2.data(), aln, aligned_length, len1, len2, params
            );

            return std::make_tuple(result.score, result.Z);
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("alignment"),
        py::arg("len1"),
        py::arg("len2"),
        py::arg("horizon") = 20.0f,
        R"DOC(
Compute DALI (Distance Alignment) score and Z-score.

DALI measures structural similarity by comparing internal distance matrices.
The Z-score is length-normalized for statistical significance.

Args:
    coords1: Coordinates for structure 1 (L1, 3)
    coords2: Coordinates for structure 2 (L2, 3)
    alignment: Aligned position pairs (M, 2) where alignment[i] = [pos1, pos2]
    len1: Full length of structure 1 (for Z-score normalization)
    len2: Full length of structure 2 (for Z-score normalization)
    horizon: Distance weighting decay parameter in Ångströms (default: 20.0)

Returns:
    Tuple of (dali_score, z_score)
    - Z > 2: Likely structural similarity
    - Z > 5: High confidence homology
    - Z > 20: Highly significant similarity

Example:
    >>> alignment = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    >>> score, z = pfalign.metrics.dali_score(coords1, coords2, alignment, 150, 145)
    >>> print(f"DALI: {score:.2f} (Z={z:.2f})")
)DOC");

    // ----------------------- Structure/Superposition Submodule -----------------------
    py::module_ structure = m.def_submodule("structure", "Structural superposition and transformation utilities");

    // Kabsch algorithm
    structure.def("kabsch",
        [](py::array_t<float> coords1, py::array_t<float> coords2, py::object /* weights */ = py::none())
            -> std::tuple<py::array_t<float>, py::array_t<float>, float> {
            auto buf1 = coords1.request();
            auto buf2 = coords2.request();

            if (buf1.ndim != 2 || buf2.ndim != 2 || buf1.shape[1] != 3 || buf2.shape[1] != 3) {
                throw std::runtime_error("coords must be 2D arrays (N, 3)");
            }
            if (buf1.shape[0] != buf2.shape[0]) {
                throw std::runtime_error("coords1 and coords2 must have same number of points");
            }

            int N = static_cast<int>(buf1.shape[0]);
            const float* P = static_cast<const float*>(buf1.ptr);
            const float* Q = static_cast<const float*>(buf2.ptr);

            // Output arrays
            std::vector<float> R(9);
            std::vector<float> t(3);
            float rmsd;

            // Call Kabsch
            pfalign::kabsch::kabsch_align<pfalign::ScalarBackend>(
                P, Q, N, R.data(), t.data(), &rmsd, nullptr, nullptr
            );

            // Convert to NumPy arrays
            auto R_array = py::array_t<float>({3, 3});
            auto t_array = py::array_t<float>(3);

            auto R_buf = R_array.request();
            auto t_buf = t_array.request();

            std::copy(R.begin(), R.end(), static_cast<float*>(R_buf.ptr));
            std::copy(t.begin(), t.end(), static_cast<float*>(t_buf.ptr));

            return std::make_tuple(R_array, t_array, rmsd);
        },
        py::arg("coords1"),
        py::arg("coords2"),
        py::arg("weights") = py::none(),
        R"(
Compute optimal rotation and translation using Kabsch algorithm.

Finds the optimal rigid-body transformation (rotation + translation) that
minimizes RMSD between two sets of corresponding 3D points.

Args:
    coords1: Source coordinates (N, 3)
    coords2: Target coordinates (N, 3)
    weights: Optional per-point weights (N,) - currently unused

Returns:
    Tuple of (rotation, translation, rmsd):
        - rotation: 3x3 rotation matrix
        - translation: 3D translation vector
        - rmsd: Root mean square deviation after alignment

Example:
    >>> R, t, rmsd = pfalign.structure.kabsch(coords1, coords2)
    >>> print(f"RMSD: {rmsd:.3f} Å")
    >>> print(f"Rotation matrix shape: {R.shape}")
    >>> print(f"Translation vector shape: {t.shape}")
        )");

    // Transform coordinates
    structure.def("transform",
        [](py::array_t<float> coords, py::array_t<float> rotation, py::array_t<float> translation)
            -> py::array_t<float> {
            auto coords_buf = coords.request();
            auto R_buf = rotation.request();
            auto t_buf = translation.request();

            if (coords_buf.ndim != 2 || coords_buf.shape[1] != 3) {
                throw std::runtime_error("coords must be 2D array (N, 3)");
            }
            if (R_buf.ndim != 2 || R_buf.shape[0] != 3 || R_buf.shape[1] != 3) {
                throw std::runtime_error("rotation must be 3x3 matrix");
            }
            if (t_buf.ndim != 1 || t_buf.shape[0] != 3) {
                throw std::runtime_error("translation must be 3D vector");
            }

            int N = static_cast<int>(coords_buf.shape[0]);
            const float* P = static_cast<const float*>(coords_buf.ptr);
            const float* R = static_cast<const float*>(R_buf.ptr);
            const float* t = static_cast<const float*>(t_buf.ptr);

            // Create output array
            auto result = py::array_t<float>({N, 3});
            auto result_buf = result.request();
            float* out = static_cast<float*>(result_buf.ptr);

            // Apply transformation: out = R @ P + t
            for (int i = 0; i < N; i++) {
                float x = P[i*3 + 0];
                float y = P[i*3 + 1];
                float z = P[i*3 + 2];

                out[i*3 + 0] = R[0]*x + R[1]*y + R[2]*z + t[0];
                out[i*3 + 1] = R[3]*x + R[4]*y + R[5]*z + t[1];
                out[i*3 + 2] = R[6]*x + R[7]*y + R[8]*z + t[2];
            }

            return result;
        },
        py::arg("coords"),
        py::arg("rotation"),
        py::arg("translation"),
        R"(
Apply rotation and translation to coordinates.

Args:
    coords: Input coordinates (N, 3)
    rotation: Rotation matrix (3, 3)
    translation: Translation vector (3,)

Returns:
    Transformed coordinates (N, 3)

Example:
    >>> R, t, rmsd = pfalign.structure.kabsch(coords1, coords2)
    >>> coords1_aligned = pfalign.structure.transform(coords1, R, t)
        )");

    // Get coordinates from structure file
    structure.def("get_coords",
        [](const std::string& structure_path, int chain_index = 0, const std::string& atom_type = "CA")
            -> py::array_t<float> {
            // Load structure
            pfalign::io::Protein protein = pfalign::commands::LoadStructureFile(structure_path);

            if (protein.num_chains() == 0) {
                throw std::runtime_error("No chains found in structure: " + structure_path);
            }
            if (chain_index < 0 || static_cast<size_t>(chain_index) >= protein.num_chains()) {
                throw std::runtime_error("Chain index out of range: " + std::to_string(chain_index));
            }

            const auto& chain = protein.get_chain(static_cast<size_t>(chain_index));
            int L = static_cast<int>(chain.size());

            if (atom_type == "CA") {
                // Extract CA atoms only
                std::vector<float> backbone = protein.get_backbone_coords(static_cast<size_t>(chain_index));

                auto coords = py::array_t<float>({L, 3});
                auto buf = coords.request();
                float* ptr = static_cast<float*>(buf.ptr);

                pfalign::structural_metrics::extract_ca_atoms(
                    backbone.data(), L, ptr
                );

                return coords;
            } else if (atom_type == "backbone") {
                // All backbone atoms (N, CA, C, O) - 4 atoms per residue
                std::vector<float> backbone = protein.get_backbone_coords(static_cast<size_t>(chain_index));

                auto coords = py::array_t<float>({L * 4, 3});
                auto buf = coords.request();
                float* ptr = static_cast<float*>(buf.ptr);

                std::copy(backbone.begin(), backbone.end(), ptr);
                return coords;
            } else {
                throw std::runtime_error("Invalid atom_type. Must be 'CA' or 'backbone'");
            }
        },
        py::arg("structure_path"),
        py::arg("chain") = 0,
        py::arg("atom_type") = "CA",
        R"(
Extract coordinates from a structure file.

Args:
    structure_path: Path to PDB or CIF file
    chain: Chain index (default: 0)
    atom_type: 'CA' for Calpha atoms only, 'backbone' for N,CA,C,O (default: 'CA')

Returns:
    Coordinates array (N, 3) where N depends on atom_type:
        - 'CA': N = number of residues
        - 'backbone': N = number of residues * 4

Example:
    >>> ca_coords = pfalign.structure.get_coords("protein.pdb", chain=0, atom_type="CA")
    >>> print(ca_coords.shape)  # (L, 3) where L is sequence length
        )");

    // ----------------------- GuideTree and Tree Building -----------------------
    py::class_<GuideTreeHolder>(m, "GuideTree")
        .def("num_sequences", &GuideTreeHolder::num_sequences,
             "Number of leaf sequences in the tree")
        .def("num_nodes", &GuideTreeHolder::num_nodes,
             "Total number of nodes (leaves + internal)")
        .def("root_index", &GuideTreeHolder::root_index,
             "Index of the root node")
        .def("to_newick",
             [](const GuideTreeHolder& holder, const std::vector<std::string>& labels) -> std::string {
                 // Validate labels count
                 int N = holder.num_sequences();
                 if (static_cast<int>(labels.size()) != N) {
                     throw std::runtime_error("Number of labels must match number of sequences");
                 }

                 // Convert labels to C-style array
                 std::vector<const char*> label_ptrs(N);
                 for (int i = 0; i < N; ++i) {
                     label_ptrs[i] = labels[i].c_str();
                 }

                 // Generate Newick string
                 auto temp_arena = std::make_unique<pfalign::memory::GrowableArena>(10);
                 const char* newick_cstr = holder.get_tree().to_newick(label_ptrs.data(), temp_arena.get());

                 // Copy to std::string before arena is destroyed
                 return std::string(newick_cstr);
             },
             py::arg("labels"),
             R"(
Convert guide tree to Newick format string.

The Newick format is a standard way to represent phylogenetic trees using
nested parentheses and branch lengths.

Args:
    labels: List of sequence names [num_sequences]
            Must provide exactly one label per leaf sequence

Returns:
    Newick format string representing the tree topology

Example:
    >>> distances = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
    >>> tree = pfalign.tree.upgma(distances)
    >>> newick = tree.to_newick(["seqA", "seqB", "seqC"])
    >>> print(newick)
    '((seqA:0.5,seqB:0.5):1.0,seqC:2.0);'

Note:
    The Newick string includes branch lengths computed during tree construction.
    Format: ((name1:length1,name2:length2):parent_length,name3:length3);
        )")
        .def("__repr__", [](const GuideTreeHolder& holder) {
            return "<GuideTree with " + std::to_string(holder.num_sequences()) + " sequences>";
        });

    py::module_ tree = m.def_submodule("tree", "Guide tree construction for multiple sequence alignment");

    tree.def("upgma",
        [](py::array_t<float> distances) -> GuideTreeHolder {
            auto buf = distances.request();
            if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
                throw std::runtime_error("Distance matrix must be square (N, N)");
            }
            int N = static_cast<int>(buf.shape[0]);
            const float* dist_ptr = static_cast<const float*>(buf.ptr);

            auto arena = std::make_unique<pfalign::memory::GrowableArena>(100);
            pfalign::types::GuideTree tree = pfalign::tree_builders::build_upgma_tree(
                dist_ptr, N, arena.get());

            return GuideTreeHolder(std::move(arena), std::move(tree));
        },
        py::arg("distances"),
        R"(
Build UPGMA (Unweighted Pair Group Method with Arithmetic Mean) guide tree.

Args:
    distances: Square distance matrix (N, N) of pairwise distances

Returns:
    GuideTree object for use with progressive_align()

Example:
    >>> distances = pfalign.compute_distances(embeddings)
    >>> tree = pfalign.tree.upgma(distances)
    >>> print(tree.num_sequences())
        )");

    tree.def("nj",
        [](py::array_t<float> distances) -> GuideTreeHolder {
            auto buf = distances.request();
            if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
                throw std::runtime_error("Distance matrix must be square (N, N)");
            }
            int N = static_cast<int>(buf.shape[0]);
            const float* dist_ptr = static_cast<const float*>(buf.ptr);

            auto arena = std::make_unique<pfalign::memory::GrowableArena>(100);
            pfalign::types::GuideTree tree = pfalign::tree_builders::build_nj_tree(
                dist_ptr, N, arena.get());

            return GuideTreeHolder(std::move(arena), std::move(tree));
        },
        py::arg("distances"),
        R"(
Build Neighbor-Joining guide tree.

Args:
    distances: Square distance matrix (N, N) of pairwise distances

Returns:
    GuideTree object for use with progressive_align()

Example:
    >>> distances = pfalign.compute_distances(embeddings)
    >>> tree = pfalign.tree.nj(distances)
    >>> print(tree.num_sequences())
        )");

    tree.def("bionj",
        [](py::array_t<float> distances) -> GuideTreeHolder {
            auto buf = distances.request();
            if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
                throw std::runtime_error("Distance matrix must be square (N, N)");
            }
            int N = static_cast<int>(buf.shape[0]);
            const float* dist_ptr = static_cast<const float*>(buf.ptr);

            auto arena = std::make_unique<pfalign::memory::GrowableArena>(100);
            pfalign::types::GuideTree tree = pfalign::tree_builders::build_bionj_tree(
                dist_ptr, N, arena.get());

            return GuideTreeHolder(std::move(arena), std::move(tree));
        },
        py::arg("distances"),
        R"(
Build BioNJ (an improved version of Neighbor-Joining) guide tree.

Args:
    distances: Square distance matrix (N, N) of pairwise distances

Returns:
    GuideTree object for use with progressive_align()

Example:
    >>> distances = pfalign.compute_distances(embeddings)
    >>> tree = pfalign.tree.bionj(distances)
    >>> print(tree.num_sequences())
        )");

    tree.def("mst",
        [](py::array_t<float> distances) -> GuideTreeHolder {
            auto buf = distances.request();
            if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
                throw std::runtime_error("Distance matrix must be square (N, N)");
            }
            int N = static_cast<int>(buf.shape[0]);
            const float* dist_ptr = static_cast<const float*>(buf.ptr);

            auto arena = std::make_unique<pfalign::memory::GrowableArena>(100);
            pfalign::types::GuideTree tree = pfalign::tree_builders::build_mst_tree(
                dist_ptr, N, arena.get());

            return GuideTreeHolder(std::move(arena), std::move(tree));
        },
        py::arg("distances"),
        R"(
Build Minimum Spanning Tree guide tree.

Args:
    distances: Square distance matrix (N, N) of pairwise distances

Returns:
    GuideTree object for use with progressive_align()

Example:
    >>> distances = pfalign.compute_distances(embeddings)
    >>> tree = pfalign.tree.mst(distances)
    >>> print(tree.num_sequences())
        )");

    // ----------------------- Alignment Primitives -----------------------
    py::module_ alignment = m.def_submodule("alignment", "Low-level alignment algorithms");

    alignment.def("forward_backward",
        [](py::array_t<float> similarity,
           float gap_open,
           float gap_extend,
           float temperature) -> py::tuple {

            auto buf = similarity.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Similarity matrix must be 2D");
            }

            const int L1 = static_cast<int>(buf.shape[0]);
            const int L2 = static_cast<int>(buf.shape[1]);
            const float* sim_ptr = static_cast<const float*>(buf.ptr);

            // Allocate DP matrix (3 states for affine flexible)
            std::vector<float> hij(static_cast<size_t>(L1) * L2 * 3);
            float partition = 0.0f;

            // Setup SW config
            pfalign::smith_waterman::SWConfig sw_cfg;
            sw_cfg.affine = true;
            sw_cfg.gap_open = gap_open;
            sw_cfg.gap_extend = gap_extend;
            sw_cfg.temperature = temperature;

            // Forward pass
            pfalign::smith_waterman::smith_waterman_jax_affine_flexible<pfalign::ScalarBackend>(
                sim_ptr, L1, L2, sw_cfg, hij.data(), &partition);

            // Backward pass
            std::vector<float> posteriors(static_cast<size_t>(L1) * L2);
            pfalign::memory::GrowableArena temp_arena(10);
            pfalign::smith_waterman::smith_waterman_jax_affine_flexible_backward<pfalign::ScalarBackend>(
                hij.data(), sim_ptr, L1, L2, sw_cfg, partition, posteriors.data(), &temp_arena);

            // Return posterior matrix and score
            auto post_array = py::array_t<float>({L1, L2});
            auto post_buf = post_array.request();
            float* post_ptr = static_cast<float*>(post_buf.ptr);
            std::copy(posteriors.begin(), posteriors.end(), post_ptr);

            return py::make_tuple(post_array, partition);
        },
        py::arg("similarity"),
        py::arg("gap_open") = -2.544f,
        py::arg("gap_extend") = 0.194f,
        py::arg("temperature") = 1.0f,
        R"(
Compute Smith-Waterman soft alignment (forward + backward pass).

This is the core soft alignment algorithm that computes posterior probabilities
for each position pair. The posteriors represent P(position i aligns to position j)
given the similarity matrix and gap parameters.

Args:
    similarity: Similarity matrix (L1, L2) from dot product of embeddings
    gap_open: Gap opening penalty (default: -2.544, trained model value)
    gap_extend: Gap extension penalty (default: 0.194, trained model value)
    temperature: Softmax temperature for soft alignment (default: 1.0)

Returns:
    Tuple of (posterior_matrix, score):
        posterior_matrix: (L1, L2) posterior probabilities P(i aligns to j)
        score: Log partition function (alignment quality score)

Example:
    >>> # Get embeddings
    >>> emb1 = pfalign.encode("protein1.pdb")
    >>> emb2 = pfalign.encode("protein2.pdb")
    >>>
    >>> # Compute similarity
    >>> sim = emb1.embeddings @ emb2.embeddings.T
    >>>
    >>> # Soft alignment
    >>> posterior, score = pfalign.alignment.forward_backward(sim)
    >>> print(f"Alignment score: {score:.2f}")
    >>> print(f"Max posterior: {posterior.max():.3f}")
        )");

    alignment.def("score",
        [](py::array_t<float> similarity,
           float gap_open,
           float gap_extend,
           float temperature) -> float {

            auto buf = similarity.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Similarity matrix must be 2D");
            }

            const int L1 = static_cast<int>(buf.shape[0]);
            const int L2 = static_cast<int>(buf.shape[1]);
            const float* sim_ptr = static_cast<const float*>(buf.ptr);

            // Allocate DP matrix
            std::vector<float> hij(static_cast<size_t>(L1) * L2 * 3);
            float partition = 0.0f;

            // Setup SW config
            pfalign::smith_waterman::SWConfig sw_cfg;
            sw_cfg.affine = true;
            sw_cfg.gap_open = gap_open;
            sw_cfg.gap_extend = gap_extend;
            sw_cfg.temperature = temperature;

            // Forward pass only
            pfalign::smith_waterman::smith_waterman_jax_affine_flexible<pfalign::ScalarBackend>(
                sim_ptr, L1, L2, sw_cfg, hij.data(), &partition);

            return partition;
        },
        py::arg("similarity"),
        py::arg("gap_open") = -2.544f,
        py::arg("gap_extend") = 0.194f,
        py::arg("temperature") = 1.0f,
        R"(
Compute alignment score without computing posteriors (fast).

This runs only the Smith-Waterman forward pass to get the partition function
(alignment score) without computing the posterior matrix. Useful when you only
need the score and not the full alignment.

Args:
    similarity: Similarity matrix (L1, L2)
    gap_open: Gap opening penalty (default: -2.544)
    gap_extend: Gap extension penalty (default: 0.194)
    temperature: Softmax temperature (default: 1.0)

Returns:
    Log partition function (alignment quality score)

Example:
    >>> sim = emb1.embeddings @ emb2.embeddings.T
    >>> score = pfalign.alignment.score(sim)
    >>> print(f"Alignment score: {score:.2f}")
        )");

    alignment.def("forward",
        [](py::array_t<float> similarity,
           float gap_open,
           float gap_extend,
           float temperature) -> py::array_t<float> {

            auto buf = similarity.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Similarity matrix must be 2D");
            }

            const int L1 = static_cast<int>(buf.shape[0]);
            const int L2 = static_cast<int>(buf.shape[1]);
            const float* sim_ptr = static_cast<const float*>(buf.ptr);

            // Allocate DP matrix (3 states for affine flexible)
            std::vector<float> hij(static_cast<size_t>(L1) * L2 * 3);
            float partition = 0.0f;

            // Setup SW config
            pfalign::smith_waterman::SWConfig sw_cfg;
            sw_cfg.affine = true;
            sw_cfg.gap_open = gap_open;
            sw_cfg.gap_extend = gap_extend;
            sw_cfg.temperature = temperature;

            // Forward pass
            pfalign::smith_waterman::smith_waterman_jax_affine_flexible<pfalign::ScalarBackend>(
                sim_ptr, L1, L2, sw_cfg, hij.data(), &partition);

            // Return forward scores (3D: L1 x L2 x 3)
            auto fwd_array = py::array_t<float>({L1, L2, 3});
            auto fwd_buf = fwd_array.request();
            float* fwd_ptr = static_cast<float*>(fwd_buf.ptr);
            std::copy(hij.begin(), hij.end(), fwd_ptr);

            return fwd_array;
        },
        py::arg("similarity"),
        py::arg("gap_open") = -2.544f,
        py::arg("gap_extend") = 0.194f,
        py::arg("temperature") = 1.0f,
        R"(
Smith-Waterman forward pass only.

Computes forward scores (log probabilities) for soft alignment without running
the backward pass. Returns the DP matrix with 3 states (Match, Right, Down).
Useful for research and custom alignment pipelines.

Args:
    similarity: Similarity matrix (L1, L2) from dot product of embeddings
    gap_open: Gap opening penalty (default: -2.544, trained model value)
    gap_extend: Gap extension penalty (default: 0.194, trained model value)
    temperature: Softmax temperature for soft alignment (default: 1.0)

Returns:
    Forward DP scores array (L1, L2, 3) with 3 states per position:
        - State 0 (Match): Match/mismatch state
        - State 1 (Right): Gap in sequence 2 (moving right)
        - State 2 (Down): Gap in sequence 1 (moving down)

Example:
    >>> # Get embeddings
    >>> emb1 = pfalign.encode("protein1.pdb")
    >>> emb2 = pfalign.encode("protein2.pdb")
    >>>
    >>> # Compute similarity
    >>> sim = emb1.embeddings @ emb2.embeddings.T
    >>>
    >>> # Forward pass only
    >>> fwd = pfalign.alignment.forward(sim)
    >>> print(f"Forward DP shape: {fwd.shape}")  # (L1, L2, 3)
    >>> print(f"Max forward score: {fwd.max():.2f}")
        )");

    alignment.def("backward",
        [](py::array_t<float> forward_scores,
           py::array_t<float> similarity,
           float partition,
           float gap_open,
           float gap_extend,
           float temperature) -> py::array_t<float> {

            auto fwd_buf = forward_scores.request();
            auto sim_buf = similarity.request();

            if (fwd_buf.ndim != 3 || fwd_buf.shape[2] != 3) {
                throw std::runtime_error("Forward scores must be 3D array (L1, L2, 3)");
            }
            if (sim_buf.ndim != 2) {
                throw std::runtime_error("Similarity matrix must be 2D");
            }

            const int L1 = static_cast<int>(fwd_buf.shape[0]);
            const int L2 = static_cast<int>(fwd_buf.shape[1]);

            if (sim_buf.shape[0] != L1 || sim_buf.shape[1] != L2) {
                throw std::runtime_error("Forward scores and similarity matrix shape mismatch");
            }

            const float* fwd_ptr = static_cast<const float*>(fwd_buf.ptr);
            const float* sim_ptr = static_cast<const float*>(sim_buf.ptr);

            // Setup SW config
            pfalign::smith_waterman::SWConfig sw_cfg;
            sw_cfg.affine = true;
            sw_cfg.gap_open = gap_open;
            sw_cfg.gap_extend = gap_extend;
            sw_cfg.temperature = temperature;

            // Backward pass
            std::vector<float> posteriors(static_cast<size_t>(L1) * L2);
            pfalign::memory::GrowableArena temp_arena(10);
            pfalign::smith_waterman::smith_waterman_jax_affine_flexible_backward<pfalign::ScalarBackend>(
                fwd_ptr, sim_ptr, L1, L2, sw_cfg, partition, posteriors.data(), &temp_arena);

            // Return posterior matrix
            auto post_array = py::array_t<float>({L1, L2});
            auto post_buf = post_array.request();
            float* post_ptr = static_cast<float*>(post_buf.ptr);
            std::copy(posteriors.begin(), posteriors.end(), post_ptr);

            return post_array;
        },
        py::arg("forward_scores"),
        py::arg("similarity"),
        py::arg("partition"),
        py::arg("gap_open") = -2.544f,
        py::arg("gap_extend") = 0.194f,
        py::arg("temperature") = 1.0f,
        R"(
Smith-Waterman backward pass only.

Computes posterior probabilities from forward scores. This is the gradient
computation that converts forward DP values into alignment posteriors.
Useful for research and custom alignment pipelines.

Args:
    forward_scores: Forward DP scores (L1, L2, 3) from forward()
    similarity: Original similarity matrix (L1, L2)
    partition: Partition function (scalar) from forward pass
    gap_open: Gap opening penalty (must match forward pass)
    gap_extend: Gap extension penalty (must match forward pass)
    temperature: Softmax temperature (must match forward pass)

Returns:
    Posterior probability matrix (L1, L2) where each entry is P(i aligns to j)

Example:
    >>> # Forward pass
    >>> fwd = pfalign.alignment.forward(sim)
    >>> partition = pfalign.alignment.score(sim)  # or compute from fwd
    >>>
    >>> # Backward pass
    >>> posterior = pfalign.alignment.backward(fwd, sim, partition)
    >>> print(f"Posterior shape: {posterior.shape}")  # (L1, L2)
    >>> print(f"Sum of posteriors: {posterior.sum():.3f}")  # Should be ~1.0

Note:
    For most use cases, use forward_backward() instead which combines both passes.
    This separate backward() function is for research and experimentation.
        )");

    alignment.def("viterbi_decode",
        [](py::array_t<float> posterior,
           const std::string& seq1,
           const std::string& seq2,
           float gap_penalty) -> py::tuple {

            auto buf = posterior.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Posterior matrix must be 2D");
            }

            const int L1 = static_cast<int>(buf.shape[0]);
            const int L2 = static_cast<int>(buf.shape[1]);
            const float* post_ptr = static_cast<const float*>(buf.ptr);

            // Validate sequence lengths
            if (static_cast<int>(seq1.size()) != L1) {
                throw std::runtime_error("Sequence 1 length must match posterior rows");
            }
            if (static_cast<int>(seq2.size()) != L2) {
                throw std::runtime_error("Sequence 2 length must match posterior columns");
            }

            // Allocate scratch buffers for Viterbi DP
            std::vector<float> dp_score(static_cast<size_t>(L1 + 1) * (L2 + 1));
            std::vector<uint8_t> dp_traceback(static_cast<size_t>(L1 + 1) * (L2 + 1));

            // Allocate output path
            const int max_path_length = L1 + L2;
            std::vector<pfalign::AlignmentPair> alignment_path(max_path_length);

            // Decode alignment
            int path_len = pfalign::alignment_decode::decode_alignment<pfalign::ScalarBackend>(
                post_ptr, L1, L2, gap_penalty,
                alignment_path.data(), max_path_length,
                dp_score.data(), dp_traceback.data());

            if (path_len < 0) {
                throw std::runtime_error("Viterbi decode failed (buffer overflow or invalid input)");
            }

            // Insert gaps to create aligned sequences
            std::string aligned_seq1 = pfalign::io::insert_gaps(seq1, alignment_path.data(), path_len, true);
            std::string aligned_seq2 = pfalign::io::insert_gaps(seq2, alignment_path.data(), path_len, false);

            // Convert path to Python list of tuples
            py::list path_list;
            for (int k = 0; k < path_len; k++) {
                path_list.append(py::make_tuple(
                    alignment_path[k].i,
                    alignment_path[k].j
                ));
            }

            return py::make_tuple(aligned_seq1, aligned_seq2, path_list);
        },
        py::arg("posterior"),
        py::arg("seq1"),
        py::arg("seq2"),
        py::arg("gap_penalty") = -5.0f,
        R"(
Viterbi decode: extract discrete alignment with sequences from posterior.

Decodes hard alignment from soft posterior matrix and returns aligned sequences
with gaps ('-') inserted. This is the full decoding function that returns both
gapped sequences and the alignment path.

Args:
    posterior: Soft alignment probabilities (L1, L2) from forward_backward()
    seq1: First sequence string (length must match L1)
    seq2: Second sequence string (length must match L2)
    gap_penalty: Log-probability penalty for gaps (default: -5.0)
                 More negative = fewer gaps, less negative = more gaps

Returns:
    Tuple of (aligned_seq1, aligned_seq2, alignment_path):
        - aligned_seq1: First sequence with gaps ('-') inserted
        - aligned_seq2: Second sequence with gaps ('-') inserted
        - alignment_path: List of (i, j) coordinate pairs
          Gaps are represented as (-1, j) for gap in seq1 or (i, -1) for gap in seq2

Example:
    >>> # Get embeddings and sequences
    >>> result1 = pfalign.encode("protein1.pdb")
    >>> result2 = pfalign.encode("protein2.pdb")
    >>> seq1 = result1.sequence
    >>> seq2 = result2.sequence
    >>>
    >>> # Compute similarity and soft alignment
    >>> sim = result1.embeddings @ result2.embeddings.T
    >>> posterior, score = pfalign.alignment.forward_backward(sim)
    >>>
    >>> # Hard alignment decoding
    >>> aligned1, aligned2, path = pfalign.alignment.viterbi_decode(
    ...     posterior, seq1, seq2, gap_penalty=-5.0
    ... )
    >>> print(f"Alignment length: {len(aligned1)}")
    >>> print(aligned1)
    'ACDEFG-HIKLM'
    >>> print(aligned2)
    'ACDE-GPHIKLM'
    >>>
    >>> # Try different gap penalty
    >>> aligned1_relaxed, aligned2_relaxed, _ = pfalign.alignment.viterbi_decode(
    ...     posterior, seq1, seq2, gap_penalty=-1.0
    ... )

Note:
    - Use viterbi_path() if you only need coordinates without sequences
    - Gap penalty controls alignment stringency (typical range: -10 to -1)
    - Sequences must match the posterior matrix dimensions
        )");

    alignment.def("viterbi_path",
        [](py::array_t<float> posterior,
           float gap_penalty) -> py::list {

            auto buf = posterior.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Posterior matrix must be 2D");
            }

            const int L1 = static_cast<int>(buf.shape[0]);
            const int L2 = static_cast<int>(buf.shape[1]);
            const float* post_ptr = static_cast<const float*>(buf.ptr);

            // Allocate scratch buffers for Viterbi DP
            std::vector<float> dp_score(static_cast<size_t>(L1 + 1) * (L2 + 1));
            std::vector<uint8_t> dp_traceback(static_cast<size_t>(L1 + 1) * (L2 + 1));

            // Allocate output path
            const int max_path_length = L1 + L2;
            std::vector<pfalign::AlignmentPair> alignment_path(max_path_length);

            // Decode alignment
            int path_len = pfalign::alignment_decode::decode_alignment<pfalign::ScalarBackend>(
                post_ptr, L1, L2, gap_penalty,
                alignment_path.data(), max_path_length,
                dp_score.data(), dp_traceback.data());

            if (path_len < 0) {
                throw std::runtime_error("Viterbi decode failed (buffer overflow or invalid input)");
            }

            // Convert to Python list of tuples
            py::list result;
            for (int k = 0; k < path_len; k++) {
                result.append(py::make_tuple(
                    alignment_path[k].i,
                    alignment_path[k].j
                ));
            }

            return result;
        },
        py::arg("posterior"),
        py::arg("gap_penalty") = -5.0f,
        R"(
Decode hard alignment path from posterior matrix using Viterbi algorithm.

Uses Maximum A Posteriori (MAP) decoding to extract the single most likely
alignment path from soft alignment posteriors. Returns only the coordinates;
gaps are represented as -1.

Args:
    posterior: Posterior probability matrix (L1, L2) from forward_backward()
    gap_penalty: Log-probability penalty for gaps (default: -5.0)
                 More negative = fewer gaps, less negative = more gaps

Returns:
    List of (i, j) tuples representing the alignment path.
    Gaps are represented as (-1, j) for gap in seq1 or (i, -1) for gap in seq2.

Example:
    >>> # Soft alignment
    >>> posterior, score = pfalign.alignment.forward_backward(sim)
    >>>
    >>> # Hard decode with strict gap penalty
    >>> path = pfalign.alignment.viterbi_path(posterior, gap_penalty=-10.0)
    >>> print(f"Alignment has {len(path)} positions")
    >>>
    >>> # Count gaps
    >>> gaps_seq1 = sum(1 for i, j in path if i == -1)
    >>> gaps_seq2 = sum(1 for i, j in path if j == -1)
    >>> print(f"Gaps in seq1: {gaps_seq1}, Gaps in seq2: {gaps_seq2}")
    >>>
    >>> # Try relaxed gap penalty for comparison
    >>> path_relaxed = pfalign.alignment.viterbi_path(posterior, gap_penalty=-1.0)
        )");

    // ----------------------- Result Classes -----------------------
    py::class_<PairwiseResult>(m, "PairwiseResult")
        .def_property_readonly("L1", &PairwiseResult::L1)
        .def_property_readonly("L2", &PairwiseResult::L2)
        .def_property_readonly("score", &PairwiseResult::score)
        .def_property_readonly("partition", &PairwiseResult::partition)
        .def_property_readonly("shape", [](const PairwiseResult& self) {
            return py::make_tuple(self.L1(), self.L2());
        }, "Shape of alignment (L1, L2) - NumPy-style")
        .def_property_readonly("coverage", &PairwiseResult::compute_coverage,
            "Alignment coverage (convenience property)")
        .def("alignment", &PairwiseResult::alignment, py::return_value_policy::reference_internal)
        .def_property_readonly("posteriors", [](py::object self_py) {
            auto& self = self_py.cast<PairwiseResult&>();
            return py::array_t<float>(
                {self.L1(), self.L2()},
                self.posteriors(),
                self_py);
        })
        .def("write_fasta",
             &PairwiseResult::write_fasta,
             py::arg("path"),
             py::arg("seq1"),
             py::arg("seq2"),
             py::arg("id1") = "seq1",
             py::arg("id2") = "seq2")
        .def("write_npy", &PairwiseResult::write_npy)
        .def("write_pdb", [](const PairwiseResult& self,
                             const std::string& path,
                             py::array_t<float, py::array::c_style | py::array::forcecast> coords1,
                             py::array_t<float, py::array::c_style | py::array::forcecast> coords2,
                             int reference) {
            self.write_pdb(path,
                           coords1.data(),
                           static_cast<int>(coords1.shape(0)),
                           coords2.data(),
                           static_cast<int>(coords2.shape(0)),
                           reference);
        }, py::arg("path"), py::arg("coords1"), py::arg("coords2"), py::arg("reference") = 0)
        .def("get_aligned_residues", &PairwiseResult::get_aligned_residues)
        .def("compute_coverage", &PairwiseResult::compute_coverage)
        .def("threshold", &PairwiseResult::threshold, py::arg("cutoff"))
        .def("get_aligned_coords", [](const PairwiseResult& self,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> coords1,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> coords2,
                                      int reference) {
            // Count aligned residues (pairs where both indices >= 0)
            int num_aligned = 0;
            for (const auto& pair : self.alignment()) {
                if (pair.first >= 0 && pair.second >= 0) {
                    ++num_aligned;
                }
            }

            // Allocate output arrays sized for aligned residues only
            auto out1 = py::array_t<float>({num_aligned, 3});
            auto out2 = py::array_t<float>({num_aligned, 3});
            self.get_aligned_coords(coords1.data(),
                                    coords2.data(),
                                    out1.mutable_data(),
                                    out2.mutable_data(),
                                    reference);
            return py::make_tuple(out1, out2);
        }, py::arg("coords1"), py::arg("coords2"), py::arg("reference") = 0)
        .def("compute_rmsd", [](const PairwiseResult& self,
                                py::array_t<float, py::array::c_style | py::array::forcecast> coords1,
                                py::array_t<float, py::array::c_style | py::array::forcecast> coords2) {
            return self.compute_rmsd(coords1.data(), coords2.data());
        })
        .def("compute_tm_score", [](const PairwiseResult& self,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> coords1,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> coords2) {
            return self.compute_tm_score(coords1.data(), coords2.data());
        })
        // New format conversion methods
        .def("save", [](const PairwiseResult& self,
                        const std::string& path,
                        const std::string& format) {
            // Create alignment
            pfalign::io::MultipleAlignment aln;
            aln.names = {"seq1", "seq2"};

            // Get aligned sequences from result
            std::string seq1, seq2;
            const auto& alignment = self.alignment();
            for (const auto& pair : alignment) {
                if (pair.first >= 0) seq1 += 'X';  // Placeholder - will need actual sequence
                else seq1 += '-';
                if (pair.second >= 0) seq2 += 'X';
                else seq2 += '-';
            }
            aln.sequences = {seq1, seq2};

            // Detect or parse format
            auto fmt = format.empty()
                ? pfalign::io::AlignmentFormatParser::detect_format(path)
                : pfalign::io::parse_format(format);

            // Write
            pfalign::io::AlignmentFormatWriter::write_file(aln, path, fmt);
        },
        py::arg("path"),
        py::arg("format") = "",
        "Save alignment to file (format auto-detected from extension)")
        .def("statistics", [](const PairwiseResult& self) {
            // Compute alignment statistics
            std::string seq1, seq2;
            const auto& alignment = self.alignment();
            for (const auto& pair : alignment) {
                seq1 += (pair.first >= 0) ? 'X' : '-';
                seq2 += (pair.second >= 0) ? 'X' : '-';
            }

            auto stats = pfalign::io::compute_alignment_stats(seq1, seq2);

            py::dict result;
            result["identity"] = stats.identity;
            result["similarity"] = stats.similarity;
            result["coverage"] = stats.coverage;
            result["gaps"] = stats.gaps;
            result["gap_percentage"] = stats.gap_percentage;
            result["score"] = self.score();
            return result;
        },
        "Get alignment statistics as dictionary")
        .def("summary", [](const PairwiseResult& self) {
            std::string seq1, seq2;
            const auto& alignment = self.alignment();
            for (const auto& pair : alignment) {
                seq1 += (pair.first >= 0) ? 'X' : '-';
                seq2 += (pair.second >= 0) ? 'X' : '-';
            }

            auto stats = pfalign::io::compute_alignment_stats(seq1, seq2);

            std::ostringstream ss;
            ss << "Alignment Summary:\n"
               << "  Score: " << self.score() << "\n"
               << "  Identity: " << (stats.identity * 100.0) << "%\n"
               << "  Coverage: " << (stats.coverage * 100.0) << "%\n"
               << "  Gaps: " << stats.gaps << " ("
               << (stats.gap_percentage * 100.0) << "%)\n";
            return ss.str();
        },
        "Get formatted alignment summary")
        .def("__repr__", [](const PairwiseResult& self) {
            std::ostringstream ss;
            ss << "<PairwiseResult: shape=(" << self.L1() << ", " << self.L2()
               << "), score=" << std::fixed << std::setprecision(2) << self.score() << ">";
            return ss.str();
        })
        .def("__str__", [](const PairwiseResult& self) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "Pairwise Alignment Result\n"
               << "  Shape: (" << self.L1() << ", " << self.L2() << ")\n"
               << "  Score: " << self.score() << "\n"
               << "  Coverage: " << self.compute_coverage();
            return ss.str();
        });

    py::class_<MSAResult>(m, "MSAResult")
        .def_property_readonly("num_sequences", &MSAResult::num_sequences)
        .def_property_readonly("alignment_length", &MSAResult::alignment_length)
        .def_property_readonly("ecs_score", &MSAResult::ecs_score)
        .def_property_readonly("conservation", [](const MSAResult& self) {
            auto arr = py::array_t<float>(self.alignment_length());
            self.compute_conservation(arr.mutable_data());
            return arr;
        }, "Per-column conservation scores (convenience property)")
        .def_property_readonly("consensus", [](const MSAResult& self) {
            return self.get_consensus(0.5f);
        }, "Consensus sequence with threshold=0.5 (convenience property)")
        .def("sequences", &MSAResult::sequences, py::return_value_policy::reference_internal)
        .def("identifiers", &MSAResult::identifiers, py::return_value_policy::reference_internal)
        .def("write_fasta", &MSAResult::write_fasta)
        .def("write_pdb", [](const MSAResult& self,
                             const std::string& path,
                             const std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>>& coords) {
            std::vector<const float*> ptrs;
            ptrs.reserve(coords.size());
            for (const auto& arr : coords) {
                ptrs.push_back(arr.data());
            }
            self.write_pdb(path, ptrs.data(), static_cast<int>(ptrs.size()));
        })
        .def("to_array", [](const MSAResult& self) {
            auto arr = py::array_t<int>({self.alignment_length(), self.num_sequences()});
            self.to_array(arr.mutable_data());
            return arr;
        })
        .def("get_sequence", &MSAResult::get_sequence, py::arg("index"))
        .def("get_column", &MSAResult::get_column, py::arg("index"))
        .def("get_consensus", &MSAResult::get_consensus, py::arg("threshold") = 0.5f)
        .def("compute_conservation", [](const MSAResult& self) {
            auto arr = py::array_t<float>(self.alignment_length());
            self.compute_conservation(arr.mutable_data());
            return arr;
        })
        .def("filter_gaps", &MSAResult::filter_gaps, py::arg("threshold") = 0.5f)
        .def("compute_pairwise_identity", [](const MSAResult& self) {
            auto arr = py::array_t<float>({self.num_sequences(), self.num_sequences()});
            self.compute_pairwise_identity(arr.mutable_data());
            return arr;
        })
        // New format conversion and utility methods
        .def("save", [](const MSAResult& self,
                        const std::string& path,
                        const std::string& format) {
            // Create alignment
            pfalign::io::MultipleAlignment aln;
            const auto& sequences = self.sequences();
            const auto& identifiers = self.identifiers();

            aln.names = identifiers;
            aln.sequences = sequences;

            // Detect or parse format
            auto fmt = format.empty()
                ? pfalign::io::AlignmentFormatParser::detect_format(path)
                : pfalign::io::parse_format(format);

            // Write
            pfalign::io::AlignmentFormatWriter::write_file(aln, path, fmt);
        },
        py::arg("path"),
        py::arg("format") = "",
        "Save MSA to file (format auto-detected from extension)")
        .def("__getitem__", [](const MSAResult& self, int index) {
            if (index < 0 || index >= static_cast<int>(self.num_sequences())) {
                throw py::index_error("Index out of range");
            }
            return self.get_sequence(index);
        },
        "Get sequence by index")
        .def("__len__", &MSAResult::num_sequences, "Get number of sequences")
        // Note: "consensus" and "conservation" already defined as properties above (lines 2629, 2624)
        .def("statistics", [](const MSAResult& self) {
            py::dict stats;

            // Basic MSA properties
            stats["num_sequences"] = self.num_sequences();
            stats["alignment_length"] = self.alignment_length();
            stats["ecs_score"] = self.ecs_score();

            int num_seqs = static_cast<int>(self.num_sequences());
            int aln_len = static_cast<int>(self.alignment_length());

            if (num_seqs < 2) {
                stats["mean_pairwise_identity"] = 0.0;
                stats["mean_conservation"] = 0.0;
                stats["gap_percentage"] = 0.0;
                return stats;
            }

            // Calculate mean pairwise identity
            const auto& sequences = self.sequences();
            double total_identity = 0.0;
            int num_pairs = 0;

            for (int i = 0; i < num_seqs; i++) {
                for (int j = i + 1; j < num_seqs; j++) {
                    auto pair_stats = pfalign::io::compute_alignment_stats(
                        sequences[i], sequences[j]
                    );
                    total_identity += pair_stats.identity;
                    num_pairs++;
                }
            }

            stats["mean_pairwise_identity"] = num_pairs > 0
                ? total_identity / num_pairs
                : 0.0;

            // Calculate mean conservation
            auto conservation_scores = py::array_t<float>(aln_len);
            self.compute_conservation(conservation_scores.mutable_data());

            float* conservation_data = conservation_scores.mutable_data();
            double total_conservation = 0.0;
            for (int i = 0; i < aln_len; i++) {
                total_conservation += conservation_data[i];
            }
            stats["mean_conservation"] = total_conservation / aln_len;

            // Calculate gap percentage
            int total_gaps = 0;
            int total_positions = num_seqs * aln_len;

            for (const auto& seq : sequences) {
                for (char c : seq) {
                    if (c == '-') total_gaps++;
                }
            }

            stats["gap_percentage"] = static_cast<double>(total_gaps) / total_positions;

            return stats;
        },
        "Compute MSA statistics as dictionary")
        .def("summary", [](const MSAResult& self) {
            int num_seqs = static_cast<int>(self.num_sequences());
            int aln_len = static_cast<int>(self.alignment_length());

            if (num_seqs < 2) {
                std::ostringstream ss;
                ss << "MSA Summary:\n"
                   << "  Sequences: " << num_seqs << "\n"
                   << "  Alignment Length: " << aln_len << "\n"
                   << "  (Statistics require at least 2 sequences)\n";
                return ss.str();
            }

            const auto& sequences = self.sequences();

            // Calculate mean pairwise identity
            double total_identity = 0.0;
            int num_pairs = 0;

            for (int i = 0; i < num_seqs; i++) {
                for (int j = i + 1; j < num_seqs; j++) {
                    auto pair_stats = pfalign::io::compute_alignment_stats(
                        sequences[i], sequences[j]
                    );
                    total_identity += pair_stats.identity;
                    num_pairs++;
                }
            }

            double mean_identity = num_pairs > 0 ? total_identity / num_pairs : 0.0;

            // Calculate mean conservation
            auto conservation_scores = py::array_t<float>(aln_len);
            self.compute_conservation(conservation_scores.mutable_data());

            float* conservation_data = conservation_scores.mutable_data();
            double total_conservation = 0.0;
            for (int i = 0; i < aln_len; i++) {
                total_conservation += conservation_data[i];
            }
            double mean_conservation = total_conservation / aln_len;

            // Calculate gap percentage
            int total_gaps = 0;
            int total_positions = num_seqs * aln_len;

            for (const auto& seq : sequences) {
                for (char c : seq) {
                    if (c == '-') total_gaps++;
                }
            }

            double gap_percentage = static_cast<double>(total_gaps) / total_positions;

            // Format output
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1);
            ss << "MSA Summary:\n"
               << "  Sequences: " << num_seqs << "\n"
               << "  Alignment Length: " << aln_len << "\n"
               << std::setprecision(4)
               << "  ECS Score: " << self.ecs_score() << "\n"
               << std::setprecision(1)
               << "  Mean Pairwise Identity: " << (mean_identity * 100.0) << "%\n"
               << std::setprecision(3)
               << "  Mean Conservation: " << mean_conservation << "\n"
               << std::setprecision(1)
               << "  Gap Content: " << (gap_percentage * 100.0) << "%\n";

            return ss.str();
        },
        "Get formatted MSA summary")
        .def("__repr__", [](const MSAResult& self) {
            std::ostringstream ss;
            ss << "<MSAResult: sequences=" << self.num_sequences()
               << ", length=" << self.alignment_length()
               << ", ecs=" << std::fixed << std::setprecision(2) << self.ecs_score() << ">";
            return ss.str();
        })
        .def("__str__", [](const MSAResult& self) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4);
            ss << "MSA Result\n"
               << "  Sequences: " << self.num_sequences() << "\n"
               << "  Length: " << self.alignment_length() << "\n"
               << "  ECS Score: " << self.ecs_score();
            return ss.str();
        });

    py::class_<EmbeddingResult>(m, "EmbeddingResult")
        .def_property_readonly("sequence_length", &EmbeddingResult::sequence_length)
        .def_property_readonly("hidden_dim", &EmbeddingResult::hidden_dim)
        .def_property_readonly("embeddings", [](py::object self_py) {
            auto& self = self_py.cast<EmbeddingResult&>();
            return py::array_t<float>(
                {self.sequence_length(), self.hidden_dim()},
                self.embeddings(),
                self_py);
        })
        .def("save", &EmbeddingResult::save)
        .def("__array__", [](py::object self_py) {
            auto& self = self_py.cast<EmbeddingResult&>();
            return py::array_t<float>(
                {self.sequence_length(), self.hidden_dim()},
                self.embeddings(),
                self_py);
        })
        .def("compute_pairwise_distances", [](const EmbeddingResult& self) {
            auto arr = py::array_t<float>({self.sequence_length(), self.sequence_length()});
            self.compute_pairwise_distances(arr.mutable_data());
            return arr;
        })
        .def("get_subset", &EmbeddingResult::get_subset, py::arg("indices"))
        .def("normalize", &EmbeddingResult::normalize)
        .def("__repr__", [](const EmbeddingResult& self) {
            std::ostringstream ss;
            ss << "<EmbeddingResult: shape=(" << self.sequence_length()
               << ", " << self.hidden_dim() << ")>";
            return ss.str();
        });

    py::class_<SimilarityResult>(m, "SimilarityResult")
        .def_property_readonly("L1", &SimilarityResult::L1)
        .def_property_readonly("L2", &SimilarityResult::L2)
        .def_property_readonly("shape", &SimilarityResult::shape)
        .def_property_readonly("similarity", [](py::object self_py) {
            auto& self = self_py.cast<SimilarityResult&>();
            return py::array_t<float>(
                {self.L1(), self.L2()},
                self.similarity(),
                self_py);
        })
        .def("save", &SimilarityResult::save)
        .def("__array__", [](py::object self_py) {
            auto& self = self_py.cast<SimilarityResult&>();
            return py::array_t<float>(
                {self.L1(), self.L2()},
                self.similarity(),
                self_py);
        })
        .def("get_top_k", &SimilarityResult::get_top_k, py::arg("k"))
        .def("threshold", &SimilarityResult::threshold, py::arg("cutoff"))
        .def("normalize", &SimilarityResult::normalize)
        .def("__repr__", [](const SimilarityResult& self) {
            std::ostringstream ss;
            ss << "<SimilarityResult: shape=(" << self.L1()
               << ", " << self.L2() << ")>";
            return ss.str();
        });

    // ----------------------- Format Conversion -----------------------
    m.def("_reformat",
          [](const std::string& infile,
             const std::string& outfile,
             const std::string& inform,
             const std::string& outform,
             const std::string& match_mode,
             int gap_threshold,
             bool remove_inserts,
             int remove_gapped,
             bool uppercase,
             bool lowercase,
             bool remove_ss) {
              // Parse input
              auto in_fmt = inform.empty()
                  ? pfalign::io::AlignmentFormatParser::detect_format(infile)
                  : pfalign::io::parse_format(inform);
              auto aln = pfalign::io::AlignmentFormatParser::parse_file(infile, in_fmt);

              // Setup writer options
              pfalign::io::AlignmentFormatWriter::Options opts;
              if (match_mode == "first") {
                  opts.match_mode = pfalign::io::AlignmentFormatWriter::Options::MatchMode::FIRST_SEQUENCE;
              } else if (match_mode == "gap") {
                  opts.match_mode = pfalign::io::AlignmentFormatWriter::Options::MatchMode::GAP_THRESHOLD;
                  opts.gap_threshold_percent = gap_threshold;
              }
              opts.remove_inserts = remove_inserts;
              opts.remove_gapped_threshold = remove_gapped;
              opts.uppercase = uppercase;
              opts.lowercase = lowercase;
              opts.remove_secondary_structure = remove_ss;

              // Parse output format
              auto out_fmt = outform.empty()
                  ? pfalign::io::AlignmentFormatParser::detect_format(outfile)
                  : pfalign::io::parse_format(outform);

              // Write
              pfalign::io::AlignmentFormatWriter::write_file(aln, outfile, out_fmt, opts);
          },
          py::arg("infile"),
          py::arg("outfile"),
          py::arg("inform") = "",
          py::arg("outform") = "",
          py::arg("match_mode") = "",
          py::arg("gap_threshold") = 0,
          py::arg("remove_inserts") = false,
          py::arg("remove_gapped") = 0,
          py::arg("uppercase") = false,
          py::arg("lowercase") = false,
          py::arg("remove_ss") = false,
          "Convert alignment between formats");

    // ----------------------- Helper Functions -----------------------

    // Get structure information
    m.def("_get_structure_info",
          [](const std::string& path, bool show_chains) {
              // Load structure
              auto protein = pfalign::commands::LoadStructureFile(path);

              // Detect format (C++17 compatible suffix check)
              std::string format = "pdb";
              if (path.size() >= 4 && path.substr(path.size() - 4) == ".cif") {
                  format = "cif";
              } else if (path.size() >= 6 && path.substr(path.size() - 6) == ".mmcif") {
                  format = "cif";
              }

              // Build result dictionary
              py::dict result;
              result["path"] = path;
              result["format"] = format;
              result["num_chains"] = protein.num_chains();

              if (show_chains) {
                  py::list chains;
                  for (size_t i = 0; i < protein.num_chains(); ++i) {
                      const auto& chain = protein.get_chain(i);
                      py::dict chain_info;
                      chain_info["index"] = py::int_(i);
                      chain_info["id"] = std::string(1, chain.chain_id);
                      chain_info["length"] = py::int_(chain.size());

                      size_t atom_count = 0;
                      for (const auto& residue : chain.residues) {
                          atom_count += residue.atoms.size();
                      }
                      chain_info["num_atoms"] = py::int_(atom_count);

                      chains.append(chain_info);
                  }
                  result["chains"] = chains;
              }

              return result;
          },
          py::arg("path"),
          py::arg("show_chains") = false,
          "Get structure file information");

    // ----------------------- Progress Bar -----------------------
    // APT-style progress bar for CLI and API use
    py::class_<pfalign::common::ProgressBar>(m, "ProgressBar",
        "APT-style progress bar for tracking long-running operations.\n\n"
        "Example:\n"
        "    >>> bar = ProgressBar(100, 'Processing files')\n"
        "    >>> for i in range(100):\n"
        "    ...     bar.update(i + 1)\n"
        "    >>> bar.finish()\n"
        "    [####################] 100% (100/100) Processing files")
        .def(py::init<int, const std::string&, int, bool, bool>(),
             py::arg("total"),
             py::arg("description") = "",
             py::arg("width") = 20,
             py::arg("show_percent") = true,
             py::arg("show_count") = true,
             "Create a progress bar.\n\n"
             "Args:\n"
             "    total: Total number of items to process\n"
             "    description: Optional description shown after percentage\n"
             "    width: Width of the progress bar in characters (default: 20)\n"
             "    show_percent: Show percentage (default: True)\n"
             "    show_count: Show count like (20/50) (default: True)")
        .def("update", &pfalign::common::ProgressBar::update,
             py::arg("current"),
             "Update progress to current count.\n\n"
             "Args:\n"
             "    current: Current progress count")
        .def("tick",
             py::overload_cast<>(&pfalign::common::ProgressBar::tick),
             "Increment progress by 1")
        .def("tick",
             py::overload_cast<int>(&pfalign::common::ProgressBar::tick),
             py::arg("delta"),
             "Increment progress by specified amount.\n\n"
             "Args:\n"
             "    delta: Amount to increment")
        .def("finish", &pfalign::common::ProgressBar::finish,
             "Mark progress as complete and clear the bar")
        .def("current", &pfalign::common::ProgressBar::current,
             "Get current progress (0 to total)")
        .def("total", &pfalign::common::ProgressBar::total,
             "Get total count")
        .def("fraction", &pfalign::common::ProgressBar::fraction,
             "Get completion fraction (0.0 to 1.0)")
        .def("is_finished", &pfalign::common::ProgressBar::is_finished,
             "Check if progress bar is finished")
        .def("set_description", &pfalign::common::ProgressBar::set_description,
             py::arg("description"),
             "Update the description text displayed in the progress bar")
        .def("reset", &pfalign::common::ProgressBar::reset,
             py::arg("new_total"),
             py::arg("description"),
             "Reset progress for a new phase with different total")
        .def("__enter__", [](pfalign::common::ProgressBar& self) -> pfalign::common::ProgressBar& {
            return self;
        }, py::return_value_policy::reference_internal,
           "Enter context manager")
        .def("__exit__", [](pfalign::common::ProgressBar& self, py::object, py::object, py::object) {
            if (!self.is_finished()) {
                self.finish();
            }
        }, "Exit context manager and auto-finish");

    // Get version information
    m.def("_version_info",
          []() {
              py::dict info;

              // Git information (would be populated at build time)
              #ifdef GIT_COMMIT_HASH
              info["git_commit"] = GIT_COMMIT_HASH;
              #else
              info["git_commit"] = "unknown";
              #endif

              #ifdef GIT_BRANCH
              info["git_branch"] = GIT_BRANCH;
              #else
              info["git_branch"] = "unknown";
              #endif

              // Build type
              #ifdef NDEBUG
              info["build_type"] = "Release";
              #else
              info["build_type"] = "Debug";
              #endif

              // Compiler info
              #if defined(__clang__)
              info["compiler"] = "Clang " + std::string(__clang_version__);
              #elif defined(__GNUC__)
              info["compiler"] = "GCC " + std::to_string(__GNUC__) + "." +
                                std::to_string(__GNUC_MINOR__) + "." +
                                std::to_string(__GNUC_PATCHLEVEL__);
              #elif defined(_MSC_VER)
              info["compiler"] = "MSVC " + std::to_string(_MSC_VER);
              #else
              info["compiler"] = "unknown";
              #endif

              // Build date/time
              info["build_date"] = std::string(__DATE__) + " " + std::string(__TIME__);

              return info;
          },
          "Get version and build information");
}
