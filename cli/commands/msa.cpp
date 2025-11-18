#include "commands/commands.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <thread>
#include <limits>

// Foundation layer
#include "pfalign/common/growable_arena.h"
#include "pfalign/common/memory_profiler.h"
#include "pfalign/common/thread_pool.h"
#include "pfalign/dispatch/backend_traits.h"

// Module layer
#include "pfalign/modules/mpnn/mpnn_cache_adapter.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/modules/msa/profile.h"

// Types layer
#include "pfalign/types/guide_tree_types.h"

// Algorithm layer
#include "pfalign/algorithms/distance_metrics/distance_matrix.h"
#include "pfalign/algorithms/progressive_msa/progressive_msa.h"
#include "pfalign/algorithms/tree_builders/builders.h"

// I/O layer
#include "pfalign/io/protein_structure.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"

// CLI utilities
#include "commands/input_utils.h"

// Error handling
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"

namespace pfalign {
namespace commands {

namespace {

std::string Trim(const std::string& value) {
    size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> ParseFileList(const std::string& list_path) {
    std::ifstream file(list_path);
    if (!file) {
        throw errors::FileNotFoundError(list_path, "input list file");
    }

    std::filesystem::path base_dir = std::filesystem::absolute(list_path).parent_path();
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(file, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::filesystem::path candidate(line);
        if (!candidate.is_absolute()) {
            candidate = base_dir / candidate;
        }
        paths.push_back(candidate.string());
    }
    return paths;
}

std::vector<std::string> ScanDirectory(const std::string& dir_path) {
    std::filesystem::path dir(dir_path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        throw errors::FileNotFoundError(dir_path, "directory");
    }

    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file())
            continue;
        const auto ext = ToLower(entry.path().extension().string());
        if (ext == ".pdb" || ext == ".cif" || ext == ".mmcif" || ext == ".ent" || ext == ".npy") {
            paths.push_back(entry.path().string());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

std::vector<std::string> ResolveInputs(const std::vector<std::string>& inputs,
                                       const std::string& input_list,
                                       const std::string& input_dir) {
    int specified = 0;
    if (!inputs.empty())
        specified++;
    if (!input_list.empty())
        specified++;
    if (!input_dir.empty())
        specified++;

    if (specified == 0) {
        throw errors::ValidationError(
            "No inputs provided",
            "Provide one of: positional inputs, --input-list, or --input-dir");
    }
    if (specified > 1) {
        throw errors::ValidationError(
            "Multiple input methods specified",
            "Specify only one of: positional inputs, --input-list, or --input-dir");
    }

    if (!inputs.empty()) {
        return inputs;
    }
    if (!input_list.empty()) {
        return ParseFileList(input_list);
    }
    return ScanDirectory(input_dir);
}

std::string MakeIdentifier(const std::string& path, char chain_id = '\0') {
    std::filesystem::path p(path);
    std::string stem = p.stem().string();
    if (chain_id != '\0') {
        stem += "_";
        stem.push_back(chain_id);
    }
    return stem;
}

void LoadStructureSequence(const std::string& path, io::Protein* protein,
                           std::vector<float>* coords_out, std::string* sequence_out,
                           int* length_out) {
    *protein = LoadStructureFile(path);
    if (protein->num_chains() == 0) {
        throw errors::messages::no_chains_in_structure(path);
    }

    // Default to first chain
    int chain_idx = 0;
    const auto& chain = protein->get_chain(chain_idx);
    *coords_out = protein->get_backbone_coords(chain_idx);
    *sequence_out = protein->get_sequence(chain_idx);
    *length_out = static_cast<int>(chain.size());

    if (*length_out <= 0) {
        throw errors::messages::empty_structure(path);
    }
}

void LoadEmbeddingSequence(const std::string& path, EmbeddingArray* array,
                           std::string* sequence_out) {
    *array = LoadEmbeddingFile(path);
    sequence_out->assign(static_cast<size_t>(array->rows), 'X');
}

msa::GuideTree BuildGuideTree(const std::string& method, const float* distances, int N,
                              pfalign::memory::GrowableArena* arena) {
    std::string lower = ToLower(method);
    if (lower == "upgma") {
        return tree_builders::build_upgma_tree(distances, N, arena);
    } else if (lower == "nj") {
        return tree_builders::build_nj_tree(distances, N, arena);
    } else if (lower == "bionj") {
        return tree_builders::build_bionj_tree(distances, N, arena);
    } else if (lower == "mst") {
        return tree_builders::build_mst_tree(distances, N, arena);
    }
    throw errors::messages::unknown_tree_method(method);
}

int SelectReferenceFromDistances(const std::vector<float>& distances, int N) {
    if (N <= 0 || static_cast<int>(distances.size()) < N * N) {
        return 0;
    }

    double best_score = std::numeric_limits<double>::infinity();
    int best_index = 0;

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i == j)
                continue;
            sum += distances[static_cast<size_t>(i) * N + j];
        }
        const double avg = (N > 1) ? sum / static_cast<double>(N - 1) : 0.0;
        if (avg < best_score) {
            best_score = avg;
            best_index = i;
        }
    }
    return best_index;
}

void WriteMsaMetrics(const std::string& metrics_path,
                     const std::vector<std::string>& resolved_inputs,
                     const std::string& output_path, const std::string& superpose_path,
                     const msa::MSAResult& result, const std::string& method, float gap_open,
                     float gap_extend, float temperature, float ecs_temperature, int k_neighbors,
                     int thread_count, int reference_index) {
    if (metrics_path.empty()) {
        return;
    }

    std::ofstream out(metrics_path);
    if (!out) {
        throw errors::FileWriteError(metrics_path);
    }

    out << "# PFalign MSA Metrics\n";
    out << "inputs_total: " << resolved_inputs.size() << "\n";
    for (const auto& path : resolved_inputs) {
        out << "input: " << path << "\n";
    }
    out << "method: " << method << "\n";
    out << "gap_open: " << gap_open << "\n";
    out << "gap_extend: " << gap_extend << "\n";
    out << "temperature: " << temperature << "\n";
    out << "ecs_temperature: " << ecs_temperature << "\n";
    out << "k_neighbors: " << k_neighbors << "\n";
    out << "threads: " << thread_count << "\n";
    out << "output_fasta: " << output_path << "\n";
    if (!superpose_path.empty()) {
        out << "superposed_pdb: " << superpose_path << "\n";
    }
    out << "num_sequences: " << result.num_sequences << "\n";
    out << "alignment_columns: " << result.aligned_length << "\n";
    out << "ecs_score: " << result.ecs << "\n";
    out << "reference_index: " << reference_index << "\n";
}

struct MSAProfiler {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    TimePoint start_total;
    TimePoint start_io;
    TimePoint start_mpnn;
    TimePoint start_similarity;
    TimePoint start_tree;
    TimePoint start_alignment;

    double time_io = 0.0;
    double time_mpnn = 0.0;
    double time_similarity = 0.0;
    double time_tree = 0.0;
    double time_alignment = 0.0;
    double time_total = 0.0;

    // CPU time tracking (user + system time)
    struct rusage rusage_start_mpnn;
    struct rusage rusage_start_similarity;
    struct rusage rusage_start_alignment;

    double cpu_time_mpnn = 0.0;
    double cpu_time_similarity = 0.0;
    double cpu_time_alignment = 0.0;

    int num_threads = 0;  // Available hardware threads

    MSAProfiler() {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    }

    void begin_total() {
        start_total = Clock::now();
    }

    void begin_io() {
        start_io = Clock::now();
    }

    void end_io() {
        time_io = Duration(Clock::now() - start_io).count();
    }

    void begin_mpnn() {
        start_mpnn = Clock::now();
        getrusage(RUSAGE_SELF, &rusage_start_mpnn);
    }

    void end_mpnn() {
        time_mpnn = Duration(Clock::now() - start_mpnn).count();

        struct rusage rusage_end;
        getrusage(RUSAGE_SELF, &rusage_end);

        // Calculate CPU time (user + system)
        double user_time = (rusage_end.ru_utime.tv_sec - rusage_start_mpnn.ru_utime.tv_sec) +
                           (rusage_end.ru_utime.tv_usec - rusage_start_mpnn.ru_utime.tv_usec) / 1e6;
        double sys_time = (rusage_end.ru_stime.tv_sec - rusage_start_mpnn.ru_stime.tv_sec) +
                          (rusage_end.ru_stime.tv_usec - rusage_start_mpnn.ru_stime.tv_usec) / 1e6;
        cpu_time_mpnn = user_time + sys_time;
    }

    void begin_similarity() {
        start_similarity = Clock::now();
        getrusage(RUSAGE_SELF, &rusage_start_similarity);
    }

    void end_similarity() {
        time_similarity = Duration(Clock::now() - start_similarity).count();

        struct rusage rusage_end;
        getrusage(RUSAGE_SELF, &rusage_end);

        // Calculate CPU time (user + system)
        double user_time =
            (rusage_end.ru_utime.tv_sec - rusage_start_similarity.ru_utime.tv_sec) +
            (rusage_end.ru_utime.tv_usec - rusage_start_similarity.ru_utime.tv_usec) / 1e6;
        double sys_time =
            (rusage_end.ru_stime.tv_sec - rusage_start_similarity.ru_stime.tv_sec) +
            (rusage_end.ru_stime.tv_usec - rusage_start_similarity.ru_stime.tv_usec) / 1e6;
        cpu_time_similarity = user_time + sys_time;
    }

    void begin_tree() {
        start_tree = Clock::now();
    }

    void end_tree() {
        time_tree = Duration(Clock::now() - start_tree).count();
    }

    void begin_alignment() {
        start_alignment = Clock::now();
        getrusage(RUSAGE_SELF, &rusage_start_alignment);
    }

    void end_alignment() {
        time_alignment = Duration(Clock::now() - start_alignment).count();

        struct rusage rusage_end;
        getrusage(RUSAGE_SELF, &rusage_end);

        // Calculate CPU time (user + system)
        double user_time =
            (rusage_end.ru_utime.tv_sec - rusage_start_alignment.ru_utime.tv_sec) +
            (rusage_end.ru_utime.tv_usec - rusage_start_alignment.ru_utime.tv_usec) / 1e6;
        double sys_time =
            (rusage_end.ru_stime.tv_sec - rusage_start_alignment.ru_stime.tv_sec) +
            (rusage_end.ru_stime.tv_usec - rusage_start_alignment.ru_stime.tv_usec) / 1e6;
        cpu_time_alignment = user_time + sys_time;
    }

    void end_total() {
        time_total = Duration(Clock::now() - start_total).count();
    }

    void print_summary(int N, int avg_length, int aligned_length) const {
        std::cout << "\n===========================================\n";
        std::cout << "  Performance Profile\n";
        std::cout << "===========================================\n\n";

        std::cout << "Total time:        " << format_time(time_total) << "\n";
        std::cout << "|- I/O:            " << format_time(time_io) << "  ("
                  << format_pct(time_io / time_total) << ")\n";
        std::cout << "|- MPNN Encoding:  " << format_time(time_mpnn) << "  ("
                  << format_pct(time_mpnn / time_total) << ")\n";
        std::cout << "|- Similarity:     " << format_time(time_similarity) << "  ("
                  << format_pct(time_similarity / time_total) << ")\n";
        std::cout << "|- Guide Tree:     " << format_time(time_tree) << "  ("
                  << format_pct(time_tree / time_total) << ")\n";
        std::cout << "+- Alignment:      " << format_time(time_alignment) << "  ("
                  << format_pct(time_alignment / time_total) << ")\n\n";

        // CPU utilization for the 3 main compute phases
        std::cout << "CPU Utilization (Hardware threads: " << num_threads << "):\n";

        // Calculate CPU utilization as: cpu_time / (wall_time * num_threads) * 100%
        // This shows how efficiently we're using available CPU resources
        double cpu_util_mpnn = (time_mpnn > 0 && num_threads > 0)
                                   ? (cpu_time_mpnn / (time_mpnn * num_threads)) * 100.0
                                   : 0.0;
        double cpu_util_similarity =
            (time_similarity > 0 && num_threads > 0)
                ? (cpu_time_similarity / (time_similarity * num_threads)) * 100.0
                : 0.0;
        double cpu_util_alignment =
            (time_alignment > 0 && num_threads > 0)
                ? (cpu_time_alignment / (time_alignment * num_threads)) * 100.0
                : 0.0;

        // Effective parallelism (how many cores we're using on average)
        double eff_cores_mpnn = (time_mpnn > 0) ? cpu_time_mpnn / time_mpnn : 0.0;
        double eff_cores_similarity =
            (time_similarity > 0) ? cpu_time_similarity / time_similarity : 0.0;
        double eff_cores_alignment =
            (time_alignment > 0) ? cpu_time_alignment / time_alignment : 0.0;

        char buf[256];
        snprintf(buf, sizeof(buf),
                 "  MPNN:       CPU: %6.2fs  Wall: %6.2fs  Util: %5.1f%%  (%.1f* cores)",
                 cpu_time_mpnn, time_mpnn, cpu_util_mpnn, eff_cores_mpnn);
        std::cout << buf << "\n";

        snprintf(buf, sizeof(buf),
                 "  Similarity: CPU: %6.2fs  Wall: %6.2fs  Util: %5.1f%%  (%.1f* cores)",
                 cpu_time_similarity, time_similarity, cpu_util_similarity, eff_cores_similarity);
        std::cout << buf << "\n";

        snprintf(buf, sizeof(buf),
                 "  Alignment:  CPU: %6.2fs  Wall: %6.2fs  Util: %5.1f%%  (%.1f* cores)",
                 cpu_time_alignment, time_alignment, cpu_util_alignment, eff_cores_alignment);
        std::cout << buf << "\n\n";

        // Per-residue metrics
        double total_residues = static_cast<double>(N) * avg_length;
        double ms_per_residue = (time_total * 1000.0) / total_residues;
        double ms_per_residue_mpnn = (time_mpnn * 1000.0) / total_residues;
        double ms_per_residue_alignment = (time_alignment * 1000.0) / (N * aligned_length);

        std::cout << "Throughput:\n";
        std::cout << "  MPNN:       " << format_throughput(ms_per_residue_mpnn) << "\n";
        std::cout << "  Alignment:  " << format_throughput(ms_per_residue_alignment) << "\n";
        std::cout << "  Overall:    " << format_throughput(ms_per_residue) << "\n";
        std::cout << "\n";
    }

private:
    static std::string format_time(double seconds) {
        if (seconds < 0.001) {
            return std::to_string(static_cast<int>(seconds * 1e6)) + " mus";
        } else if (seconds < 1.0) {
            return std::to_string(static_cast<int>(seconds * 1e3)) + " ms";
        } else if (seconds < 60.0) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f s", seconds);
            return std::string(buf);
        } else {
            int mins = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(seconds) % 60;
            return std::to_string(mins) + "m " + std::to_string(secs) + "s";
        }
    }

    static std::string format_pct(double fraction) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%5.1f%%", fraction * 100.0);
        return std::string(buf);
    }

    static std::string format_throughput(double ms_per_residue) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.3f ms/residue", ms_per_residue);
        return std::string(buf);
    }
};

}  // namespace

int msa(const std::vector<std::string>& inputs, const std::string& input_list,
        const std::string& input_dir, const std::string& output_path,
        const std::string& superpose_path, const std::string& metrics_path,
        const std::string& method, float gap_open, float gap_extend, float temperature,
        float ecs_temperature, int k_neighbors, int arena_size_mb, int thread_count) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign MSA Command\n";
    std::cout << "===========================================\n\n";

    MSAProfiler profiler;
    profiler.begin_total();

    try {
        auto resolved_inputs = ResolveInputs(inputs, input_list, input_dir);
        if (resolved_inputs.size() < 2) {
            throw errors::messages::insufficient_sequences_for_msa(
                static_cast<int>(resolved_inputs.size()));
        }

        InputType input_type = DetectInputType(resolved_inputs.front());
        for (const auto& path : resolved_inputs) {
            if (DetectInputType(path) != input_type) {
                throw errors::ValidationError(
                    "Mixed input types not supported",
                    "All inputs must be the same type (all structures or all embeddings)");
            }
        }
        if (!superpose_path.empty() && input_type != InputType::kStructure) {
            throw errors::ValidationError(
                "Superposed PDB output requires structure inputs",
                "All inputs must be structure files to generate superposed PDB");
        }

        mpnn::MPNNWeights weights(0);
        mpnn::MPNNConfig mpnn_config;
        int embedding_dim = 0;
        int reference_index = 0;

        if (input_type == InputType::kStructure) {
            auto loaded = weights::load_embedded_mpnn_weights();
            weights = std::move(std::get<0>(loaded));
            mpnn_config = std::get<1>(loaded);
            mpnn_config.k_neighbors = k_neighbors;
        }

        // Validate arena_size_mb
        if (arena_size_mb <= 0) {
            throw errors::messages::parameter_must_be_positive(
                "arena_size_mb", std::to_string(arena_size_mb));
        }

        const size_t arena_mb = static_cast<size_t>(arena_size_mb);
        pfalign::memory::GrowableArena arena(arena_mb);
        const size_t scratch_mb = std::max<size_t>(64, arena_size_mb / 4);
        pfalign::memory::GrowableArena scratch_arena(scratch_mb);
        SequenceCache cache(&arena);

        // Create memory profiler to track allocations
        std::string session_name = "msa_n" + std::to_string(resolved_inputs.size()) + "_" + method;
        pfalign::memory::MemoryProfiler mem_profiler(session_name);

        std::cout << "Inputs:       " << resolved_inputs.size() << "\n";
        std::cout << "Mode:         " << method << "\n";
        std::cout << "gap_open:     " << gap_open << "\n";
        std::cout << "gap_extend:   " << gap_extend << "\n";
        std::cout << "temperature:  " << temperature << "\n";
        std::cout << "ECS temp:     " << ecs_temperature << "\n";
        std::cout << "Arena (MB):   " << arena_size_mb << "\n";
        if (!superpose_path.empty()) {
            std::cout << "Superpose:    " << superpose_path << "\n";
        }
        if (!metrics_path.empty()) {
            std::cout << "Metrics:      " << metrics_path << "\n";
        }
        std::cout << "\n";

        profiler.begin_io();

        if (thread_count < 0) {
            throw errors::messages::parameter_must_be_non_negative(
                "threads", std::to_string(thread_count));
        }
        if (thread_count > 0) {
            int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
            if (hardware_threads <= 0) {
                hardware_threads = 1;
            }
            if (thread_count > hardware_threads) {
                std::cout << "Requested " << thread_count << " threads but only "
                          << hardware_threads << " hardware threads detected. "
                          << "Using " << hardware_threads << " threads.\n";
                thread_count = hardware_threads;
            }
            profiler.num_threads = thread_count;
        }

        if (input_type == InputType::kStructure) {
            // Load all structures first (I/O is sequential)
            struct StructureData {
                std::vector<float> coords;
                std::string identifier;
                std::string sequence;
                int length;
            };

            std::vector<StructureData> structures;
            structures.reserve(resolved_inputs.size());

            for (const auto& path : resolved_inputs) {
                io::Protein protein;
                std::vector<float> coords;
                std::string sequence;
                int length = 0;
                LoadStructureSequence(path, &protein, &coords, &sequence, &length);
                char chain_id = protein.get_chain(0).chain_id;
                std::string identifier = MakeIdentifier(path, chain_id);
                structures.push_back(
                    {std::move(coords), std::move(identifier), std::move(sequence), length});
            }

            profiler.end_io();
            profiler.begin_mpnn();

            // Parallel MPNN encoding (default behavior)
            threading::ThreadPool pool(static_cast<size_t>(thread_count), arena_size_mb);
            profiler.num_threads = static_cast<int>(pool.num_threads());

            pool.parallel_for(structures.size(), [&](int tid, size_t begin, size_t end,
                                                     memory::GrowableArena& thread_arena) {
                (void)tid;  // Unused

                // Create thread-local adapter using thread's arena
                mpnn::MPNNCacheAdapter local_adapter(cache, weights, mpnn_config, &thread_arena);

                for (size_t idx = begin; idx < end; ++idx) {
                    local_adapter.add_protein(static_cast<int>(idx), structures[idx].coords.data(),
                                              structures[idx].length, structures[idx].identifier,
                                              structures[idx].sequence);
                }
            });

            profiler.end_mpnn();

            // Record MPNN thread pool memory
            mem_profiler.record_threadpool("mpnn_pool", pool.num_threads(), pool.total_peak_mb());

            embedding_dim = mpnn_config.hidden_dim;
        } else {
            // Embedding path: I/O only, no MPNN encoding needed
            for (size_t idx = 0; idx < resolved_inputs.size(); ++idx) {
                EmbeddingArray array;
                std::string sequence;
                LoadEmbeddingSequence(resolved_inputs[idx], &array, &sequence);

                // Validate that embeddings are not empty
                if (array.rows == 0) {
                    throw errors::ValidationError(
                        "Empty embedding array",
                        "File " + resolved_inputs[idx] + " contains zero rows");
                }

                if (embedding_dim == 0) {
                    embedding_dim = array.cols;
                } else if (embedding_dim != array.cols) {
                    throw errors::messages::embedding_dimension_mismatch(
                        array.cols, embedding_dim, resolved_inputs[idx], "previous inputs");
                }
                std::string identifier = MakeIdentifier(resolved_inputs[idx]);
                cache.add_precomputed(static_cast<int>(idx), array.values.data(), array.rows,
                                      array.cols, nullptr, identifier, sequence);
            }
            if (embedding_dim == 0) {
                throw errors::ValidationError(
                    "Failed to determine embedding dimension",
                    "Check that embedding files are valid and non-empty");
            }
            mpnn_config.hidden_dim = embedding_dim;
            mpnn_config.k_neighbors = k_neighbors;

            profiler.end_io();
            // No MPNN encoding for precomputed embeddings
        }

        const int N = cache.size();
        if (N < 2) {
            throw errors::messages::insufficient_sequences_for_msa(N);
        }

        profiler.begin_similarity();

        std::vector<float> distances(static_cast<size_t>(N) * N, 0.0f);
        smith_waterman::SWConfig sw_config;
        sw_config.affine = true;
        sw_config.gap_open = gap_open;
        sw_config.gap_extend = gap_extend;
        sw_config.temperature = temperature;
        sw_config.gap = gap_extend;

        msa::compute_distance_matrix_alignment(cache, sw_config, &scratch_arena, distances.data());

        profiler.end_similarity();

        if (!msa::validate_distance_matrix(distances.data(), N)) {
            throw errors::AlgorithmError(
                "Distance matrix validation",
                "Distance matrix contains invalid values",
                "Check input sequences are valid and alignment parameters are reasonable");
        }

        reference_index = SelectReferenceFromDistances(distances, N);

        profiler.begin_tree();
        auto tree = BuildGuideTree(method, distances.data(), N, &arena);
        profiler.end_tree();

        profiler.begin_alignment();

        msa::MSAConfig msa_config;
        msa_config.gap_open = gap_open;
        msa_config.gap_extend = gap_extend;
        msa_config.temperature = temperature;
        msa_config.gap_penalty = gap_extend;
        msa_config.ecs_temperature = ecs_temperature;
        msa_config.thread_count = thread_count;
        msa_config.use_affine_gaps = true;

        auto result = msa::progressive_msa<ScalarBackend>(cache, tree, msa_config, &arena);

        profiler.end_alignment();
        profiler.end_total();

        // Record memory usage from all components
        mem_profiler.record_arena("main_arena", arena.peak(), arena.capacity());
        mem_profiler.record_arena("scratch_arena", scratch_arena.peak(), scratch_arena.capacity());

        // Record sequence cache size
        size_t cache_bytes = static_cast<size_t>(cache.size()) * cache.max_length() *
                             cache.hidden_dim() * sizeof(float);
        mem_profiler.record_cache("sequence_cache", cache_bytes, cache.size());

        if (!result.alignment) {
            throw errors::messages::alignment_failed("alignment profile is null");
        }

        if (!result.write_fasta(output_path)) {
            throw errors::FileWriteError(output_path, "Failed to write FASTA alignment");
        }

        std::cout << "✓ MSA complete\n";
        std::cout << "  Sequences:  " << result.num_sequences << "\n";
        std::cout << "  Alignment:  " << result.aligned_length << " columns\n";
        std::cout << "  ECS score:  " << result.ecs << "\n";
        std::cout << "  Output:     " << output_path << "\n";
        if (!superpose_path.empty()) {
            if (!result.write_superposed_pdb(superpose_path, reference_index)) {
                throw errors::FileWriteError(superpose_path, "Failed to write superposed structure");
            }
            std::cout << "  Superpose:  " << superpose_path << "\n";
        }

        if (!metrics_path.empty()) {
            WriteMsaMetrics(metrics_path, resolved_inputs, output_path, superpose_path, result,
                            method, gap_open, gap_extend, temperature, ecs_temperature, k_neighbors,
                            thread_count, reference_index);
            std::cout << "  Metrics:    " << metrics_path << "\n";
        }

        // Calculate average sequence length for profiling
        int total_length = 0;
        int valid_sequences = 0;
        for (int i = 0; i < N; ++i) {
            const auto* seq = cache.get(i);
            if (seq) {
                total_length += seq->length;
                valid_sequences++;
            }
        }
        int avg_length = (valid_sequences > 0) ? (total_length / valid_sequences) : 0;

        profiler.print_summary(N, avg_length, result.aligned_length);

        // Print memory profile
        std::cout << "\n";
        mem_profiler.print_summary();

        if (result.alignment) {
            msa::Profile::destroy(result.alignment);
        }

        return 0;
    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "✗ MSA command failed: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace commands
}  // namespace pfalign
