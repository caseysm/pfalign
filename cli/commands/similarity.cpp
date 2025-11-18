#include "commands.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "commands/input_utils.h"
#include "pfalign/algorithms/distance_metrics/distance_matrix.h"
#include "pfalign/common/growable_arena.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/tools/weights/save_npy.h"

namespace pfalign {
namespace commands {

int similarity(const std::string& emb1_path, const std::string& emb2_path,
               const std::string& output_path) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign Similarity Command\n";
    std::cout << "===========================================\n\n";
    std::cout << "Embedding 1:  " << emb1_path << "\n";
    std::cout << "Embedding 2:  " << emb2_path << "\n";
    std::cout << "Output:       " << output_path << "\n\n";

    try {
        EmbeddingArray emb1 = LoadEmbeddingFile(emb1_path);
        EmbeddingArray emb2 = LoadEmbeddingFile(emb2_path);

        if (emb1.cols != emb2.cols) {
            throw errors::messages::embedding_dimension_mismatch(
                emb1.cols, emb2.cols, emb1_path, emb2_path);
        }

        std::vector<float> similarity(static_cast<size_t>(emb1.rows) * emb2.rows);
        similarity::compute_similarity<ScalarBackend>(emb1.values.data(), emb2.values.data(),
                                                      similarity.data(), emb1.rows, emb2.rows,
                                                      emb1.cols);

        save_npy_2d(output_path, similarity.data(), emb1.rows, emb2.rows);

        std::cout << "✓ Similarity matrix saved (" << emb1.rows << " x " << emb2.rows << ")\n";
        return 0;
    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "✗ Similarity command failed: " << e.what() << "\n";
        return 1;
    }
}

int compute_distances(const std::vector<std::string>& embedding_paths,
                     const std::string& output_path, float gap_open, float gap_extend,
                     float temperature) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign Compute Distances Command\n";
    std::cout << "===========================================\n\n";
    std::cout << "Input embeddings: " << embedding_paths.size() << "\n";
    std::cout << "Output:           " << output_path << "\n";
    std::cout << "Gap open:         " << gap_open << "\n";
    std::cout << "Gap extend:       " << gap_extend << "\n";
    std::cout << "Temperature:      " << temperature << "\n\n";

    try {
        const int N = static_cast<int>(embedding_paths.size());

        // Load all embeddings
        std::cout << "Loading embeddings...\n";
        std::vector<EmbeddingArray> embeddings;
        embeddings.reserve(N);

        for (int i = 0; i < N; i++) {
            embeddings.push_back(LoadEmbeddingFile(embedding_paths[i]));
            std::cout << "  [" << (i + 1) << "/" << N << "] " << embedding_paths[i]
                      << " (" << embeddings[i].rows << ", " << embeddings[i].cols << ")\n";
        }

        // Create sequence cache
        const size_t arena_mb = 200;
        pfalign::memory::GrowableArena arena(arena_mb);
        pfalign::SequenceCache cache(&arena);

        // Add embeddings to cache
        for (int i = 0; i < N; i++) {
            std::string identifier = "seq_" + std::to_string(i);
            std::string sequence(embeddings[i].rows, 'X');
            cache.add_precomputed(i, embeddings[i].values.data(), embeddings[i].rows,
                                 embeddings[i].cols, nullptr, identifier, sequence);
        }

        // Compute distance matrix
        std::cout << "\nComputing distance matrix...\n";
        const size_t scratch_mb = 10;
        pfalign::memory::GrowableArena scratch(scratch_mb);

        pfalign::smith_waterman::SWConfig sw_cfg;
        sw_cfg.affine = true;
        sw_cfg.gap_open = gap_open;
        sw_cfg.gap_extend = gap_extend;
        sw_cfg.temperature = temperature;

        std::vector<float> distances(static_cast<size_t>(N) * N, 0.0f);
        pfalign::msa::compute_distance_matrix_alignment(cache, sw_cfg, &scratch,
                                                        distances.data());

        // Save distance matrix
        save_npy_2d(output_path, distances.data(), N, N);

        std::cout << "✓ Distance matrix saved (" << N << " x " << N << ")\n";
        std::cout << "  Range: [" << *std::min_element(distances.begin(), distances.end())
                  << ", " << *std::max_element(distances.begin(), distances.end()) << "]\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Compute distances command failed: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace commands
}  // namespace pfalign
