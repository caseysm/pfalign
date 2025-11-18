/**
 * Utility to regenerate MPNN golden embeddings from the C++ encoder.
 *
 * Usage:
 *   ./regenerate_mpnn_golden <label> [<label> ...]
 *
 * Labels correspond to directory names under data/golden/mpnn/.
 * If no labels are provided, all known cases are regenerated.
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/common/golden_data_test.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/tools/weights/save_npy.h"
#include "pfalign/tools/weights/mpnn_weight_loader.h"

#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::testing;

std::filesystem::path project_root() {
    auto path = std::filesystem::path(__FILE__).parent_path();
    for (int i = 0; i < 5; ++i) {
        path = path.parent_path();
    }
    return path;
}

std::tuple<MPNNWeights, MPNNConfig> load_weights() {
    auto root = project_root();
    auto weights_path = root / "data" / "golden" / "mpnn" / "weights" / "mpnn_golden_weights.safetensors";
    if (std::filesystem::exists(weights_path)) {
        auto [weights, config, sw_params] = pfalign::weights::MPNNWeightLoader::load(weights_path.string());
        (void)sw_params;
        return {std::move(weights), config};
    }
    auto [weights, config, sw_params] = pfalign::weights::load_embedded_mpnn_weights();
    (void)sw_params;
    return {std::move(weights), config};
}

bool regenerate_case(
    const std::filesystem::path& dir,
    const MPNNWeights& weights,
    const MPNNConfig& config
) {
    GoldenDataTest test(dir.string());
    auto [coords_flat, shape] = test.load_with_shape("input_coords.npy");
    if (shape.size() != 3 || shape[1] != 4 || shape[2] != 3) {
        std::cerr << "  ✗ Unexpected coordinate shape" << std::endl;
        return false;
    }

    const int L = static_cast<int>(shape[0]);
    std::cout << "  Length: " << L << std::endl;

    // Allocate workspace and output
    std::vector<float> embeddings(static_cast<size_t>(L) * config.hidden_dim, 0.0f);
    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);
    for (int i = 0; i < L; ++i) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    mpnn_forward<ScalarBackend>(
        coords_flat.data(),
        L,
        weights,
        config,
        embeddings.data(),
        &workspace
    );

    std::vector<float> ref_embeddings;
    std::vector<size_t> ref_shape;
    try {
        auto [ref_flat, ref_dims] = test.load_with_shape("output_embeddings.npy");
        ref_embeddings = std::move(ref_flat);
        ref_shape = ref_dims;
    } catch (...) {
        // No existing file
    }

    if (!ref_embeddings.empty() && ref_shape == std::vector<size_t>{static_cast<size_t>(L), static_cast<size_t>(config.hidden_dim)}) {
        float max_diff = 0.0f;
        for (size_t i = 0; i < ref_embeddings.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(ref_embeddings[i] - embeddings[i]));
        }
        std::cout << "  Max diff vs existing file: " << max_diff << std::endl;
    }

    const std::vector<size_t> shape_out = {
        static_cast<size_t>(L),
        static_cast<size_t>(config.hidden_dim)
    };
    const auto emb_path = dir / "output_embeddings.npy";
    if (!save_npy_float32(emb_path.string(), embeddings.data(), shape_out)) {
        std::cerr << "  ✗ Failed to write " << emb_path << std::endl;
        return false;
    }
    std::cout << "  ✓ Updated " << emb_path << std::endl;
    return true;
}

int main(int argc, char** argv) {
    auto root = project_root();
    auto golden_root = root / "data" / "golden" / "mpnn";

    std::vector<std::string> all_cases = {
        "small_10res",
        "medium_50res",
        "large_100res",
        "villin_1VII",
        "crambin_1CRN",
        "ubiquitin_1UBQ",
        "myoglobin_1MBO",
        "hemoglobin_1HBS",
        "ribonuclease_1RNH",
        "lysozyme_2LYZ",
        "afdb_kinase_P00519",
        "afdb_globin_P69905",
        "afdb_enzyme_P04406",
        "esm_MGYP003592128331",
        "esm_MGYP002802792805",
        "esm_MGYP001105483357",
        "lysozyme_1LYZ",
        "immunoglobulin_1IGT",
        "immunoglobulin_1FBI"
    };

    std::unordered_set<std::string> selected;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            selected.insert(argv[i]);
        }
    } else {
        selected.insert(all_cases.begin(), all_cases.end());
    }

    auto [weights, config] = load_weights();

    int updated = 0;
    for (const auto& label : all_cases) {
        if (!selected.count(label)) {
            continue;
        }
        auto dir = golden_root / label;
        if (!std::filesystem::exists(dir / "input_coords.npy")) {
            std::cout << "SKIP " << label << " (missing coords)" << std::endl;
            continue;
        }
        std::cout << "=== Regenerating " << label << " ===" << std::endl;
        if (regenerate_case(dir, weights, config)) {
            updated++;
        } else {
            std::cerr << "Failed to regenerate " << label << std::endl;
        }
    }

    std::cout << "Updated " << updated << " cases" << std::endl;
    return 0;
}
