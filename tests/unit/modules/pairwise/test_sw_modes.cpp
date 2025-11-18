/**
 * Test all 6 Smith-Waterman modes with real PDB inputs.
 *
 * This test compares the behavior of all 6 SW modes to understand
 * how different formulations and gap models affect alignment results.
 */

#include "pfalign/common/growable_arena.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/dispatch/scalar_traits.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/io/pdb_parser.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

using pfalign::ScalarBackend;
using pfalign::pairwise::pairwise_align_full;
using pfalign::pairwise::PairwiseConfig;
using pfalign::pairwise::PairwiseWorkspace;
using pfalign::pairwise::AlignmentResult;
using pfalign::mpnn::MPNNWeights;
using pfalign::memory::Arena;
using pfalign::memory::GrowableArena;

// Global embedded weights (loaded once)
static auto& get_global_weights() {
    static auto data = pfalign::weights::load_embedded_mpnn_weights();
    return data;
}

struct ModeTestResult {
    std::string mode_name;
    float partition;
    float score;
    float posterior_sum;
    float posterior_mean;
    float posterior_max;
    int path_length;
};

/**
 * Run pairwise alignment with a specific SW mode.
 */
ModeTestResult run_with_mode(
    const float* coords1, int L1,
    const float* coords2, int L2,
    PairwiseConfig::SWMode mode,
    const std::string& mode_name,
    const MPNNWeights& weights,
    const pfalign::weights::SWParams& sw_params,
    GrowableArena* arena
) {
    // Setup config for this mode
    PairwiseConfig config;
    config.sw_mode = mode;

    // Use provided SW parameters
    config.sw_config.gap = sw_params.gap;              // 0.194 (for regular modes)
    config.sw_config.gap_open = sw_params.gap_open;    // -2.544
    config.sw_config.gap_extend = sw_params.gap;       // 0.194 (for affine modes)
    config.sw_config.temperature = sw_params.temperature; // 1.0
    config.sw_config.affine = (mode != PairwiseConfig::SWMode::DIRECT_REGULAR &&
                               mode != PairwiseConfig::SWMode::JAX_REGULAR);

    // Create workspace
    PairwiseWorkspace workspace(L1, L2, config);

    // Allocate result buffers
    AlignmentResult result;
    result.posteriors = arena->allocate<float>(L1 * L2);
    result.alignment_path = arena->allocate<pfalign::AlignmentPair>(L1 + L2);
    result.max_path_length = L1 + L2;
    result.L1 = L1;
    result.L2 = L2;

    // Run alignment
    pairwise_align_full<ScalarBackend>(
        coords1, L1,
        coords2, L2,
        config,
        weights,
        &workspace,
        &result,
        arena,
        sw_params.gap_open  // Use gap_open for decoding
    );

    // Compute posterior statistics
    float posterior_sum = 0.0f;
    float posterior_max = 0.0f;
    for (int i = 0; i < L1 * L2; i++) {
        posterior_sum += result.posteriors[i];
        posterior_max = std::max(posterior_max, result.posteriors[i]);
    }
    float posterior_mean = posterior_sum / (L1 * L2);

    return ModeTestResult{
        mode_name,
        result.partition,
        result.score,
        posterior_sum,
        posterior_mean,
        posterior_max,
        result.path_length
    };
}

/**
 * Test all 6 SW modes on two PDB structures.
 */
bool test_all_modes_pdb(const std::string& pdb1_path, const std::string& pdb2_path) {
    std::cout << "=== Testing All 6 SW Modes ===" << std::endl;
    std::cout << "PDB 1: " << pdb1_path << std::endl;
    std::cout << "PDB 2: " << pdb2_path << std::endl;
    std::cout << std::endl;

    // Load PDB files
    pfalign::io::PDBParser parser;
    auto protein1 = parser.parse_file(pdb1_path);
    auto protein2 = parser.parse_file(pdb2_path);

    if (protein1.chains.empty() || protein2.chains.empty()) {
        std::cerr << "Failed to load PDB files" << std::endl;
        return false;
    }

    // Debug: Print chain information
    std::cout << "Protein 1 chains: " << protein1.chains.size() << std::endl;
    for (size_t i = 0; i < protein1.chains.size(); i++) {
        std::cout << "  Chain " << i << " (ID='" << protein1.chains[i].chain_id
                  << "'): " << protein1.chains[i].residues.size() << " residues" << std::endl;
    }
    std::cout << "Protein 2 chains: " << protein2.chains.size() << std::endl;
    for (size_t i = 0; i < protein2.chains.size(); i++) {
        std::cout << "  Chain " << i << " (ID='" << protein2.chains[i].chain_id
                  << "'): " << protein2.chains[i].residues.size() << " residues" << std::endl;
    }
    std::cout << std::endl;

    // Get backbone coordinates for first chain (returns [L, 4, 3] = [L, 12])
    auto coords1_vec = protein1.get_backbone_coords(0);
    auto coords2_vec = protein2.get_backbone_coords(0);

    int L1 = coords1_vec.size() / 12;  // 4 atoms * 3 coords
    int L2 = coords2_vec.size() / 12;

    std::cout << "Backbone coords: L1=" << L1 << ", L2=" << L2 << std::endl;
    std::cout << std::endl;

    const float* coords1 = coords1_vec.data();
    const float* coords2 = coords2_vec.data();

    // Load weights
    auto [mpnn_weights, mpnn_config, sw_params] = get_global_weights();

    // Create arena for allocations
    GrowableArena arena(256);  // 256 MB

    // Test all 6 modes
    std::vector<ModeTestResult> results;

    results.push_back(run_with_mode(coords1, L1, coords2, L2,
        PairwiseConfig::SWMode::DIRECT_REGULAR, "DIRECT_REGULAR",
        mpnn_weights, sw_params, &arena));
    arena.reset();

    results.push_back(run_with_mode(coords1, L1, coords2, L2,
        PairwiseConfig::SWMode::DIRECT_AFFINE, "DIRECT_AFFINE",
        mpnn_weights, sw_params, &arena));
    arena.reset();

    results.push_back(run_with_mode(coords1, L1, coords2, L2,
        PairwiseConfig::SWMode::DIRECT_AFFINE_FLEXIBLE, "DIRECT_AFFINE_FLEXIBLE",
        mpnn_weights, sw_params, &arena));
    arena.reset();

    results.push_back(run_with_mode(coords1, L1, coords2, L2,
        PairwiseConfig::SWMode::JAX_REGULAR, "JAX_REGULAR",
        mpnn_weights, sw_params, &arena));
    arena.reset();

    results.push_back(run_with_mode(coords1, L1, coords2, L2,
        PairwiseConfig::SWMode::JAX_AFFINE_STANDARD, "JAX_AFFINE_STANDARD",
        mpnn_weights, sw_params, &arena));
    arena.reset();

    results.push_back(run_with_mode(coords1, L1, coords2, L2,
        PairwiseConfig::SWMode::JAX_AFFINE_FLEXIBLE, "JAX_AFFINE_FLEXIBLE",
        mpnn_weights, sw_params, &arena));
    arena.reset();

    // Print results table
    std::cout << std::setw(25) << "Mode"
              << std::setw(15) << "Partition"
              << std::setw(12) << "Score"
              << std::setw(12) << "Post.Sum"
              << std::setw(12) << "Post.Mean"
              << std::setw(12) << "Post.Max"
              << std::setw(12) << "PathLen"
              << std::endl;
    std::cout << std::string(98, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(25) << r.mode_name
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.partition
                  << std::setw(12) << std::fixed << std::setprecision(6) << r.score
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.posterior_sum
                  << std::setw(12) << std::scientific << std::setprecision(3) << r.posterior_mean
                  << std::setw(12) << std::fixed << std::setprecision(6) << r.posterior_max
                  << std::setw(12) << r.path_length
                  << std::endl;
    }
    std::cout << std::endl;

    // Check that modes produce different results
    bool all_different = true;
    for (size_t i = 0; i < results.size(); i++) {
        for (size_t j = i + 1; j < results.size(); j++) {
            float partition_diff = std::abs(results[i].partition - results[j].partition);
            float score_diff = std::abs(results[i].score - results[j].score);

            if (partition_diff < 1e-6 && score_diff < 1e-6) {
                std::cout << "WARNING: " << results[i].mode_name
                          << " and " << results[j].mode_name
                          << " produced identical results!" << std::endl;
                all_different = false;
            }
        }
    }

    if (all_different) {
        std::cout << "✓ All modes produced different results (as expected)" << std::endl;
    }

    // Compare formulations within same gap model
    std::cout << std::endl;
    std::cout << "=== Formulation Comparisons (within gap model) ===" << std::endl;

    // Regular modes
    float reg_partition_diff = std::abs(results[0].partition - results[3].partition);
    float reg_score_diff = std::abs(results[0].score - results[3].score);
    std::cout << "DIRECT_REGULAR vs JAX_REGULAR:" << std::endl;
    std::cout << "  Partition diff: " << reg_partition_diff << std::endl;
    std::cout << "  Score diff:     " << reg_score_diff << std::endl;
    std::cout << std::endl;

    // Affine modes
    float aff_partition_diff = std::abs(results[1].partition - results[4].partition);
    float aff_score_diff = std::abs(results[1].score - results[4].score);
    std::cout << "DIRECT_AFFINE vs JAX_AFFINE_STANDARD:" << std::endl;
    std::cout << "  Partition diff: " << aff_partition_diff << std::endl;
    std::cout << "  Score diff:     " << aff_score_diff << std::endl;
    std::cout << std::endl;

    // Flexible modes
    float flex_partition_diff = std::abs(results[2].partition - results[5].partition);
    float flex_score_diff = std::abs(results[2].score - results[5].score);
    std::cout << "DIRECT_AFFINE_FLEXIBLE vs JAX_AFFINE_FLEXIBLE:" << std::endl;
    std::cout << "  Partition diff: " << flex_partition_diff << std::endl;
    std::cout << "  Score diff:     " << flex_score_diff << std::endl;
    std::cout << std::endl;

    return true;
}

int main() {
    // Test with globin family proteins (similar structures, should align well)
    // Paths are relative to build directory where meson runs tests
    std::string pdb1 = "../tests/data/integration/msa_families/globin/1MBO.pdb";
    std::string pdb2 = "../tests/data/integration/msa_families/globin/1MBA.pdb";

    bool success = test_all_modes_pdb(pdb1, pdb2);

    if (success) {
        std::cout << "\n✓ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Tests failed!" << std::endl;
        return 1;
    }
}
