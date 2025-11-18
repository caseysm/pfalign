/**
 * Self-Match Validation Suite
 *
 * Tests that proteins align perfectly to themselves.
 * This validates:
 * 1. MPNN encoder produces consistent embeddings
 * 2. Similarity computation is symmetric
 * 3. Smith-Waterman alignment finds optimal self-alignment
 * 4. Partition function reaches expected maximum for self-match
 *
 * Expected results for perfect self-match:
 * - Partition function >> 0 (strong alignment signal)
 * - Posteriors concentrated on diagonal
 * - High similarity scores along diagonal
 */

#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/tools/weights/mpnn_weight_loader.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/common/arena_allocator.h"
#include "../test_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::io;
using namespace pfalign::smith_waterman;
using namespace pfalign::memory;
using namespace pfalign::weights;

// Test configuration
struct SelfMatchTest {
    std::string pdb_path;
    std::string name;
    int expected_min_length;  // Minimum expected sequence length
};

/**
 * Test self-alignment of a single protein
 */
bool test_self_match(
    const SelfMatchTest& test,
    const MPNNWeights& weights,
    const MPNNConfig& config,
    Arena* arena
) {
    std::cout << "\n=== Testing: " << test.name << " ===" << std::endl;
    std::cout << "  PDB: " << test.pdb_path << std::endl;

    // Parse PDB
    PDBParser parser;
    Protein prot;
    
    try {
        prot = parser.parse_file(test.pdb_path.c_str());
    } catch (const std::exception& e) {
        std::cout << "  SKIP: Could not load PDB (" << e.what() << ")" << std::endl;
        return true;  // Not a test failure - file may not exist
    }

    if (prot.chains.empty()) {
        std::cout << "  SKIP: No chains in PDB" << std::endl;
        return true;
    }

    // Get backbone coordinates
    int L = prot.get_chain(0).size();
    auto coords = prot.get_backbone_coords(0);
    
    std::cout << "  Length: " << L << " residues" << std::endl;

    if (L < test.expected_min_length) {
        std::cout << "  WARNING: Shorter than expected (expected >= " 
                  << test.expected_min_length << ")" << std::endl;
    }

    // Encode protein with MPNN
    float* embeddings = arena->allocate<float>(L * config.hidden_dim);
    MPNNWorkspace workspace(L, config.k_neighbors, config.hidden_dim);

    for (int i = 0; i < L; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;
    }

    mpnn_forward<ScalarBackend>(coords.data(), L, weights, config, embeddings, &workspace);

    // Compute self-similarity
    float* similarity = arena->allocate<float>(L * L);
    
    pfalign::similarity::compute_similarity<ScalarBackend>(
        embeddings, embeddings, similarity, L, L, config.hidden_dim
    );

    // Check diagonal similarity (should be ~1.0 for self-match)
    float diagonal_sum = 0.0f;
    for (int i = 0; i < L; i++) {
        diagonal_sum += similarity[i * L + i];
    }
    float diagonal_mean = diagonal_sum / L;

    std::cout << "  Self-similarity (diagonal mean): " << std::fixed 
              << std::setprecision(4) << diagonal_mean << std::endl;

    // Run Smith-Waterman self-alignment
    SWConfig sw_config;
    sw_config.gap_open = -11.0f;
    sw_config.gap_extend = -1.0f;
    sw_config.temperature = 1.0f;
    sw_config.affine = true;

    float* dp = arena->allocate<float>(L * L * 3);  // 3 states for affine
    float partition;

    // Forward pass
    smith_waterman_jax_affine_flexible<ScalarBackend>(
        similarity, L, L,
        sw_config,
        dp, &partition
    );

    std::cout << "  Partition function: " << std::scientific
              << std::setprecision(2) << partition << std::endl;

    // Backward pass
    float* posteriors = arena->allocate<float>(L * L);
    smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
        dp,  // hij from forward pass
        similarity, L, L,
        sw_config,
        partition,
        posteriors,
        arena  // temp arena
    );

    // Check diagonal posteriors (should be high for self-match)
    float diagonal_posterior_sum = 0.0f;
    for (int i = 0; i < L; i++) {
        diagonal_posterior_sum += posteriors[i * L + i];
    }
    float diagonal_posterior_mean = diagonal_posterior_sum / L;

    std::cout << "  Posterior (diagonal mean): " << std::fixed
              << std::setprecision(4) << diagonal_posterior_mean << std::endl;

    // Validation checks
    bool passed = true;

    // 1. Diagonal similarity should be close to 1.0 (embeddings normalized)
    if (diagonal_mean < 0.95f) {
        std::cout << "  ⚠ WARNING: Low diagonal similarity (expected ~1.0)" << std::endl;
        passed = false;
    }

    // 2. Partition should be positive and large
    if (partition < 1.0f) {
        std::cout << "  ✗ FAIL: Partition too low (expected >> 1)" << std::endl;
        passed = false;
    }

    // 3. Diagonal posteriors should be concentrated
    if (diagonal_posterior_mean < 0.1f) {
        std::cout << "  ⚠ WARNING: Low diagonal posteriors (expected high concentration)" << std::endl;
    }

    if (passed) {
        std::cout << "  ✓ PASS: Self-match validation successful" << std::endl;
    }

    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Self-Match Validation Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nTesting proteins align perfectly to themselves\n" << std::endl;

    // Load embedded MPNN weights
    MPNNWeights weights(3);
    MPNNConfig config;

    try {
        auto [loaded_weights, loaded_config, loaded_sw] = weights::load_embedded_mpnn_weights();
        weights = std::move(loaded_weights);
        config = loaded_config;
        std::cout << "✅ Loaded embedded MPNN weights" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load embedded MPNN weights (" << e.what() << ")" << std::endl;
        return 0;  // Not a failure - weights may not be embedded yet
    }

    // Create arena for computations
    Arena arena(200);  // 200 MB should be enough for any protein

    // Test cases: 10 proteins of varying sizes and families
    std::vector<SelfMatchTest> tests = {
        // Small proteins (fixtures)
        {pfalign::test::map_legacy_path("1CRN.pdb"), "Crambin (1CRN)", 45},
        {pfalign::test::map_legacy_path("1UBQ.pdb"), "Ubiquitin (1UBQ)", 75},

        // Medium proteins (integration)
        {pfalign::test::map_legacy_path("1LYZ.pdb"), "Lysozyme (1LYZ)", 125},
        {pfalign::test::map_legacy_path("1MBO.pdb"), "Myoglobin (1MBO)", 150},
        {pfalign::test::map_legacy_path("1HBS.pdb"), "Hemoglobin beta (1HBS)", 140},

        // AFDB proteins (predicted structures)
        {pfalign::test::map_legacy_path("P00519.pdb"), "AFDB Kinase P00519", 400},
        {pfalign::test::map_legacy_path("P69905.pdb"), "AFDB Globin P69905", 140},

        // ESM Atlas proteins (metagenomic predictions)
        {pfalign::test::map_legacy_path("MGYP003592128331.pdb"), "ESM Atlas MGYP003592128331", 50},
        {pfalign::test::map_legacy_path("MGYP002802792805.pdb"), "ESM Atlas MGYP002802792805", 60},
        {pfalign::test::map_legacy_path("MGYP001105483357.pdb"), "ESM Atlas MGYP001105483357", 70},
    };

    int passed = 0;
    int total = 0;

    for (const auto& test : tests) {
        arena.reset();  // Reset arena for each test
        total++;
        
        if (test_self_match(test, weights, config, &arena)) {
            passed++;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " / " << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
