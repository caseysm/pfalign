/**
 * Golden data tests for profile construction.
 *
 * Validates single-sequence profile creation against Python-generated golden data.
 */

#include "pfalign/modules/msa/profile.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include "pfalign/common/golden_data_test.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>

using namespace pfalign::msa;
using namespace pfalign::memory;
using namespace pfalign::testing;

namespace {

std::filesystem::path golden_root() {
    // Use compile-time project source root (passed by Meson)
    std::filesystem::path source_root(PFALIGN_SOURCE_ROOT);
    return source_root / "data" / "golden" / "profile" / "single_sequence";
}

} // namespace

/**
 * Test single-sequence profile creation for one protein.
 */
bool test_single_sequence_profile(const std::string& name, const std::string& data_dir) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    GoldenDataTest test(data_dir);
    GrowableArena arena(10);  // 10 MB

    // Load input embeddings
    auto [input_emb, emb_shape] = test.load_with_shape("input_embeddings.npy");

    if (emb_shape.size() != 2) {
        std::cerr << "ERROR: Expected 2D embeddings array" << std::endl;
        return false;
    }

    int length = emb_shape[0];
    int hidden_dim = emb_shape[1];

    std::cout << "\nTest parameters:" << std::endl;
    std::cout << "  Length: " << length << std::endl;
    std::cout << "  Hidden dim: " << hidden_dim << std::endl;

    // Note: Golden profile validation against .npz file skipped for now
    // TODO: Add .npz loading support to GoldenDataTest to enable full validation
    std::cout << "\nNote: Validating against basic properties (full .npz validation pending)" << std::endl;

    // Create profile via C++
    std::cout << "\nCreating profile via Profile::from_single_sequence()..." << std::endl;
    Profile* profile = Profile::from_single_sequence(
        input_emb.data(),
        length,
        hidden_dim,
        0,  // seq_idx
        &arena
    );
    std::cout << "  ✓ Profile created" << std::endl;

    // Validate basic properties
    std::cout << "\nValidating basic properties..." << std::endl;

    if (profile->length != length) {
        std::cerr << "  ✗ FAIL: Length mismatch: " << profile->length << " vs " << length << std::endl;
        return false;
    }
    std::cout << "  ✓ Length correct: " << profile->length << std::endl;

    if (profile->hidden_dim != hidden_dim) {
        std::cerr << "  ✗ FAIL: Hidden dim mismatch: " << profile->hidden_dim << " vs " << hidden_dim << std::endl;
        return false;
    }
    std::cout << "  ✓ Hidden dim correct: " << profile->hidden_dim << std::endl;

    if (profile->num_sequences != 1) {
        std::cerr << "  ✗ FAIL: Expected num_sequences=1, got " << profile->num_sequences << std::endl;
        return false;
    }
    std::cout << "  ✓ Num sequences correct: " << profile->num_sequences << std::endl;

    if (profile->seq_indices.size() != 1 || profile->seq_indices[0] != 0) {
        std::cerr << "  ✗ FAIL: seq_indices should be [0]" << std::endl;
        return false;
    }
    std::cout << "  ✓ Seq indices correct: [0]" << std::endl;

    // Validate embeddings (for single-sequence, output should match input)
    std::cout << "\nValidating embeddings..." << std::endl;
    std::vector<float> profile_emb(profile->embeddings, profile->embeddings + length * hidden_dim);
    bool embeddings_match = test.compare(
        "embeddings",
        input_emb,
        profile_emb,
        1e-6,  // atol
        1e-6   // rtol
    ).passed;

    if (!embeddings_match) {
        std::cerr << "  ✗ FAIL: Embeddings don't match input" << std::endl;
        return false;
    }
    std::cout << "  ✓ Embeddings match input" << std::endl;

    // Validate weights (should all be 1.0)
    std::cout << "\nValidating weights..." << std::endl;
    bool all_ones = true;
    for (int i = 0; i < length; ++i) {
        if (std::abs(profile->weights[i] - 1.0f) > 1e-6f) {
            all_ones = false;
            break;
        }
    }

    if (!all_ones) {
        std::cerr << "  ✗ FAIL: Not all weights are 1.0 for single sequence" << std::endl;
        return false;
    }
    std::cout << "  ✓ All weights = 1.0 (correct for single sequence)" << std::endl;

    // Validate sum_norm (should be unit-normalized embeddings for single sequence)
    std::cout << "\nValidating sum_norm..." << std::endl;
    bool sum_norm_valid = true;
    for (int i = 0; i < length; ++i) {
        float norm_sq = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            float val = profile->sum_norm[i * hidden_dim + d];
            norm_sq += val * val;
        }
        float norm = std::sqrt(norm_sq);
        if (std::abs(norm - 1.0f) > 1e-5f) {
            std::cerr << "  ✗ FAIL: sum_norm not unit-normalized at position " << i
                      << " (L2 norm = " << norm << ")" << std::endl;
            sum_norm_valid = false;
            break;
        }
    }

    if (!sum_norm_valid) {
        return false;
    }
    std::cout << "  ✓ sum_norm is unit-normalized (correct for single sequence)" << std::endl;

    // Validate gap_counts (should all be 0)
    std::cout << "\nValidating gap_counts..." << std::endl;
    bool all_zero_gaps = true;
    for (int i = 0; i < length; ++i) {
        if (profile->gap_counts[i] != 0) {
            all_zero_gaps = false;
            break;
        }
    }

    if (!all_zero_gaps) {
        std::cerr << "  ✗ FAIL: Not all gap_counts are 0 for single sequence" << std::endl;
        return false;
    }
    std::cout << "  ✓ All gap_counts = 0 (correct for single sequence)" << std::endl;

    // Validate alignment columns
    std::cout << "\nValidating alignment columns..." << std::endl;
    bool columns_valid = true;
    for (int i = 0; i < length; ++i) {
        const AlignmentColumn& col = profile->columns[i];
        if (col.positions.size() != 1) {
            std::cerr << "  ✗ FAIL: Column " << i << " should have 1 position, has " << col.positions.size() << std::endl;
            columns_valid = false;
            break;
        }
        if (col.positions[0].seq_idx != 0 || col.positions[0].pos != i) {
            std::cerr << "  ✗ FAIL: Column " << i << " has wrong position: ("
                      << col.positions[0].seq_idx << ", " << col.positions[0].pos << ")" << std::endl;
            columns_valid = false;
            break;
        }
    }

    if (!columns_valid) {
        return false;
    }
    std::cout << "  ✓ All alignment columns correct" << std::endl;

    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✅ PASSED: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return true;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Profile Construction Golden Data Tests                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

    auto root = golden_root();
    std::cout << "\nGolden data root: " << root << std::endl;

    // Test cases: 3 single-sequence profiles
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"1CRN (Crambin, 46 residues)", (root / "1CRN").string()},
        {"1UBQ (Ubiquitin, 76 residues)", (root / "1UBQ").string()},
        {"1VII (Villin, 36 residues)", (root / "1VII").string()}
    };

    int passed = 0;
    int total = test_cases.size();

    for (const auto& [name, data_dir] : test_cases) {
        if (test_single_sequence_profile(name, data_dir)) {
            passed++;
        } else {
            std::cerr << "\n❌ FAILED: " << name << std::endl;
        }
    }

    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Summary: " << passed << "/" << total << " passed" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return (passed == total) ? 0 : 1;
}
