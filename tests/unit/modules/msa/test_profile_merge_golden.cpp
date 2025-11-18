/**
 * Golden data tests for profile merging.
 *
 * Validates profile-profile merging (with and without gaps) against Python-generated golden data.
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

std::filesystem::path golden_root(const std::string& merge_type) {
    // Use compile-time project source root (passed by Meson)
    std::filesystem::path source_root(PFALIGN_SOURCE_ROOT);
    return source_root / "data" / "golden" / "profile" / merge_type;
}

/**
 * Load a profile from individual .npy files.
 */
Profile* load_profile_from_npy(const std::string& data_dir, const std::string& prefix, GrowableArena* arena) {
    GoldenDataTest test(data_dir);
    std::filesystem::path dir_path(data_dir);

    // Load scalar fields first to get dimensions
    auto length_data = test.load(prefix + "_length.npy");
    auto hidden_dim_data = test.load(prefix + "_hidden_dim.npy");
    auto num_seq_data = test.load(prefix + "_num_sequences.npy");

    int length = static_cast<int>(length_data[0]);
    int hidden_dim = static_cast<int>(hidden_dim_data[0]);
    int num_sequences = static_cast<int>(num_seq_data[0]);

    // Load array fields
    auto emb = test.load(prefix + "_embeddings.npy");
    auto weights = test.load(prefix + "_weights.npy");
    auto sum_norm = test.load(prefix + "_sum_norm.npy");
    auto gap_counts_data = test.load(prefix + "_gap_counts.npy");
    auto seq_indices_data = test.load(prefix + "_seq_indices.npy");

    // Allocate profile from heap (not arena) because it contains std::vector
    // Profile constructor will allocate arrays from arena
    Profile* profile = new Profile(length, num_sequences, hidden_dim, arena);

    // Copy data into already-allocated arrays
    std::memcpy(profile->embeddings, emb.data(), length * hidden_dim * sizeof(float));
    std::memcpy(profile->weights, weights.data(), length * sizeof(float));
    std::memcpy(profile->sum_norm, sum_norm.data(), length * hidden_dim * sizeof(float));

    // Copy gap_counts (convert float to int) into already-allocated array
    for (int i = 0; i < length; ++i) {
        profile->gap_counts[i] = static_cast<int>(gap_counts_data[i]);
    }

    // Copy seq_indices
    profile->seq_indices.clear();
    for (size_t i = 0; i < seq_indices_data.size(); ++i) {
        profile->seq_indices.push_back(static_cast<int>(seq_indices_data[i]));
    }

    // Note: alignment columns not loaded (not needed for merge input)
    // They will be reconstructed during merge

    return profile;
}

/**
 * Load alignment from int32 .npy file with shape [num_cols, max_seqs, 2].
 */
std::vector<AlignmentColumn> load_alignment(const std::string& npy_path, int num_sequences) {
    GoldenDataTest test("");
    auto [data, shape] = test.load_with_shape(npy_path);

    if (shape.size() != 3 || shape[2] != 2) {
        std::cerr << "ERROR: Expected 3D alignment array with shape [num_cols, max_seqs, 2]" << std::endl;
        return {};
    }

    int num_cols = shape[0];
    int max_seqs = shape[1];

    std::vector<AlignmentColumn> alignment;
    alignment.reserve(num_cols);

    for (int col = 0; col < num_cols; ++col) {
        AlignmentColumn column(num_sequences);

        for (int seq = 0; seq < max_seqs; ++seq) {
            // data was converted from int32 to float by loader
            int offset = (col * max_seqs + seq) * 2;
            int32_t seq_idx = static_cast<int32_t>(data[offset]);
            int32_t pos = static_cast<int32_t>(data[offset + 1]);

            if (seq_idx >= 0 && seq_idx < num_sequences) {
                column.positions[seq_idx] = {seq_idx, pos};
            }
        }

        alignment.push_back(column);
    }

    return alignment;
}

} // namespace

/**
 * Test profile merging for one test case.
 */
bool test_merge_case(const std::string& name, const std::string& data_dir) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    GoldenDataTest test(data_dir);
    GrowableArena arena(20);  // 20 MB

    // Load input profiles
    std::cout << "\nLoading input profiles..." << std::endl;
    Profile* profile1 = load_profile_from_npy(data_dir, "input_profile1", &arena);
    Profile* profile2 = load_profile_from_npy(data_dir, "input_profile2", &arena);

    if (!profile1 || !profile2) {
        std::cerr << "ERROR: Failed to load input profiles" << std::endl;
        return false;
    }

    std::cout << "  Profile 1: length=" << profile1->length << ", num_sequences=" << profile1->num_sequences << std::endl;
    std::cout << "  Profile 2: length=" << profile2->length << ", num_sequences=" << profile2->num_sequences << std::endl;

    // Load alignment
    std::cout << "\nLoading alignment..." << std::endl;
    int total_sequences = profile1->num_sequences + profile2->num_sequences;
    std::filesystem::path test_path(data_dir);
    auto alignment = load_alignment((test_path / "input_alignment.npy").string(), total_sequences);

    if (alignment.empty()) {
        std::cerr << "ERROR: Failed to load alignment" << std::endl;
        return false;
    }

    int aligned_length = alignment.size();
    std::cout << "  Aligned length: " << aligned_length << std::endl;

    // Count gaps
    int gap_columns = 0;
    for (const auto& col : alignment) {
        for (const auto& pos : col.positions) {
            if (pos.is_gap()) {
                gap_columns++;
                break;
            }
        }
    }
    std::cout << "  Gap columns: " << gap_columns << std::endl;

    // Load golden output
    std::cout << "\nLoading golden output..." << std::endl;
    Profile* golden_profile = load_profile_from_npy(data_dir, "output_profile", &arena);
    auto golden_ecs = test.load("output_ecs.npy");

    if (!golden_profile || golden_ecs.empty()) {
        std::cerr << "ERROR: Failed to load golden output" << std::endl;
        return false;
    }

    float expected_ecs = golden_ecs[0];
    std::cout << "  Expected ECS: " << expected_ecs << std::endl;

    // Perform merge
    std::cout << "\nMerging profiles via Profile::from_alignment()..." << std::endl;
    Profile* merged = Profile::from_alignment(
        *profile1,
        *profile2,
        alignment.data(),
        aligned_length,
        &arena
    );

    if (!merged) {
        std::cerr << "ERROR: Profile::from_alignment() returned nullptr" << std::endl;
        return false;
    }

    std::cout << "  ✓ Merge complete" << std::endl;
    std::cout << "  Merged profile: length=" << merged->length << ", num_sequences=" << merged->num_sequences << std::endl;

    // Validate basic properties
    std::cout << "\nValidating basic properties..." << std::endl;

    if (merged->length != aligned_length) {
        std::cerr << "  ✗ FAIL: Length mismatch: " << merged->length << " vs " << aligned_length << std::endl;
        return false;
    }
    std::cout << "  ✓ Length correct: " << merged->length << std::endl;

    if (merged->num_sequences != total_sequences) {
        std::cerr << "  ✗ FAIL: Num sequences mismatch: " << merged->num_sequences << " vs " << total_sequences << std::endl;
        return false;
    }
    std::cout << "  ✓ Num sequences correct: " << merged->num_sequences << std::endl;

    // Validate embeddings
    std::cout << "\nValidating merged embeddings..." << std::endl;
    std::vector<float> golden_emb(golden_profile->embeddings,
                                   golden_profile->embeddings + aligned_length * merged->hidden_dim);
    std::vector<float> merged_emb(merged->embeddings,
                                   merged->embeddings + aligned_length * merged->hidden_dim);
    bool embeddings_match = test.compare(
        "embeddings",
        golden_emb,
        merged_emb,
        1e-5,  // atol
        1e-5   // rtol
    ).passed;

    if (!embeddings_match) {
        std::cerr << "  ✗ FAIL: Embeddings don't match golden data" << std::endl;
        // Show first mismatch
        for (int i = 0; i < std::min(5, aligned_length); ++i) {
            std::cout << "  Column " << i << " embeddings[0]: "
                      << merged->embeddings[i * merged->hidden_dim]
                      << " vs golden " << golden_profile->embeddings[i * merged->hidden_dim] << std::endl;
        }
        return false;
    }
    std::cout << "  ✓ Embeddings match golden data" << std::endl;

    // Validate weights
    std::cout << "\nValidating weights..." << std::endl;
    std::vector<float> golden_weights(golden_profile->weights,
                                       golden_profile->weights + aligned_length);
    std::vector<float> merged_weights(merged->weights,
                                       merged->weights + aligned_length);
    bool weights_match = test.compare(
        "weights",
        golden_weights,
        merged_weights,
        1e-5,
        1e-5
    ).passed;

    if (!weights_match) {
        std::cerr << "  ✗ FAIL: Weights don't match golden data" << std::endl;
        return false;
    }
    std::cout << "  ✓ Weights match golden data" << std::endl;

    // Validate sum_norm
    std::cout << "\nValidating sum_norm..." << std::endl;
    std::vector<float> golden_sum_norm(golden_profile->sum_norm,
                                        golden_profile->sum_norm + aligned_length * merged->hidden_dim);
    std::vector<float> merged_sum_norm(merged->sum_norm,
                                        merged->sum_norm + aligned_length * merged->hidden_dim);
    bool sum_norm_match = test.compare(
        "sum_norm",
        golden_sum_norm,
        merged_sum_norm,
        1e-4,  // Slightly larger tolerance for accumulated normalization
        1e-4
    ).passed;

    if (!sum_norm_match) {
        std::cerr << "  ✗ FAIL: sum_norm doesn't match golden data" << std::endl;
        return false;
    }
    std::cout << "  ✓ sum_norm matches golden data" << std::endl;

    // Validate gap_counts
    std::cout << "\nValidating gap_counts..." << std::endl;
    bool gap_counts_match = true;
    for (int i = 0; i < aligned_length; ++i) {
        if (merged->gap_counts[i] != golden_profile->gap_counts[i]) {
            std::cerr << "  ✗ FAIL: gap_count mismatch at column " << i << ": "
                      << merged->gap_counts[i] << " vs " << golden_profile->gap_counts[i] << std::endl;
            gap_counts_match = false;
            break;
        }
    }

    if (!gap_counts_match) {
        return false;
    }
    std::cout << "  ✓ gap_counts match golden data" << std::endl;

    // Validate ECS
    std::cout << "\nValidating ECS..." << std::endl;
    float computed_ecs = merged->compute_ecs();
    float ecs_diff = std::abs(computed_ecs - expected_ecs);

    std::cout << "  Computed ECS: " << computed_ecs << std::endl;
    std::cout << "  Expected ECS: " << expected_ecs << std::endl;
    std::cout << "  Difference:   " << ecs_diff << std::endl;

    if (ecs_diff > 1e-4f) {
        std::cerr << "  ✗ FAIL: ECS mismatch (tolerance 1e-4)" << std::endl;
        return false;
    }
    std::cout << "  ✓ ECS matches golden data" << std::endl;

    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✅ PASSED: " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return true;
}

int main() {
    std::cout << "+============================================================+" << std::endl;
    std::cout << "|       Profile Merging Golden Data Tests                    |" << std::endl;
    std::cout << "+============================================================+" << std::endl;

    // Test merge without gaps
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PHASE 1: Merging without gaps" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    auto no_gaps_root = golden_root("merge_no_gaps");
    std::vector<std::pair<std::string, std::string>> no_gaps_cases = {
        {"Crambin + Ubiquitin (perfect alignment)", (no_gaps_root / "crambin_ubiquitin").string()},
        {"Crambin + Crambin (self-merge, identity)", (no_gaps_root / "crambin_self").string()},
        {"Crambin + Villin (shorter sequences)", (no_gaps_root / "crambin_villin").string()}
    };

    int passed = 0;
    int total = 0;

    for (const auto& [name, data_dir] : no_gaps_cases) {
        total++;
        if (test_merge_case(name, data_dir)) {
            passed++;
        } else {
            std::cerr << "\n❌ FAILED: " << name << std::endl;
        }
    }

    // Test merge with gaps
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "PHASE 2: Merging with gaps" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    auto with_gaps_root = golden_root("merge_with_gaps");
    std::vector<std::pair<std::string, std::string>> with_gaps_cases = {
        {"Crambin + Ubiquitin (5 indels)", (with_gaps_root / "crambin_ubiquitin_indels").string()},
        {"Crambin + Ubiquitin (terminal gaps)", (with_gaps_root / "crambin_ubiquitin_terminal").string()},
        {"Crambin + Villin (mixed gaps)", (with_gaps_root / "crambin_villin_mixed").string()}
    };

    for (const auto& [name, data_dir] : with_gaps_cases) {
        total++;
        if (test_merge_case(name, data_dir)) {
            passed++;
        } else {
            std::cerr << "\n❌ FAILED: " << name << std::endl;
        }
    }

    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Summary: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "  No gaps: " << std::min(passed, 3) << "/3" << std::endl;
    std::cout << "  With gaps: " << std::max(0, passed - 3) << "/3" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    return (passed == total) ? 0 : 1;
}
