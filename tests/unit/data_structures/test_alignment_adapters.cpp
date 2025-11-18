#include "pfalign/adapters/alignment_adapters.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <cassert>
#include <cstdio>
#include <cmath>

using namespace pfalign;
using namespace pfalign::adapters;
using namespace pfalign::memory;

// Helper function for float comparison
bool approx_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_simple_match_alignment() {
    printf("Testing simple match alignment...\\n");

    // Simple alignment: all matches, no gaps
    // Path: (0,0), (1,1), (2,2)
    AlignmentPair path[] = {
        {0, 0, 0.95f},
        {1, 1, 0.90f},
        {2, 2, 0.88f}
    };
    int path_length = 3;

    // Convert to columns
    msa::AlignmentColumn columns[3];
    pairwise_to_columns(path, path_length, 0, 1, columns);

    // Check column 0: both sequences at position 0
    assert(columns[0].positions.size() == 2);
    assert(columns[0].positions[0].seq_idx == 0);
    assert(columns[0].positions[0].pos == 0);
    assert(columns[0].positions[1].seq_idx == 1);
    assert(columns[0].positions[1].pos == 0);
    assert(!columns[0].positions[0].is_gap());
    assert(!columns[0].positions[1].is_gap());

    // Check column 1
    assert(columns[1].positions[0].pos == 1);
    assert(columns[1].positions[1].pos == 1);

    // Check column 2
    assert(columns[2].positions[0].pos == 2);
    assert(columns[2].positions[1].pos == 2);

    // Check gap statistics
    assert(columns[0].count_residues() == 2);
    assert(approx_equal(columns[0].gap_fraction(), 0.0f));

    printf("  ✓ Simple match alignment works\\n");
}

void test_alignment_with_gaps() {
    printf("Testing alignment with gaps...\\n");

    // Alignment with gaps:
    // Col 0: seq1[0] ↔ seq2[0]  (match)
    // Col 1: seq1[1] ↔ gap      (gap in seq2)
    // Col 2: gap ↔ seq2[1]      (gap in seq1)
    // Col 3: seq1[2] ↔ seq2[2]  (match)
    AlignmentPair path[] = {
        {0, 0, 0.90f},
        {1, -1, 0.0f},  // Gap in seq2
        {-1, 1, 0.0f},  // Gap in seq1
        {2, 2, 0.85f}
    };
    int path_length = 4;

    msa::AlignmentColumn columns[4];
    pairwise_to_columns(path, path_length, 0, 1, columns);

    // Check column 0: match
    assert(columns[0].positions[0].pos == 0);
    assert(columns[0].positions[1].pos == 0);
    assert(columns[0].count_residues() == 2);
    assert(approx_equal(columns[0].gap_fraction(), 0.0f));

    // Check column 1: gap in seq2
    assert(columns[1].positions[0].pos == 1);
    assert(columns[1].positions[1].pos == -1);
    assert(columns[1].positions[1].is_gap());
    assert(columns[1].count_residues() == 1);
    assert(approx_equal(columns[1].gap_fraction(), 0.5f));

    // Check column 2: gap in seq1
    assert(columns[2].positions[0].pos == -1);
    assert(columns[2].positions[0].is_gap());
    assert(columns[2].positions[1].pos == 1);
    assert(columns[2].count_residues() == 1);
    assert(approx_equal(columns[2].gap_fraction(), 0.5f));

    // Check column 3: match
    assert(columns[3].positions[0].pos == 2);
    assert(columns[3].positions[1].pos == 2);
    assert(columns[3].count_residues() == 2);

    printf("  ✓ Alignment with gaps works\\n");
}

void test_alignment_result_conversion() {
    printf("Testing AlignmentResult conversion...\\n");

    GrowableArena arena(1);  // ~1 MB (enough headroom for this test)

    // Create a mock AlignmentResult
    pairwise::AlignmentResult result;
    result.L1 = 3;
    result.L2 = 3;
    result.path_length = 4;
    result.max_path_length = 4;

    // Allocate and populate alignment path
    result.alignment_path = arena.allocate<AlignmentPair>(4);
    result.alignment_path[0] = {0, 0, 0.9f};
    result.alignment_path[1] = {1, -1, 0.0f};
    result.alignment_path[2] = {-1, 1, 0.0f};
    result.alignment_path[3] = {2, 2, 0.85f};

    // Convert to columns
    int aligned_length;
    msa::AlignmentColumn* columns = alignment_result_to_columns(
        result, 5, 7, &arena, &aligned_length
    );

    // Check aligned length
    assert(aligned_length == 4);

    // Check columns
    assert(columns[0].positions[0].seq_idx == 5);
    assert(columns[0].positions[1].seq_idx == 7);
    assert(columns[0].positions[0].pos == 0);
    assert(columns[0].positions[1].pos == 0);

    assert(columns[1].positions[0].pos == 1);
    assert(columns[1].positions[1].pos == -1);

    assert(columns[2].positions[0].pos == -1);
    assert(columns[2].positions[1].pos == 1);

    assert(columns[3].positions[0].pos == 2);
    assert(columns[3].positions[1].pos == 2);

    printf("  ✓ AlignmentResult conversion works\\n");
}

void test_validation_catches_errors() {
    printf("Testing validation catches errors...\\n");

    // Test 1: Invalid negative index (not -1)
    AlignmentPair bad_path1[] = {{0, -2, 0.9f}};
    try {
        validate_alignment_path(bad_path1, 1, 5, 5);
        assert(false && "Should have thrown for invalid index");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test 2: Both indices -1 (empty column)
    AlignmentPair bad_path2[] = {{-1, -1, 0.0f}};
    try {
        validate_alignment_path(bad_path2, 1, 5, 5);
        assert(false && "Should have thrown for empty column");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test 3: Out of bounds
    AlignmentPair bad_path3[] = {{5, 0, 0.9f}};  // i=5 >= L1=5
    try {
        validate_alignment_path(bad_path3, 1, 5, 5);
        assert(false && "Should have thrown for out of bounds");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test 4: Non-monotonic
    AlignmentPair bad_path4[] = {{2, 2, 0.9f}, {1, 3, 0.8f}};  // i decreases
    try {
        validate_alignment_path(bad_path4, 2, 5, 5);
        assert(false && "Should have thrown for non-monotonic");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test 5: Valid path should not throw
    AlignmentPair good_path[] = {{0, 0, 0.9f}, {1, -1, 0.0f}, {2, 1, 0.8f}};
    validate_alignment_path(good_path, 3, 3, 2);  // Should not throw

    printf("  ✓ Validation catches errors correctly\\n");
}

void test_empty_alignment() {
    printf("Testing empty alignment edge case...\\n");

    GrowableArena arena(1);  // ~1 MB

    // Create empty result
    pairwise::AlignmentResult result;
    result.L1 = 0;
    result.L2 = 0;
    result.path_length = 0;
    result.alignment_path = nullptr;

    // Should throw because path is null
    try {
        int aligned_length;
        alignment_result_to_columns(result, 0, 1, &arena, &aligned_length);
        assert(false && "Should have thrown for null path");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    printf("  ✓ Empty alignment handled correctly\\n");
}

void test_large_sequence_indices() {
    printf("Testing large sequence indices...\\n");

    // Test that sequence indices are preserved correctly
    AlignmentPair path[] = {
        {0, 0, 0.9f},
        {1, 1, 0.8f}
    };

    msa::AlignmentColumn columns[2];

    // Use large sequence indices
    pairwise_to_columns(path, 2, 42, 99, columns);

    assert(columns[0].positions[0].seq_idx == 42);
    assert(columns[0].positions[1].seq_idx == 99);
    assert(columns[1].positions[0].seq_idx == 42);
    assert(columns[1].positions[1].seq_idx == 99);

    printf("  ✓ Large sequence indices work\\n");
}

int main() {
    printf("=== Alignment Adapters Tests ===\\n\\n");

    test_simple_match_alignment();
    test_alignment_with_gaps();
    test_alignment_result_conversion();
    test_validation_catches_errors();
    test_empty_alignment();
    test_large_sequence_indices();

    printf("\\n=== All tests passed! ===\\n");
    return 0;
}
