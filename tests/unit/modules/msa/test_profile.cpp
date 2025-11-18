#include "pfalign/modules/msa/profile.h"
#include "pfalign/common/arena_allocator.h"
#include "pfalign/common/growable_arena.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace pfalign::msa;
using namespace pfalign::memory;

// Test helpers
bool approx_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_alignment_column() {
    printf("Testing AlignmentColumn...\n");

    AlignmentColumn col(3);
    col.positions[0] = {0, 5};   // seq 0, pos 5
    col.positions[1] = {1, -1};  // seq 1, gap
    col.positions[2] = {2, 10};  // seq 2, pos 10

    assert(col.count_residues() == 2);
    assert(approx_equal(col.gap_fraction(), 1.0f / 3.0f));

    printf("  ✓ AlignmentColumn works correctly\n");
}

void test_profile_from_single_sequence() {
    printf("Testing Profile::from_single_sequence...\n");

    GrowableArena arena(1);  // 1MB arena

    // Create test embeddings (length=5, hidden_dim=4)
    int length = 5;
    int hidden_dim = 4;
    float embeddings[20] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // pos 0
        0.0f, 1.0f, 0.0f, 0.0f,  // pos 1
        0.0f, 0.0f, 1.0f, 0.0f,  // pos 2
        0.0f, 0.0f, 0.0f, 1.0f,  // pos 3
        0.5f, 0.5f, 0.0f, 0.0f   // pos 4
    };

    Profile* profile = Profile::from_single_sequence(embeddings, length, hidden_dim, 0, &arena);

    // Check basic properties
    assert(profile->length == 5);
    assert(profile->num_sequences == 1);
    assert(profile->hidden_dim == 4);
    assert(profile->seq_indices.size() == 1);
    assert(profile->seq_indices[0] == 0);

    // Check embeddings were copied correctly
    for (int i = 0; i < length * hidden_dim; ++i) {
        assert(approx_equal(profile->embeddings[i], embeddings[i]));
    }

    // Check alignment columns
    for (int i = 0; i < length; ++i) {
        assert(profile->columns[i].positions.size() == 1);
        assert(profile->columns[i].positions[0].seq_idx == 0);
        assert(profile->columns[i].positions[0].pos == i);
        assert(!profile->columns[i].positions[0].is_gap());
    }

    // Check gap counts (should be 0 for single sequence)
    for (int i = 0; i < length; ++i) {
        assert(profile->gap_counts[i] == 0);
        assert(approx_equal(profile->weights[i], 1.0f));
    }

    // Test helper methods
    assert(!profile->has_gaps(0));
    assert(approx_equal(profile->gap_fraction(0), 0.0f));
    assert(profile->residue_count(0) == 1);

    printf("  ✓ Single sequence profile created correctly\n");
}

void test_profile_from_alignment() {
    printf("Testing Profile::from_alignment...\n");

    GrowableArena arena(1);

    // Create two simple profiles
    int hidden_dim = 4;

    // Profile 1: length 3, 1 sequence
    float emb1[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // pos 0
        0.0f, 1.0f, 0.0f, 0.0f,  // pos 1
        0.0f, 0.0f, 1.0f, 0.0f   // pos 2
    };
    Profile* profile1 = Profile::from_single_sequence(emb1, 3, hidden_dim, 0, &arena);

    // Profile 2: length 3, 1 sequence
    float emb2[12] = {
        0.0f, 0.0f, 0.0f, 1.0f,  // pos 0
        0.5f, 0.5f, 0.0f, 0.0f,  // pos 1
        0.0f, 0.0f, 0.5f, 0.5f   // pos 2
    };
    Profile* profile2 = Profile::from_single_sequence(emb2, 3, hidden_dim, 1, &arena);

    // Create simple alignment (no gaps, perfect match)
    // Aligned columns: (0,0), (1,1), (2,2)
    AlignmentColumn alignment[3];
    for (int i = 0; i < 3; ++i) {
        alignment[i].positions.resize(2);
        alignment[i].positions[0] = {0, i};  // seq 0 from profile1
        alignment[i].positions[1] = {1, i};  // seq 1 from profile2
    }

    // Merge profiles
    Profile* merged = Profile::from_alignment(*profile1, *profile2, alignment, 3, &arena);

    // Check basic properties
    assert(merged->length == 3);
    assert(merged->num_sequences == 2);
    assert(merged->hidden_dim == 4);
    assert(merged->seq_indices.size() == 2);
    assert(merged->seq_indices[0] == 0);
    assert(merged->seq_indices[1] == 1);

    // Check merged embeddings (should be average of profile1 and profile2)
    for (int col = 0; col < 3; ++col) {
        const float* merged_emb = merged->get_embedding(col);
        const float* emb_1 = profile1->get_embedding(col);
        const float* emb_2 = profile2->get_embedding(col);

        for (int d = 0; d < hidden_dim; ++d) {
            float expected = (emb_1[d] + emb_2[d]) / 2.0f;
            assert(approx_equal(merged_emb[d], expected));
        }
    }

    // Check gap counts (no gaps in this alignment)
    for (int i = 0; i < 3; ++i) {
        assert(merged->gap_counts[i] == 0);
        assert(approx_equal(merged->weights[i], 2.0f));  // Two residues contribute
    }

    printf("  ✓ Profile merging works correctly (no gaps)\n");
}

void test_profile_with_gaps() {
    printf("Testing Profile with gaps...\n");

    GrowableArena arena(1);
    int hidden_dim = 4;

    // Profile 1: length 2
    float emb1[8] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };
    Profile* profile1 = Profile::from_single_sequence(emb1, 2, hidden_dim, 0, &arena);

    // Profile 2: length 2
    float emb2[8] = {
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    Profile* profile2 = Profile::from_single_sequence(emb2, 2, hidden_dim, 1, &arena);

    // Create alignment with gaps
    // Col 0: profile1[0], gap
    // Col 1: profile1[1], profile2[0]
    // Col 2: gap, profile2[1]
    AlignmentColumn alignment[3];

    alignment[0].positions.resize(2);
    alignment[0].positions[0] = {0, 0};    // profile1, pos 0
    alignment[0].positions[1] = {1, -1};   // gap

    alignment[1].positions.resize(2);
    alignment[1].positions[0] = {0, 1};    // profile1, pos 1
    alignment[1].positions[1] = {1, 0};    // profile2, pos 0

    alignment[2].positions.resize(2);
    alignment[2].positions[0] = {0, -1};   // gap
    alignment[2].positions[1] = {1, 1};    // profile2, pos 1

    Profile* merged = Profile::from_alignment(*profile1, *profile2, alignment, 3, &arena);

    // Check gap counts
    assert(merged->gap_counts[0] == 1);  // One gap in col 0
    assert(merged->gap_counts[1] == 0);  // No gaps in col 1
    assert(merged->gap_counts[2] == 1);  // One gap in col 2

    // Check weights (number of contributing residues)
    assert(approx_equal(merged->weights[0], 1.0f));  // Only one residue contributes
    assert(approx_equal(merged->weights[1], 2.0f));  // Two residues contribute
    assert(approx_equal(merged->weights[2], 1.0f));  // Only one residue contributes

    // Check embeddings
    // Col 0: only profile1[0] = [1, 0, 0, 0]
    const float* merged_emb0 = merged->get_embedding(0);
    assert(approx_equal(merged_emb0[0], 1.0f));
    assert(approx_equal(merged_emb0[1], 0.0f));

    // Col 1: average of profile1[1] and profile2[0]
    // [0, 1, 0, 0] + [0, 0, 1, 0] = [0, 0.5, 0.5, 0]
    const float* merged_emb1 = merged->get_embedding(1);
    assert(approx_equal(merged_emb1[1], 0.5f));
    assert(approx_equal(merged_emb1[2], 0.5f));

    // Col 2: only profile2[1] = [0, 0, 0, 1]
    const float* merged_emb2 = merged->get_embedding(2);
    assert(approx_equal(merged_emb2[3], 1.0f));

    printf("  ✓ Profile with gaps works correctly\n");
}

int main() {
    printf("=== MSA Profile Tests ===\n\n");

    test_alignment_column();
    test_profile_from_single_sequence();
    test_profile_from_alignment();
    test_profile_with_gaps();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
