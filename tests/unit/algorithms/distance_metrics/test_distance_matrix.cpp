/**
 * Unit tests for distance matrix computation.
 */

#include "pfalign/algorithms/distance_metrics/distance_matrix.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/common/arena_allocator.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace pfalign;
using namespace pfalign::msa;
using namespace pfalign::memory;

void test_distance_matrix_validation() {
    printf("Testing distance matrix validation...\n");

    int N = 3;
    float distances[9];

    // Valid matrix (all zeros)
    for (int i = 0; i < 9; i++) distances[i] = 0.0f;
    assert(validate_distance_matrix(distances, N));
    printf("  ✓ All-zero matrix is valid\n");

    // Valid symmetric matrix
    distances[0] = 0.0f; distances[1] = 0.3f; distances[2] = 0.5f;
    distances[3] = 0.3f; distances[4] = 0.0f; distances[5] = 0.4f;
    distances[6] = 0.5f; distances[7] = 0.4f; distances[8] = 0.0f;
    assert(validate_distance_matrix(distances, N));
    printf("  ✓ Valid symmetric matrix passes\n");

    // Invalid: non-zero diagonal
    distances[1 * N + 1] = 0.1f;
    assert(!validate_distance_matrix(distances, N));
    printf("  ✓ Non-zero diagonal rejected\n");
    distances[1 * N + 1] = 0.0f;  // Restore

    // Invalid: asymmetric
    distances[0 * N + 1] = 0.3f;
    distances[1 * N + 0] = 0.4f;
    assert(!validate_distance_matrix(distances, N));
    printf("  ✓ Asymmetric matrix rejected\n");
    distances[1 * N + 0] = 0.3f;  // Restore

    // Invalid: negative distance
    distances[0 * N + 1] = -0.1f;
    distances[1 * N + 0] = -0.1f;
    assert(!validate_distance_matrix(distances, N));
    printf("  ✓ Negative distance rejected\n");
    distances[0 * N + 1] = 0.3f;
    distances[1 * N + 0] = 0.3f;  // Restore

    // Invalid: exceeds 1.0
    distances[0 * N + 2] = 1.5f;
    distances[2 * N + 0] = 1.5f;
    assert(!validate_distance_matrix(distances, N));
    printf("  ✓ Distance > 1.0 rejected\n");

    printf("  ✓ Distance matrix validation works correctly\n");
}

void test_distance_matrix_single_sequence() {
    printf("Testing distance matrix with single sequence...\n");

    Arena cache_arena(10);   // 10 MB
    Arena scratch_arena(5);  // 5 MB

    SequenceCache cache(&cache_arena);

    int length = 10;
    int hidden_dim = 64;

    // Allocate and initialize SequenceEmbeddings manually (bypass MPNN)
    SequenceEmbeddings* seq = cache_arena.allocate<SequenceEmbeddings>(1);
    new (seq) SequenceEmbeddings();  // Placement new

    seq->length = length;
    seq->hidden_dim = hidden_dim;
    seq->identifier = "seq0";

    // Allocate and initialize embeddings
    seq->embeddings = cache_arena.allocate<float>(length * hidden_dim);
    for (int i = 0; i < length * hidden_dim; i++) {
        seq->embeddings[i] = 1.0f;  // Simple constant embeddings
    }

    // Allocate coords (required but not used in distance computation)
    seq->coords = cache_arena.allocate<float>(length * 4 * 3);
    for (int i = 0; i < length * 4 * 3; i++) {
        seq->coords[i] = 0.0f;
    }

    cache.add_sequence_from_embeddings(seq);

    // Compute distance matrix
    int N = 1;
    float distances[1];

    pfalign::smith_waterman::SWConfig sw_config;
    sw_config.affine = true;
    sw_config.gap_open = -1.0f;
    sw_config.gap_extend = -0.1f;
    sw_config.temperature = 1.0f;

    compute_distance_matrix_alignment(
        cache, sw_config, &scratch_arena, distances
    );

    // Verify: single sequence should have distance 0 to itself
    assert(std::abs(distances[0]) < 1e-6f);
    printf("  ✓ Single sequence has zero self-distance\n");

    // Validate matrix properties
    assert(validate_distance_matrix(distances, N));
    printf("  ✓ Single-sequence matrix is valid\n");
}

void test_distance_matrix_two_identical_sequences() {
    printf("Testing distance matrix with two identical sequences...\n");

    Arena cache_arena(10);
    Arena scratch_arena(5);

    SequenceCache cache(&cache_arena);

    int length = 10;
    int hidden_dim = 64;

    // Create two identical sequences
    for (int seq_idx = 0; seq_idx < 2; seq_idx++) {
        SequenceEmbeddings* seq = cache_arena.allocate<SequenceEmbeddings>(1);
        new (seq) SequenceEmbeddings();

        seq->length = length;
        seq->hidden_dim = hidden_dim;
        seq->identifier = (seq_idx == 0) ? "seq0" : "seq1";

        // Allocate and initialize embeddings (identical for both)
        seq->embeddings = cache_arena.allocate<float>(length * hidden_dim);
        for (int i = 0; i < length * hidden_dim; i++) {
            seq->embeddings[i] = 1.0f;
        }

        // Allocate coords
        seq->coords = cache_arena.allocate<float>(length * 4 * 3);
        for (int i = 0; i < length * 4 * 3; i++) {
            seq->coords[i] = 0.0f;
        }

        cache.add_sequence_from_embeddings(seq);
    }

    // Compute distance matrix
    int N = 2;
    float distances[4];

    pfalign::smith_waterman::SWConfig sw_config;
    sw_config.affine = true;
    sw_config.gap_open = -1.0f;
    sw_config.gap_extend = -0.1f;
    sw_config.temperature = 1.0f;

    compute_distance_matrix_alignment(
        cache, sw_config, &scratch_arena, distances
    );

    // Verify: identical sequences should have small distance
    // (May not be exactly 0 due to alignment, but should be close)
    printf("  Distance between identical sequences: %f\n", distances[0 * N + 1]);
    assert(distances[0 * N + 1] < 0.1f);  // Very small distance

    // Verify diagonal is zero
    assert(std::abs(distances[0]) < 1e-6f);
    assert(std::abs(distances[3]) < 1e-6f);

    // Verify symmetry
    assert(std::abs(distances[0 * N + 1] - distances[1 * N + 0]) < 1e-6f);

    printf("  ✓ Identical sequences have small distance\n");

    // Validate matrix properties
    assert(validate_distance_matrix(distances, N));
    printf("  ✓ Two-sequence matrix is valid\n");
}

void test_distance_matrix_properties() {
    printf("Testing distance matrix general properties...\n");

    Arena cache_arena(10);
    Arena scratch_arena(5);

    SequenceCache cache(&cache_arena);

    int N = 3;
    int lengths[3] = {10, 12, 15};
    int hidden_dim = 64;

    // Create three different sequences
    for (int seq_idx = 0; seq_idx < N; seq_idx++) {
        int length = lengths[seq_idx];

        SequenceEmbeddings* seq = cache_arena.allocate<SequenceEmbeddings>(1);
        new (seq) SequenceEmbeddings();

        seq->length = length;
        seq->hidden_dim = hidden_dim;

        char id_buf[16];
        snprintf(id_buf, sizeof(id_buf), "seq%d", seq_idx);
        seq->identifier = std::string(id_buf);

        // Allocate and initialize embeddings (different patterns)
        seq->embeddings = cache_arena.allocate<float>(length * hidden_dim);
        for (int i = 0; i < length * hidden_dim; i++) {
            seq->embeddings[i] = 1.0f + seq_idx * 0.1f + (i % 10) * 0.01f;
        }

        // Allocate coords
        seq->coords = cache_arena.allocate<float>(length * 4 * 3);
        for (int i = 0; i < length * 4 * 3; i++) {
            seq->coords[i] = 0.0f;
        }

        cache.add_sequence_from_embeddings(seq);
    }

    // Compute distance matrix
    float distances[9];

    pfalign::smith_waterman::SWConfig sw_config;
    sw_config.affine = true;
    sw_config.gap_open = -1.0f;
    sw_config.gap_extend = -0.1f;
    sw_config.temperature = 1.0f;

    compute_distance_matrix_alignment(
        cache, sw_config, &scratch_arena, distances
    );

    // Print distance matrix
    printf("  Distance matrix:\n");
    for (int i = 0; i < N; i++) {
        printf("    ");
        for (int j = 0; j < N; j++) {
            printf("%.4f ", distances[i * N + j]);
        }
        printf("\n");
    }

    // Verify all properties
    assert(validate_distance_matrix(distances, N));
    printf("  ✓ Distance matrix satisfies all properties\n");

    // Verify diagonal is zero
    for (int i = 0; i < N; i++) {
        assert(std::abs(distances[i * N + i]) < 1e-6f);
    }
    printf("  ✓ Diagonal is zero\n");

    // Verify symmetry
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(std::abs(distances[i * N + j] - distances[j * N + i]) < 1e-6f);
        }
    }
    printf("  ✓ Matrix is symmetric\n");

    // Verify bounds [0, 1]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(distances[i * N + j] >= 0.0f);
            assert(distances[i * N + j] <= 1.0f);
        }
    }
    printf("  ✓ All distances in [0, 1]\n");
}

int main() {
    printf("=== Distance Matrix Tests ===\n\n");

    test_distance_matrix_validation();
    test_distance_matrix_single_sequence();
    test_distance_matrix_two_identical_sequences();
    test_distance_matrix_properties();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
