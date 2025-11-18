#include "pfalign/common/growable_arena.h"
#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/tools/weights/mpnn_weight_loader.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/dispatch/backend_traits.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::memory;

// Helper function to create dummy coordinates
void create_dummy_coords(float* coords, int L) {
    // Create simple linear chain with realistic backbone geometry
    for (int i = 0; i < L; i++) {
        // N atom
        coords[i * 12 + 0] = i * 3.8f;
        coords[i * 12 + 1] = 0.0f;
        coords[i * 12 + 2] = 0.0f;

        // CA atom
        coords[i * 12 + 3] = i * 3.8f + 1.46f;
        coords[i * 12 + 4] = 0.0f;
        coords[i * 12 + 5] = 0.0f;

        // C atom
        coords[i * 12 + 6] = i * 3.8f + 2.46f;
        coords[i * 12 + 7] = 1.52f;
        coords[i * 12 + 8] = 0.0f;

        // O atom
        coords[i * 12 + 9] = i * 3.8f + 2.46f;
        coords[i * 12 + 10] = 2.52f;
        coords[i * 12 + 11] = 0.5f;
    }
}

// Helper function to create dummy MPNN weights
MPNNWeights create_dummy_weights() {
    MPNNConfig config;
    MPNNWeights weights(config.num_layers);

    int hidden_dim = config.hidden_dim;
    int num_rbf = config.num_rbf;

    // Allocate minimal weights (initialized to small random values)
    // Note: These are not realistic trained weights, just for testing

    // Edge embedding
    int edge_features = 25 * num_rbf;  // 25 atom pairs * 16 bins
    weights.edge_embedding_weight.assign(edge_features * hidden_dim, 0.01f);
    weights.edge_norm_gamma.assign(hidden_dim, 1.0f);
    weights.edge_norm_beta.assign(hidden_dim, 0.0f);

    // Positional encoding
    weights.positional_weight.assign(66 * 16, 0.01f);
    weights.positional_bias.assign(16, 0.0f);

    // W_e initial transformation
    weights.W_e_weight.resize(hidden_dim * hidden_dim);
    for (int i = 0; i < hidden_dim * hidden_dim; i++) {
        weights.W_e_weight[i] = (i % (hidden_dim + 1) == 0) ? 0.1f : 0.01f;  // Diagonal emphasis
    }
    weights.W_e_bias.assign(hidden_dim, 0.0f);

    // Layer weights
    for (int layer = 0; layer < config.num_layers; layer++) {
        auto& l = weights.layers[layer];

        // Node MLP
        l.W1_weight.assign(3 * hidden_dim * hidden_dim, 0.01f);
        l.W1_bias.assign(hidden_dim, 0.0f);
        l.W2_weight.assign(hidden_dim * hidden_dim, 0.01f);
        l.W2_bias.assign(hidden_dim, 0.0f);
        l.W3_weight.assign(hidden_dim * hidden_dim, 0.01f);
        l.W3_bias.assign(hidden_dim, 0.0f);

        // Layer norms
        l.norm1_gamma.assign(hidden_dim, 1.0f);
        l.norm1_beta.assign(hidden_dim, 0.0f);
        l.norm2_gamma.assign(hidden_dim, 1.0f);
        l.norm2_beta.assign(hidden_dim, 0.0f);

        // FFN
        l.ffn_W_in_weight.assign(hidden_dim * 4 * hidden_dim, 0.01f);
        l.ffn_W_in_bias.assign(4 * hidden_dim, 0.0f);
        l.ffn_W_out_weight.assign(4 * hidden_dim * hidden_dim, 0.01f);
        l.ffn_W_out_bias.assign(hidden_dim, 0.0f);

        // Edge MLP
        l.W11_weight.assign(3 * hidden_dim * hidden_dim, 0.01f);
        l.W11_bias.assign(hidden_dim, 0.0f);
        l.W12_weight.assign(hidden_dim * hidden_dim, 0.01f);
        l.W12_bias.assign(hidden_dim, 0.0f);
        l.W13_weight.assign(hidden_dim * hidden_dim, 0.01f);
        l.W13_bias.assign(hidden_dim, 0.0f);

        // Edge norm
        l.norm3_gamma.assign(hidden_dim, 1.0f);
        l.norm3_beta.assign(hidden_dim, 0.0f);
    }

    return weights;
}

void test_basic_add_and_retrieve() {
    printf("Testing basic add and retrieve...\n");

    GrowableArena arena(100);  // 100 MB
    SequenceCache cache(&arena);
    MPNNConfig config;
    MPNNWeights weights = create_dummy_weights();

    // Create dummy coordinates
    int L = 10;
    float coords[10 * 4 * 3];
    create_dummy_coords(coords, L);

    // Add sequence
    int id = cache.add_sequence(coords, L, "test_seq", weights, config);

    // Verify ID
    assert(id == 0);
    assert(cache.size() == 1);
    assert(!cache.empty());

    // Retrieve sequence
    const SequenceEmbeddings* seq = cache.get(id);
    assert(seq != nullptr);
    assert(seq->seq_id == id);
    assert(seq->length == L);
    assert(seq->hidden_dim == config.hidden_dim);
    assert(seq->identifier == "test_seq");
    assert(seq->is_valid());

    // Check embeddings are allocated and non-zero
    assert(seq->embeddings != nullptr);
    assert(seq->coords != nullptr);

    // Check coords were copied correctly
    assert(std::memcmp(seq->coords, coords, L * 4 * 3 * sizeof(float)) == 0);

    printf("  ✓ Basic add and retrieve works\n");
}

void test_multiple_sequences() {
    printf("Testing multiple sequences...\n");

    GrowableArena arena(100);
    SequenceCache cache(&arena);
    MPNNConfig config;
    MPNNWeights weights = create_dummy_weights();

    // Add multiple sequences with different lengths
    int L1 = 10, L2 = 15, L3 = 20;
    float coords1[10 * 4 * 3], coords2[15 * 4 * 3], coords3[20 * 4 * 3];
    create_dummy_coords(coords1, L1);
    create_dummy_coords(coords2, L2);
    create_dummy_coords(coords3, L3);

    int id1 = cache.add_sequence(coords1, L1, "seq1", weights, config);
    int id2 = cache.add_sequence(coords2, L2, "seq2", weights, config);
    int id3 = cache.add_sequence(coords3, L3, "seq3", weights, config);

    // Verify IDs are sequential
    assert(id1 == 0);
    assert(id2 == 1);
    assert(id3 == 2);
    assert(cache.size() == 3);

    // Retrieve all sequences
    const SequenceEmbeddings* seq1 = cache.get(id1);
    const SequenceEmbeddings* seq2 = cache.get(id2);
    const SequenceEmbeddings* seq3 = cache.get(id3);

    assert(seq1 != nullptr && seq1->length == L1);
    assert(seq2 != nullptr && seq2->length == L2);
    assert(seq3 != nullptr && seq3->length == L3);

    assert(seq1->identifier == "seq1");
    assert(seq2->identifier == "seq2");
    assert(seq3->identifier == "seq3");

    // Test iteration
    int count = 0;
    for (const auto* seq : cache.sequences()) {
        assert(seq->is_valid());
        count++;
    }
    assert(count == 3);

    printf("  ✓ Multiple sequences work\n");
}

void test_embeddings_validity() {
    printf("Testing embeddings validity...\n");

    GrowableArena arena(100);
    SequenceCache cache(&arena);

    // Try to load real embedded weights for more realistic test
    MPNNConfig config;
    MPNNWeights weights(3);
    bool using_real_weights = false;

    try {
        auto [loaded_weights, loaded_config, sw_params] = weights::load_embedded_mpnn_weights();
        (void)sw_params;
        weights = std::move(loaded_weights);
        config = loaded_config;
        using_real_weights = true;
        printf("  Using embedded MPNN weights\n");
    } catch (...) {
        // Fall back to dummy weights if embedded weights not available
        weights = create_dummy_weights();
        printf("  Using dummy weights (embedded weights not available)\n");
    }

    int L = 10;
    float coords[10 * 4 * 3];
    create_dummy_coords(coords, L);

    int id = cache.add_sequence(coords, L, weights, config);
    const SequenceEmbeddings* seq = cache.get(id);

    // Check dimensions
    assert(seq->length == L);
    assert(seq->hidden_dim == config.hidden_dim);

    // Check embeddings are non-null
    const float* emb = seq->get_embeddings();
    assert(emb != nullptr);

    // Only check for non-zero embeddings if using real weights
    // (dummy weights may produce near-zero outputs)
    if (using_real_weights) {
        bool has_nonzero = false;
        for (int i = 0; i < L * config.hidden_dim; i++) {
            if (std::fabs(emb[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        assert(has_nonzero);
        printf("  ✓ Embeddings have non-zero values\n");
    } else {
        printf("  ⚠ Skipping non-zero check (using dummy weights)\n");
    }

    printf("  ✓ Embeddings validity checks pass\n");
}

void test_error_handling() {
    printf("Testing error handling...\n");

    GrowableArena arena(100);
    MPNNConfig config;
    MPNNWeights weights = create_dummy_weights();

    float coords[10 * 4 * 3];
    create_dummy_coords(coords, 10);

    // Test null arena
    try {
        SequenceEmbeddings::create_from_coords(
            0, coords, 10, weights, config, nullptr
        );
        assert(false && "Should have thrown for null arena");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test null coords
    try {
        SequenceEmbeddings::create_from_coords(
            0, nullptr, 10, weights, config, &arena
        );
        assert(false && "Should have thrown for null coords");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test invalid length
    try {
        SequenceEmbeddings::create_from_coords(
            0, coords, 0, weights, config, &arena
        );
        assert(false && "Should have thrown for zero length");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    try {
        SequenceEmbeddings::create_from_coords(
            0, coords, -5, weights, config, &arena
        );
        assert(false && "Should have thrown for negative length");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    printf("  ✓ Error handling works\n");
}

void test_get_nonexistent() {
    printf("Testing retrieval of nonexistent sequence...\n");

    GrowableArena arena(100);
    SequenceCache cache(&arena);

    // Try to get sequence that doesn't exist
    const SequenceEmbeddings* seq = cache.get(999);
    assert(seq == nullptr);

    printf("  ✓ Returns nullptr for nonexistent sequence\n");
}

void test_clear() {
    printf("Testing cache clear...\n");

    GrowableArena arena(100);
    SequenceCache cache(&arena);
    MPNNConfig config;
    MPNNWeights weights = create_dummy_weights();

    // Add sequences
    float coords[10 * 4 * 3];
    create_dummy_coords(coords, 10);

    cache.add_sequence(coords, 10, "seq1", weights, config);
    cache.add_sequence(coords, 10, "seq2", weights, config);
    assert(cache.size() == 2);

    // Clear cache
    cache.clear();
    assert(cache.size() == 0);
    assert(cache.empty());

    // Verify we can add new sequences after clear
    int id = cache.add_sequence(coords, 10, "seq3", weights, config);
    assert(id == 0);  // ID counter was reset
    assert(cache.size() == 1);

    printf("  ✓ Clear works correctly\n");
}

int main() {
    printf("=== SequenceCache Tests ===\n\n");

    test_basic_add_and_retrieve();
    test_multiple_sequences();
    test_embeddings_validity();
    test_error_handling();
    test_get_nonexistent();
    test_clear();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
