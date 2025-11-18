#include "sequence_cache.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"  // For deprecated add_sequence() methods
#include "pfalign/dispatch/backend_traits.h"
#include <cstring>
#include <stdexcept>

namespace pfalign {

// ============================================================================
// SequenceEmbeddings Factory Methods
// ============================================================================

SequenceEmbeddings* SequenceEmbeddings::create_from_coords(
    int seq_id, const float* coords_data, int length, const pfalign::mpnn::MPNNWeights& weights,
    const pfalign::mpnn::MPNNConfig& config, pfalign::memory::GrowableArena* arena,
    const std::string& sequence) {
    return create_from_coords(seq_id, coords_data, length, "", weights, config, arena, sequence);
}

SequenceEmbeddings* SequenceEmbeddings::create_from_coords(
    int seq_id, const float* coords_data, int length, const std::string& identifier,
    const pfalign::mpnn::MPNNWeights& weights, const pfalign::mpnn::MPNNConfig& config,
    pfalign::memory::GrowableArena* arena, const std::string& sequence) {
    if (arena == nullptr) {
        throw std::invalid_argument("Arena pointer is null");
    }

    if (coords_data == nullptr) {
        throw std::invalid_argument("Coordinates pointer is null");
    }

    if (length <= 0) {
        throw std::invalid_argument("Sequence length must be positive");
    }

    if (!sequence.empty() && static_cast<int>(sequence.size()) != length) {
        throw std::invalid_argument("Provided sequence length does not match coordinates length");
    }

    // Allocate SequenceEmbeddings struct from arena
    SequenceEmbeddings* seq = arena->allocate<SequenceEmbeddings>(static_cast<size_t>(1));

    // Use placement new to initialize
    new (seq) SequenceEmbeddings();

    // Set basic properties
    seq->seq_id = seq_id;
    seq->length = length;
    seq->hidden_dim = config.hidden_dim;
    seq->identifier = identifier;
    seq->sequence = sequence;

    // Allocate and copy coordinates [L * 4 * 3]
    int coords_size = length * 4 * 3;
    seq->coords = arena->allocate<float>(static_cast<size_t>(coords_size));
    std::memcpy(seq->coords, coords_data, static_cast<size_t>(coords_size) * sizeof(float));

    // Allocate embeddings buffer [L * hidden_dim]
    int emb_size = length * config.hidden_dim;
    seq->embeddings = arena->allocate<float>(static_cast<size_t>(emb_size));

    // Create MPNN workspace for computing embeddings
    // Note: MPNNWorkspace allocates its own buffers (not from arena)
    pfalign::mpnn::MPNNWorkspace workspace(length, config.k_neighbors, config.hidden_dim,
                                           config.num_rbf);

    // Compute MPNN embeddings
    // Using ScalarBackend as the default; could be templated if needed
    pfalign::mpnn::mpnn_forward<pfalign::ScalarBackend>(seq->coords, length, weights, config,
                                                        seq->embeddings, &workspace);

    return seq;
}

// ============================================================================
// SequenceCache Implementation
// ============================================================================

void SequenceCache::add_precomputed(int seq_id, const float* embeddings, int length, int hidden_dim,
                                    const float* coords, const std::string& identifier,
                                    const std::string& sequence) {
    if (embeddings == nullptr) {
        throw std::invalid_argument("Embeddings pointer is null");
    }

    if (length <= 0) {
        throw std::invalid_argument("Sequence length must be positive");
    }

    if (hidden_dim <= 0) {
        throw std::invalid_argument("Hidden dimension must be positive");
    }

    if (!sequence.empty() && static_cast<int>(sequence.size()) != length) {
        throw std::invalid_argument("Provided sequence length does not match embeddings length");
    }

    // CRITICAL: Lock must protect ALL arena allocations and vector operations
    // Arena is shared across threads, so concurrent allocations will corrupt memory
    std::lock_guard<std::mutex> lock(mutex_);

    // Allocate SequenceEmbeddings struct from arena
    SequenceEmbeddings* seq = arena_->allocate<SequenceEmbeddings>(static_cast<size_t>(1));

    // Use placement new to initialize (required for std::string)
    new (seq) SequenceEmbeddings();

    // Set properties
    seq->seq_id = seq_id;
    seq->length = length;
    seq->hidden_dim = hidden_dim;
    seq->identifier = identifier;
    seq->sequence = sequence;

    // Copy embeddings to arena
    int emb_size = length * hidden_dim;
    seq->embeddings = arena_->allocate<float>(static_cast<size_t>(emb_size));
    std::memcpy(seq->embeddings, embeddings, static_cast<size_t>(emb_size) * sizeof(float));

    // Copy coordinates if provided
    if (coords != nullptr) {
        int coords_size = length * 4 * 3;
        seq->coords = arena_->allocate<float>(static_cast<size_t>(coords_size));
        std::memcpy(seq->coords, coords, static_cast<size_t>(coords_size) * sizeof(float));
    } else {
        seq->coords = nullptr;
    }

    // Add to collection
    sequences_.push_back(seq);

    // Update next_id if necessary
    if (seq_id >= next_id_) {
        next_id_ = seq_id + 1;
    }
}

int SequenceCache::add_sequence(const float* coords, int length, const std::string& identifier,
                                const pfalign::mpnn::MPNNWeights& weights,
                                const pfalign::mpnn::MPNNConfig& config,
                                const std::string& sequence) {
    // Lock for thread-safe arena allocation, ID assignment, and vector modification
    // Note: create_from_coords allocates from arena, so must be inside lock
    std::lock_guard<std::mutex> lock(mutex_);

    // Create SequenceEmbeddings with next available ID
    int seq_id = next_id_++;

    SequenceEmbeddings* seq = SequenceEmbeddings::create_from_coords(
        seq_id, coords, length, identifier, weights, config, arena_, sequence);

    // Add to collection
    sequences_.push_back(seq);

    return seq_id;
}

int SequenceCache::add_sequence(const float* coords, int length,
                                const pfalign::mpnn::MPNNWeights& weights,
                                const pfalign::mpnn::MPNNConfig& config) {
    // Delegate to version with identifier (empty string)
    // Suppress deprecation warning for internal delegation
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return add_sequence(coords, length, "", weights, config);
    #pragma GCC diagnostic pop
}

const SequenceEmbeddings* SequenceCache::get(int seq_id) const {
    // Linear search through sequences
    // This is fine for typical MSA sizes (< 1000 sequences)
    for (const auto* seq : sequences_) {
        if (seq->seq_id == seq_id) {
            return seq;
        }
    }
    return nullptr;
}

void SequenceCache::clear() {
    // CRITICAL: Destroy std::string members before clearing
    // SequenceEmbeddings contains std::string (identifier, sequence)
    // which must be explicitly destructed to avoid heap leaks
    //
    // NOTE: We destroy the objects here to release std::string heap allocations,
    // but the arena memory itself is not reclaimed until arena reset.
    // The destructed memory should NOT be accessed again.
    for (SequenceEmbeddings* seq : sequences_) {
        if (seq != nullptr) {
            seq->~SequenceEmbeddings();
        }
    }

    // Clear the vector (no more pointers to worry about)
    sequences_.clear();

    // Reset next_id counter
    next_id_ = 0;

    // Note: Arena memory is not freed here.
    // Caller should reset the arena if they want to reclaim memory.
}

int SequenceCache::add_sequence_from_embeddings(SequenceEmbeddings* seq) {
    if (seq == nullptr) {
        throw std::invalid_argument("SequenceEmbeddings pointer is null");
    }

    // Lock for thread-safe ID assignment and vector modification
    std::lock_guard<std::mutex> lock(mutex_);

    // Assign next available ID
    int seq_id = next_id_++;
    seq->seq_id = seq_id;

    // Add to collection
    sequences_.push_back(seq);

    return seq_id;
}

int SequenceCache::max_length() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sequences_.empty()) {
        return 0;
    }

    int max_len = 0;
    for (const auto* seq : sequences_) {
        if (seq->length > max_len) {
            max_len = seq->length;
        }
    }
    return max_len;
}

int SequenceCache::hidden_dim() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sequences_.empty()) {
        return 0;
    }

    // Return hidden_dim from first sequence
    // (All sequences should have same hidden_dim when using same MPNN weights)
    return sequences_[0]->hidden_dim;
}

}  // namespace pfalign
