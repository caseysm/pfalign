/**
 * Sequence Embeddings Cache
 *
 * Caches precomputed MPNN embeddings for protein sequences to avoid
 * redundant encoding during progressive MSA.
 *
 * Problem:
 * - MSA runs dozens of pairwise alignments on the same sequences
 * - MPNN encoding is expensive (~500ms for L=200)
 * - Need to compute embeddings once and reuse them
 *
 * Solution:
 * - SequenceEmbeddings: Stores coords + embeddings for one sequence
 * - SequenceCache: Manages collection of cached sequences
 * - Arena-based allocation for automatic cleanup
 *
 * Usage:
 *   Arena arena(100 * 1024 * 1024);  // 100 MB
 *   SequenceCache cache(&arena);
 *
 *   // Add sequences (MPNN encoding happens once)
 *   int id1 = cache.add_sequence(coords1, L1, "1CRN_A", weights, config);
 *   int id2 = cache.add_sequence(coords2, L2, "2CI2_I", weights, config);
 *
 *   // Retrieve cached embeddings (fast!)
 *   const SequenceEmbeddings* seq1 = cache.get(id1);
 *   similarity::compute_similarity<ScalarBackend>(
 *       seq1->get_embeddings(),
 *       cache.get(id2)->get_embeddings(),
 *       similarity_matrix, L1, L2, hidden_dim
 *   );
 */

#pragma once

#include <string>
#include <vector>
#include <mutex>
#include "pfalign/common/growable_arena.h"
#include "mpnn_encoder.h"

namespace pfalign {

/**
 * Cached MPNN embeddings for a single protein sequence.
 *
 * Stores backbone coordinates and precomputed MPNN embeddings.
 * Used to avoid redundant MPNN encoding during MSA.
 *
 * Memory layout:
 * - coords: [L * 4 * 3] floats (N, CA, C, O backbone atoms)
 * - embeddings: [L * hidden_dim] floats (MPNN output)
 */
struct SequenceEmbeddings {
    int seq_id;      // Unique sequence identifier
    int length;      // Sequence length
    int hidden_dim;  // Embedding dimension (typically 64)

    // Owned data (allocated from arena)
    float* coords;      // [L * 4 * 3] backbone coordinates
    float* embeddings;  // [L * hidden_dim] MPNN embeddings

    // Metadata (optional)
    std::string identifier;  // Sequence ID (e.g., "1CRN_A")
    std::string sequence;    // One-letter amino-acid sequence (optional)

    SequenceEmbeddings()
        : seq_id(-1), length(0), hidden_dim(0), coords(nullptr), embeddings(nullptr) {
    }

    /**
     * Create SequenceEmbeddings from coordinates.
     *
     * Allocates storage from arena and computes MPNN embeddings.
     *
     * @param seq_id        Unique sequence identifier
     * @param coords_data   Input coordinates [L * 4 * 3]
     * @param length        Sequence length
     * @param weights       MPNN weights
     * @param config        MPNN configuration
     * @param arena         Arena for allocation
     * @return              Pointer to allocated SequenceEmbeddings
     */
    static SequenceEmbeddings* create_from_coords(int seq_id, const float* coords_data, int length,
                                                  const pfalign::mpnn::MPNNWeights& weights,
                                                  const pfalign::mpnn::MPNNConfig& config,
                                                  pfalign::memory::GrowableArena* arena,
                                                  const std::string& sequence = "");

    /**
     * Create SequenceEmbeddings with identifier.
     *
     * Same as create_from_coords but also sets identifier string.
     *
     * @param seq_id        Unique sequence identifier
     * @param coords_data   Input coordinates [L * 4 * 3]
     * @param length        Sequence length
     * @param identifier    Sequence ID string (e.g., "1CRN_A")
     * @param weights       MPNN weights
     * @param config        MPNN configuration
     * @param arena         Arena for allocation
     * @return              Pointer to allocated SequenceEmbeddings
     */
    static SequenceEmbeddings* create_from_coords(int seq_id, const float* coords_data, int length,
                                                  const std::string& identifier,
                                                  const pfalign::mpnn::MPNNWeights& weights,
                                                  const pfalign::mpnn::MPNNConfig& config,
                                                  pfalign::memory::GrowableArena* arena,
                                                  const std::string& sequence = "");

    /**
     * Get embeddings pointer (for similarity computation).
     *
     * @return Pointer to embeddings [L * hidden_dim]
     */
    const float* get_embeddings() const {
        return embeddings;
    }

    /**
     * Get coordinates pointer.
     *
     * @return Pointer to coordinates [L * 4 * 3]
     */
    const float* get_coords() const {
        return coords;
    }

    /**
     * Check if embeddings are valid.
     *
     * @return True if embeddings have been computed
     */
    bool is_valid() const {
        return coords != nullptr && embeddings != nullptr && length > 0;
    }
};

/**
 * Cache of sequence embeddings for MSA.
 *
 * Manages a collection of precomputed MPNN embeddings to avoid redundant
 * encoding during progressive alignment.
 *
 * Features:
 * - Add sequences and compute embeddings once
 * - Retrieve cached embeddings by seq_id
 * - Iterate over all cached sequences
 * - Arena-based allocation for automatic cleanup
 *
 * Usage:
 *   SequenceCache cache(&arena);
 *
 *   // Add sequences
 *   int id1 = cache.add_sequence(coords1, L1, "1CRN_A", weights, config);
 *   int id2 = cache.add_sequence(coords2, L2, "2CI2_I", weights, config);
 *
 *   // Retrieve
 *   const SequenceEmbeddings* seq = cache.get(id1);
 *   const float* emb = seq->get_embeddings();
 *
 *   // Iterate
 *   for (const auto* seq : cache.sequences()) {
 *       printf("Seq %d: %s (L=%d)\n", seq->seq_id,
 *              seq->identifier.c_str(), seq->length);
 *   }
 */
class SequenceCache {
public:
    /**
     * Create sequence cache.
     *
     * @param arena Arena for allocating sequence embeddings
     */
    explicit SequenceCache(pfalign::memory::GrowableArena* arena) : arena_(arena), next_id_(0) {
    }

    /**
     * Destructor - ensures std::string members are properly destroyed.
     *
     * CRITICAL: SequenceEmbeddings contains std::string members (identifier, sequence)
     * that must be explicitly destructed to avoid heap leaks when using arena allocation.
     */
    ~SequenceCache() {
        destroy_sequences();
    }

    /**
     * Add precomputed embeddings (generic, encoder-agnostic).
     *
     * NEW: Use this with encoder adapters (MPNNCacheAdapter, ESMCacheAdapter, etc.)
     *
     * This method is encoder-agnostic and simply stores precomputed embeddings
     * without knowing how they were generated. Use encoder-specific adapters
     * (like MPNNCacheAdapter) to compute and add embeddings.
     *
     * @param seq_id        Unique sequence identifier
     * @param embeddings    Precomputed embeddings [length * hidden_dim]
     * @param length        Sequence length
     * @param hidden_dim    Embedding dimension
     * @param coords        Optional coordinates [length * 4 * 3]
     * @param identifier    Optional sequence ID string
     */
    void add_precomputed(int seq_id, const float* embeddings, int length, int hidden_dim,
                         const float* coords = nullptr, const std::string& identifier = "",
                         const std::string& sequence = "");

    /**
     * Add sequence and compute embeddings with MPNN.
     *
     * DEPRECATED: Use MPNNCacheAdapter::add_protein() instead.
     * This method is kept for backward compatibility but couples SequenceCache
     * to MPNN. New code should use the adapter pattern.
     *
     * @param coords        Input coordinates [L * 4 * 3]
     * @param length        Sequence length
     * @param identifier    Sequence ID string (e.g., "1CRN_A")
     * @param weights       MPNN weights
     * @param config        MPNN configuration
     * @return              Unique sequence ID for retrieval
     */
    [[deprecated("Use MPNNCacheAdapter::add_protein() instead")]]
    int add_sequence(const float* coords, int length, const std::string& identifier,
                     const pfalign::mpnn::MPNNWeights& weights,
                     const pfalign::mpnn::MPNNConfig& config, const std::string& sequence = "");

    /**
     * Add sequence without identifier (DEPRECATED).
     *
     * DEPRECATED: Use MPNNCacheAdapter::add_protein() instead.
     *
     * @param coords        Input coordinates [L * 4 * 3]
     * @param length        Sequence length
     * @param weights       MPNN weights
     * @param config        MPNN configuration
     * @return              Unique sequence ID for retrieval
     */
    [[deprecated("Use MPNNCacheAdapter::add_protein() instead")]]
    int add_sequence(const float* coords, int length, const pfalign::mpnn::MPNNWeights& weights,
                     const pfalign::mpnn::MPNNConfig& config);

    /**
     * Retrieve cached embeddings by seq_id.
     *
     * @param seq_id        Sequence ID (from add_sequence)
     * @return              Pointer to SequenceEmbeddings, or nullptr if not found
     */
    const SequenceEmbeddings* get(int seq_id) const;

    /**
     * Get number of cached sequences.
     *
     * Thread-safe: Acquires mutex to safely read sequences_.size().
     *
     * @return Number of sequences in cache
     */
    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(sequences_.size());
    }

    /**
     * Check if cache is empty.
     *
     * Thread-safe: Acquires mutex to safely read sequences_.empty().
     *
     * @return True if no sequences cached
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sequences_.empty();
    }

    /**
     * Iterate over all cached sequences.
     *
     * Thread-safe: Returns a copy of the sequences vector to avoid races.
     * Note: Callers should cache this result if iterating multiple times.
     *
     * @return Copy of vector of pointers to all SequenceEmbeddings
     */
    std::vector<SequenceEmbeddings*> sequences() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sequences_;  // Return copy
    }

    /**
     * Clear cache (reset arena).
     *
     * WARNING: Invalidates all pointers returned by get().
     * Only call this if you're done with all cached sequences.
     */
    void clear();

    /**
     * Add pre-computed embeddings (for testing).
     *
     * Bypasses MPNN encoding by directly adding pre-allocated SequenceEmbeddings.
     * Useful for unit tests that don't want to run expensive MPNN forward passes.
     *
     * @param seq           Pre-allocated SequenceEmbeddings (arena-allocated)
     * @return              Sequence ID assigned to this sequence
     */
    int add_sequence_from_embeddings(SequenceEmbeddings* seq);

    /**
     * Get maximum sequence length in cache.
     *
     * @return Maximum length, or 0 if cache is empty
     */
    int max_length() const;

    /**
     * Get embedding hidden dimension.
     *
     * Assumes all sequences have the same hidden_dim (which is always true
     * when using the same MPNN weights).
     *
     * @return Hidden dimension, or 0 if cache is empty
     */
    int hidden_dim() const;

private:
    /**
     * Helper: Destroy all SequenceEmbeddings objects.
     *
     * CRITICAL: Must be called before clearing sequences_ or destroying cache
     * to release std::string heap allocations.
     */
    void destroy_sequences() {
        for (SequenceEmbeddings* seq : sequences_) {
            if (seq != nullptr) {
                seq->~SequenceEmbeddings();
            }
        }
    }

    pfalign::memory::GrowableArena* arena_;
    std::vector<SequenceEmbeddings*> sequences_;
    int next_id_;
    mutable std::mutex mutex_;  // Thread-safe access to sequences_
};

}  // namespace pfalign
