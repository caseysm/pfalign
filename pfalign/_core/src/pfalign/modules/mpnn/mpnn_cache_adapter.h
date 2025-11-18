/**
 * MPNN Cache Adapter
 *
 * Adapter pattern to decouple SequenceCache from MPNN encoder.
 *
 * Problem:
 * - SequenceCache hard-coded to MPNN (content coupling)
 * - Can't swap MPNN for other encoders (ESM, ProtBERT, etc.)
 * - Violates dependency inversion principle
 *
 * Solution:
 * - MPNNCacheAdapter: Encode proteins with MPNN and add to cache
 * - SequenceCache: Generic add_precomputed() method (encoder-agnostic)
 * - Clean separation: cache doesn't know about MPNN, adapter does
 *
 * Usage:
 *   Arena arena(100 * 1024 * 1024);
 *   SequenceCache cache(&arena);
 *   MPNNCacheAdapter adapter(cache, weights, config, &arena);
 *
 *   // Add proteins (MPNN encoding happens in adapter)
 *   adapter.add_protein(0, coords1, L1, "1CRN_A");
 *   adapter.add_protein(1, coords2, L2, "2CI2_I");
 *
 *   // Retrieve from cache (adapter not needed)
 *   const SequenceEmbeddings* seq = cache.get(0);
 *
 * Benefits:
 * - SequenceCache encoder-agnostic (can use ESM, ProtBERT, etc.)
 * - Clean tier separation (cache in Tier 1, adapter in Tier 3)
 * - Can swap encoders without changing cache
 *
 * CRITICAL: Stores REFERENCES to weights/config (not copies!)
 * - MPNNWeights deletes copy constructor (manages raw buffers)
 * - Copying would cause double-free on destruction
 * - References ensure adapter doesn't own weights (caller owns them)
 */

#pragma once

#include "pfalign/modules/mpnn/sequence_cache.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/common/growable_arena.h"

namespace pfalign {
namespace mpnn {

/**
 * Adapter for populating SequenceCache with MPNN encodings.
 *
 * Uses Adapter pattern (GoF) to decouple SequenceCache from MPNN.
 * Stores REFERENCES to weights/config to avoid copy issues.
 */
class MPNNCacheAdapter {
public:
    /**
     * Create MPNN cache adapter.
     *
     * IMPORTANT: Stores references to weights and config.
     * Caller must ensure weights and config outlive this adapter.
     *
     * @param cache     SequenceCache to populate
     * @param weights   MPNN weights (stored by REFERENCE)
     * @param config    MPNN configuration (stored by REFERENCE)
     * @param arena     Arena for allocating embeddings
     */
    MPNNCacheAdapter(SequenceCache& cache, const pfalign::mpnn::MPNNWeights& weights,
                     const pfalign::mpnn::MPNNConfig& config, pfalign::memory::GrowableArena* arena)
        : cache_(cache),
          weights_(weights)  // Reference member initialization
          ,
          config_(config)  // Reference member initialization
          ,
          arena_(arena) {
    }

    /**
     * Add protein and encode with MPNN.
     *
     * Computes MPNN embeddings and stores them in cache via add_precomputed().
     *
     * @param seq_id        Unique sequence identifier
     * @param coords        Input coordinates [length * 4 * 3]
     * @param length        Sequence length
     * @param identifier    Optional sequence ID string (e.g., "1CRN_A")
     */
    void add_protein(int seq_id, const float* coords, int length,
                     const std::string& identifier = "", const std::string& sequence = "");

private:
    SequenceCache& cache_;                       // Reference to cache
    const pfalign::mpnn::MPNNWeights& weights_;  // Reference (NOT copy!)
    const pfalign::mpnn::MPNNConfig& config_;    // Reference (NOT copy!)
    pfalign::memory::GrowableArena* arena_;      // Pointer for allocations
};

}  // namespace mpnn
}  // namespace pfalign
