/**
 * MPNN Cache Adapter Implementation
 */

#include "mpnn_cache_adapter.h"
#include "pfalign/dispatch/backend_traits.h"
#include <stdexcept>

namespace pfalign {
namespace mpnn {

void MPNNCacheAdapter::add_protein(int seq_id, const float* coords, int length,
                                   const std::string& identifier, const std::string& sequence) {
    if (coords == nullptr) {
        throw std::invalid_argument("Coordinates pointer is null");
    }

    if (length <= 0) {
        throw std::invalid_argument("Sequence length must be positive");
    }

    int hidden_dim = config_.hidden_dim;

    // Allocate embedding buffer from arena
    float* embeddings =
        arena_->allocate<float>(static_cast<size_t>(length) * static_cast<size_t>(hidden_dim));

    // Create MPNN workspace
    pfalign::mpnn::MPNNWorkspace workspace(length, config_.k_neighbors, hidden_dim);

    // Initialize workspace (residue indices, chain labels)
    for (int i = 0; i < length; i++) {
        workspace.residue_idx[i] = i;
        workspace.chain_labels[i] = 0;  // Single chain
    }

    // Encode with MPNN (using references to weights/config)
    pfalign::mpnn::mpnn_forward<pfalign::ScalarBackend>(coords, length,
                                                        weights_,  // Reference member
                                                        config_,   // Reference member
                                                        embeddings, &workspace);

    // Store in cache via generic method (encoder-agnostic!)
    cache_.add_precomputed(seq_id, embeddings, length, hidden_dim, coords, identifier, sequence);
}

}  // namespace mpnn
}  // namespace pfalign
