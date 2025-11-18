/**
 * MPNN Type Forward Declarations
 *
 * This header provides forward declarations for MPNN types to avoid
 * circular dependencies and layer violations.
 *
 * Problem:
 * - data_structures/sequence_cache.h (Layer 2.5) needs MPNNWeights/MPNNConfig types
 * - modules/mpnn/mpnn_encoder.h (Layer 2) defines these types
 * - Layer 2.5 should NOT depend on Layer 2 (wrong direction)
 *
 * Solution:
 * - This header (Layer 0) provides forward declarations only
 * - data_structures headers include this (no full dependency)
 * - data_structures .cpp files include full mpnn_encoder.h when needed
 *
 * Benefits:
 * - Proper layer separation
 * - Reduced compile-time dependencies
 * - data_structures can be used without pulling in MPNN module
 *
 * Usage:
 *   // In headers (.h files):
 *   #include "pfalign/common/mpnn_types.h"
 *   void func(const pfalign::mpnn::MPNNWeights& weights);  // OK: forward decl
 *
 *   // In implementation (.cpp files):
 *   #include "pfalign/modules/mpnn/mpnn_encoder.h"  // Full definition when needed
 *   void func(const pfalign::mpnn::MPNNWeights& weights) {
 *       // Can access weight members here
 *   }
 */

#pragma once

namespace pfalign {
namespace mpnn {

// Forward declarations for MPNN types
struct MPNNWeights;
struct MPNNConfig;
struct MPNNWorkspace;

}  // namespace mpnn
}  // namespace pfalign
