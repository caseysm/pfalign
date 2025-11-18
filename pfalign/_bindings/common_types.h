/**
 * Common types shared across PyBind11 binding modules.
 */

#pragma once

#include "pfalign/tools/weights/mpnn_weight_loader.h"
#include "pfalign/tools/weights/safetensors_loader.h"

namespace pfalign {
namespace bindings {

/**
 * Wrapper structure to hold MPNN weights + config together.
 *
 * This is necessary because MPNNWeights and MPNNConfig are loaded together
 * from the safetensors file, and both are needed for encoding proteins.
 */
struct MPNNWeightsWithConfig {
    mpnn::MPNNWeights weights;
    mpnn::MPNNConfig config;
    weights::SWParams sw_params;

    MPNNWeightsWithConfig(mpnn::MPNNWeights&& w, mpnn::MPNNConfig c, weights::SWParams p)
        : weights(std::move(w)), config(c), sw_params(p) {}

    // Disable copy (MPNNWeights is move-only)
    MPNNWeightsWithConfig(const MPNNWeightsWithConfig&) = delete;
    MPNNWeightsWithConfig& operator=(const MPNNWeightsWithConfig&) = delete;

    // Enable move
    MPNNWeightsWithConfig(MPNNWeightsWithConfig&&) = default;
    MPNNWeightsWithConfig& operator=(MPNNWeightsWithConfig&&) = default;
};

}  // namespace bindings
}  // namespace pfalign
