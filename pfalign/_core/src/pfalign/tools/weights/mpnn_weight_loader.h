/**
 * MPNN weight loader using safetensors format
 *
 * Expected tensor names in safetensors file:
 *   - edge_embedding.weight       [edge_features, hidden_dim]
 *   - edge_embedding.norm.weight  [hidden_dim]
 *   - edge_embedding.norm.bias    [hidden_dim]
 *   - layers.{i}.W1.weight        [3*hidden_dim, hidden_dim]  # Message input: [h_i, h_j, e_ij]
 *   - layers.{i}.W1.bias          [hidden_dim]
 *   - layers.{i}.W2.weight        [hidden_dim, hidden_dim]
 *   - layers.{i}.W2.bias          [hidden_dim]
 *   - layers.{i}.W3.weight        [hidden_dim, hidden_dim]
 *   - layers.{i}.W3.bias          [hidden_dim]
 *   - layers.{i}.norm1.weight     [hidden_dim]
 *   - layers.{i}.norm1.bias       [hidden_dim]
 *   - layers.{i}.norm2.weight     [hidden_dim]
 *   - layers.{i}.norm2.bias       [hidden_dim]
 *   - layers.{i}.ffn.W_in.weight  [hidden_dim, 4*hidden_dim]
 *   - layers.{i}.ffn.W_in.bias    [4*hidden_dim]
 *   - layers.{i}.ffn.W_out.weight [4*hidden_dim, hidden_dim]
 *   - layers.{i}.ffn.W_out.bias   [hidden_dim]
 *   - layers.{i}.norm3.weight     [hidden_dim]
 *   - layers.{i}.norm3.bias       [hidden_dim]
 *
 * Optional Smith-Waterman parameters (scalars):
 *   - gap                         [1]  Gap extension penalty
 *   - gap_open                    [1]  Gap opening penalty
 *   - temperature                 [1]  Softmax temperature
 *
 * Config is inferred from tensor shapes.
 */

#pragma once

#include "safetensors_loader.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include <sstream>

namespace pfalign {
namespace weights {

/**
 * Smith-Waterman parameters loaded from model file.
 */
struct SWParams {
    float gap;          // Gap extension penalty (default: 0.194)
    float gap_open;     // Gap opening penalty (default: -2.544)
    float temperature;  // Softmax temperature (default: 1.0)

    SWParams() : gap(0.194f), gap_open(-2.544f), temperature(1.0f) {
    }
};

class MPNNWeightLoader {
public:
    /**
     * Load MPNN weights from safetensors file.
     *
     * Returns weights, config, and Smith-Waterman parameters.
     */
    static std::tuple<mpnn::MPNNWeights, mpnn::MPNNConfig, SWParams>
    load(const std::string& filepath) {
        SafetensorsLoader loader(filepath);

        // Infer config from tensor shapes
        mpnn::MPNNConfig config = infer_config(loader);

        // Allocate weights
        mpnn::MPNNWeights weights(config.num_layers);

        // Load edge embedding
        load_edge_embedding(loader, weights, config);

        // Load layer weights
        for (int i = 0; i < config.num_layers; i++) {
            load_layer(loader, weights, config, i);
        }

        // Load SW parameters (optional, uses defaults if not present)
        SWParams sw_params = load_sw_params(loader);

        return {std::move(weights), config, sw_params};
    }

    /**
     * Load MPNN weights from memory buffer.
     *
     * Loads from embedded weights or other in-memory SafeTensors data.
     * Same return value as load() but no file I/O required.
     *
     * @param buffer SafeTensors data in memory
     * @return Tuple of (weights, config, sw_params)
     */
    static std::tuple<mpnn::MPNNWeights, mpnn::MPNNConfig, SWParams>
    load_from_buffer(const std::vector<uint8_t>& buffer) {
        SafetensorsLoader loader(buffer.data(), buffer.size());

        // Infer config from tensor shapes
        mpnn::MPNNConfig config = infer_config(loader);

        // Allocate weights
        mpnn::MPNNWeights weights(config.num_layers);

        // Load edge embedding
        load_edge_embedding(loader, weights, config);

        // Load layer weights
        for (int i = 0; i < config.num_layers; i++) {
            load_layer(loader, weights, config, i);
        }

        // Load SW parameters
        SWParams sw_params = load_sw_params(loader);

        return {std::move(weights), config, sw_params};
    }

    /**
     * Load MPNN weights from safetensors file with known config.
     *
     * Validates that file matches expected config.
     */
    static std::pair<mpnn::MPNNWeights, SWParams> load_with_config(const std::string& filepath,
                                                                   const mpnn::MPNNConfig& config) {
        SafetensorsLoader loader(filepath);

        // Validate config matches file
        mpnn::MPNNConfig file_config = infer_config(loader);
        validate_config_match(config, file_config);

        // Allocate weights
        mpnn::MPNNWeights weights(config.num_layers);

        // Load edge embedding
        load_edge_embedding(loader, weights, config);

        // Load layer weights
        for (int i = 0; i < config.num_layers; i++) {
            load_layer(loader, weights, config, i);
        }

        // Load SW parameters
        SWParams sw_params = load_sw_params(loader);

        return {std::move(weights), sw_params};
    }

private:
    static mpnn::MPNNConfig infer_config(const SafetensorsLoader& loader) {
        mpnn::MPNNConfig config;

        // Get hidden_dim from edge_embedding.norm.weight shape
        if (!loader.has_tensor("edge_embedding.norm.weight")) {
            throw std::runtime_error("Missing edge_embedding.norm.weight");
        }
        auto norm_info = loader.get_info("edge_embedding.norm.weight");
        if (norm_info.shape.size() != 1) {
            throw std::runtime_error("Invalid edge_embedding.norm.weight shape");
        }
        config.hidden_dim = norm_info.shape[0];

        // Get num_rbf from edge_embedding.weight shape
        if (!loader.has_tensor("edge_embedding.weight")) {
            throw std::runtime_error("Missing edge_embedding.weight");
        }
        auto edge_info = loader.get_info("edge_embedding.weight");
        if (edge_info.shape.size() != 2) {
            throw std::runtime_error("Invalid edge_embedding.weight shape");
        }
        int edge_features = edge_info.shape[0];
        // V1: edge_features = 16 (positional) + 25 * num_rbf
        config.num_rbf = (edge_features - 16) / 25;  // 25 atom-pair feature sets

        // Count number of layers
        config.num_layers = 0;
        for (int i = 0; i < 10; i++) {  // Max 10 layers
            std::string key = "layers." + std::to_string(i) + ".W1.weight";
            if (loader.has_tensor(key)) {
                config.num_layers = i + 1;
            } else {
                break;
            }
        }

        if (config.num_layers == 0) {
            throw std::runtime_error("No layers found in safetensors file");
        }

        // Default values (not stored in weights)
        config.k_neighbors = 30;

        return config;
    }

    static void validate_config_match(const mpnn::MPNNConfig& expected,
                                      const mpnn::MPNNConfig& actual) {
        if (expected.hidden_dim != actual.hidden_dim) {
            throw std::runtime_error("Config mismatch: hidden_dim " +
                                     std::to_string(expected.hidden_dim) +
                                     " != " + std::to_string(actual.hidden_dim));
        }

        if (expected.num_layers != actual.num_layers) {
            throw std::runtime_error("Config mismatch: num_layers " +
                                     std::to_string(expected.num_layers) +
                                     " != " + std::to_string(actual.num_layers));
        }

        if (expected.num_rbf != actual.num_rbf) {
            throw std::runtime_error("Config mismatch: num_rbf " +
                                     std::to_string(expected.num_rbf) +
                                     " != " + std::to_string(actual.num_rbf));
        }
    }

    // Helper: Load tensor into vector (handles memory management)
    static void load_tensor_into_vector(const SafetensorsLoader& loader, const std::string& name,
                                        std::vector<float>& vec) {
        float* data = loader.load_tensor(name);
        size_t num_elements = loader.get_info(name).num_elements();
        vec.assign(data, data + num_elements);  // Copy to vector
        delete[] data;                          // Free temporary allocation
    }

    static void load_edge_embedding(const SafetensorsLoader& loader, mpnn::MPNNWeights& weights,
                                    [[maybe_unused]] const mpnn::MPNNConfig& config) {
        // Edge embedding weight (NO bias - V1 uses with_bias=False)
        load_tensor_into_vector(loader, "edge_embedding.weight", weights.edge_embedding_weight);

        // Edge norms
        load_tensor_into_vector(loader, "edge_embedding.norm.weight", weights.edge_norm_gamma);
        load_tensor_into_vector(loader, "edge_embedding.norm.bias", weights.edge_norm_beta);

        // Initial edge transformation W_e
        load_tensor_into_vector(loader, "W_e.weight", weights.W_e_weight);
        load_tensor_into_vector(loader, "W_e.bias", weights.W_e_bias);

        // Positional encoding linear layer
        load_tensor_into_vector(loader, "positional_encoding.weight", weights.positional_weight);
        load_tensor_into_vector(loader, "positional_encoding.bias", weights.positional_bias);
    }

    static void load_layer(const SafetensorsLoader& loader, mpnn::MPNNWeights& weights,
                           [[maybe_unused]] const mpnn::MPNNConfig& config, int layer_idx) {
        auto& layer = weights.layers[layer_idx];
        std::string prefix = "layers." + std::to_string(layer_idx) + ".";

        // Message MLP
        load_tensor_into_vector(loader, prefix + "W1.weight", layer.W1_weight);
        load_tensor_into_vector(loader, prefix + "W1.bias", layer.W1_bias);
        load_tensor_into_vector(loader, prefix + "W2.weight", layer.W2_weight);
        load_tensor_into_vector(loader, prefix + "W2.bias", layer.W2_bias);
        load_tensor_into_vector(loader, prefix + "W3.weight", layer.W3_weight);
        load_tensor_into_vector(loader, prefix + "W3.bias", layer.W3_bias);

        // Layer norms
        load_tensor_into_vector(loader, prefix + "norm1.weight", layer.norm1_gamma);
        load_tensor_into_vector(loader, prefix + "norm1.bias", layer.norm1_beta);
        load_tensor_into_vector(loader, prefix + "norm2.weight", layer.norm2_gamma);
        load_tensor_into_vector(loader, prefix + "norm2.bias", layer.norm2_beta);

        // FFN
        load_tensor_into_vector(loader, prefix + "ffn.W_in.weight", layer.ffn_W_in_weight);
        load_tensor_into_vector(loader, prefix + "ffn.W_in.bias", layer.ffn_W_in_bias);
        load_tensor_into_vector(loader, prefix + "ffn.W_out.weight", layer.ffn_W_out_weight);
        load_tensor_into_vector(loader, prefix + "ffn.W_out.bias", layer.ffn_W_out_bias);

        // Edge update MLP
        load_tensor_into_vector(loader, prefix + "W11.weight", layer.W11_weight);
        load_tensor_into_vector(loader, prefix + "W11.bias", layer.W11_bias);
        load_tensor_into_vector(loader, prefix + "W12.weight", layer.W12_weight);
        load_tensor_into_vector(loader, prefix + "W12.bias", layer.W12_bias);
        load_tensor_into_vector(loader, prefix + "W13.weight", layer.W13_weight);
        load_tensor_into_vector(loader, prefix + "W13.bias", layer.W13_bias);

        // Edge norm (norm3)
        load_tensor_into_vector(loader, prefix + "norm3.weight", layer.norm3_gamma);
        load_tensor_into_vector(loader, prefix + "norm3.bias", layer.norm3_beta);
    }

    static SWParams load_sw_params(const SafetensorsLoader& loader) {
        SWParams params;

        // Load gap extension penalty (optional)
        if (loader.has_tensor("gap")) {
            float* gap_tensor = loader.load_tensor("gap");
            params.gap = gap_tensor[0];
            delete[] gap_tensor;
        }

        // Load gap opening penalty (optional)
        if (loader.has_tensor("gap_open")) {
            float* gap_open_tensor = loader.load_tensor("gap_open");
            params.gap_open = gap_open_tensor[0];
            delete[] gap_open_tensor;
        }

        // Load temperature (optional)
        if (loader.has_tensor("temperature")) {
            float* temp_tensor = loader.load_tensor("temperature");
            params.temperature = temp_tensor[0];
            delete[] temp_tensor;
        }

        return params;
    }
};

}  // namespace weights
}  // namespace pfalign
