#include "validators.h"
#include <algorithm>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace pfalign {
namespace validation {

void validate_file_exists(const std::string& path, const std::string& file_type) {
    if (!fs::exists(path)) {
        throw errors::FileNotFoundError(path, file_type);
    }
    if (!fs::is_regular_file(path)) {
        throw errors::ValidationError(
            file_type + " is not a regular file: " + path,
            "Provide a path to a file, not a directory"
        );
    }
}

void validate_file_format(const std::string& path,
                         const std::vector<std::string>& valid_extensions) {
    fs::path p(path);
    std::string ext = p.extension().string();

    // Convert to lowercase for case-insensitive comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    for (const auto& valid_ext : valid_extensions) {
        std::string valid_lower = valid_ext;
        std::transform(valid_lower.begin(), valid_lower.end(), valid_lower.begin(), ::tolower);
        if (ext == valid_lower) {
            return;  // Valid format
        }
    }

    throw errors::FormatError(
        path,
        "Unsupported file extension: " + ext,
        valid_extensions
    );
}

void validate_directory_exists(const std::string& path) {
    if (!fs::exists(path)) {
        throw errors::FileNotFoundError(path, "Directory");
    }
    if (!fs::is_directory(path)) {
        throw errors::ValidationError(
            "Path is not a directory: " + path,
            "Provide a path to a directory, not a file"
        );
    }
}

void validate_output_path(const std::string& path) {
    fs::path p(path);
    fs::path parent = p.parent_path();

    if (!parent.empty() && !fs::exists(parent)) {
        throw errors::FileWriteError(
            path,
            "Parent directory does not exist: " + parent.string()
        );
    }
}

void validate_chain_index(int num_chains, int chain_index,
                         const std::string& structure_path) {
    if (chain_index < 0 || chain_index >= num_chains) {
        std::ostringstream oss;
        oss << "Chain index " << chain_index << " out of range for " << structure_path;
        std::string expected = "index in range [0, " + std::to_string(num_chains - 1) + "]";
        throw errors::ValidationError(
            "chain",
            std::to_string(chain_index),
            expected
        );
    }
}

void validate_chain_id(const std::vector<std::string>& available_chains,
                      const std::string& chain_id,
                      const std::string& structure_path) {
    for (const auto& chain : available_chains) {
        if (chain == chain_id) {
            return;  // Found it
        }
    }

    // Not found
    throw errors::ChainNotFoundError(chain_id, structure_path, available_chains);
}

void validate_positive(int value, const std::string& param_name) {
    if (value <= 0) {
        throw errors::ValidationError(
            param_name,
            std::to_string(value),
            "positive integer (> 0)"
        );
    }
}

void validate_positive(float value, const std::string& param_name) {
    if (value <= 0.0f) {
        std::ostringstream oss;
        oss << value;
        throw errors::ValidationError(
            param_name,
            oss.str(),
            "positive number (> 0.0)"
        );
    }
}

void validate_non_negative(int value, const std::string& param_name) {
    if (value < 0) {
        throw errors::ValidationError(
            param_name,
            std::to_string(value),
            "non-negative integer (>= 0)"
        );
    }
}

void validate_non_negative(float value, const std::string& param_name) {
    if (value < 0.0f) {
        std::ostringstream oss;
        oss << value;
        throw errors::ValidationError(
            param_name,
            oss.str(),
            "non-negative number (>= 0.0)"
        );
    }
}

void validate_range(int value, int min, int max, const std::string& param_name) {
    if (value < min || value > max) {
        std::ostringstream expected;
        expected << "value in range [" << min << ", " << max << "]";
        throw errors::ValidationError(
            param_name,
            std::to_string(value),
            expected.str()
        );
    }
}

void validate_range(float value, float min, float max, const std::string& param_name) {
    if (value < min || value > max) {
        std::ostringstream actual_str, expected;
        actual_str << value;
        expected << "value in range [" << min << ", " << max << "]";
        throw errors::ValidationError(
            param_name,
            actual_str.str(),
            expected.str()
        );
    }
}

void validate_shape(const std::string& actual_shape,
                   const std::string& expected_shape,
                   const std::string& param_name) {
    if (actual_shape != expected_shape) {
        throw errors::DimensionError(param_name, actual_shape, expected_shape);
    }
}

void validate_dimensions_match(int dim1, int dim2,
                              const std::string& param1_name,
                              const std::string& param2_name) {
    if (dim1 != dim2) {
        std::ostringstream msg;
        msg << "Dimension mismatch: " << param1_name << " has dimension " << dim1
            << " but " << param2_name << " has dimension " << dim2;
        throw errors::DimensionError(
            param1_name + " vs " + param2_name,
            std::to_string(dim1),
            std::to_string(dim2)
        );
    }
}

}  // namespace validation
}  // namespace pfalign
