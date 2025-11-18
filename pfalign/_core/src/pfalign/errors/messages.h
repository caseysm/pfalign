#pragma once

#include "pfalign_error.h"
#include "error_categories.h"
#include <string>
#include <vector>

namespace pfalign {
namespace errors {
namespace messages {

/**
 * Pre-defined error message templates for common error scenarios.
 *
 * These functions create consistent, helpful error messages with
 * appropriate suggestions and context.
 */

// ============================================================================
// File I/O Errors
// ============================================================================

inline PFalignError file_not_found(const std::string& path,
                                   const std::string& file_type = "file") {
    return FileNotFoundError(path, file_type);
}

inline PFalignError file_write_error(const std::string& path,
                                     const std::string& reason = "") {
    return FileWriteError(path, reason);
}

inline PFalignError file_parse_error(const std::string& path,
                                     const std::string& format,
                                     const std::string& error_detail) {
    return FormatError(
        path,
        "Failed to parse " + format + " file: " + error_detail,
        {".pdb", ".cif", ".mmcif"}
    );
}

// ============================================================================
// Format Errors
// ============================================================================

inline PFalignError unsupported_format(const std::string& path,
                                       const std::string& extension,
                                       const std::vector<std::string>& supported) {
    return FormatError(
        path,
        "Unsupported file extension: " + extension,
        supported
    );
}

inline PFalignError invalid_npy_format(const std::string& path,
                                       const std::string& reason) {
    return FormatError(
        path,
        "Invalid .npy file: " + reason,
        {".npy (float32, C-order)"}
    );
}

// ============================================================================
// Chain Errors
// ============================================================================

inline PFalignError chain_not_found(const std::string& chain_id,
                                    const std::string& path,
                                    const std::vector<std::string>& available) {
    return ChainNotFoundError(chain_id, path, available);
}

inline PFalignError chain_index_out_of_range(int index,
                                             int num_chains,
                                             const std::string& path) {
    return ValidationError(
        "chain",
        std::to_string(index),
        "index in range [0, " + std::to_string(num_chains - 1) + "] for " + path
    );
}

inline PFalignError no_chains_in_structure(const std::string& path) {
    return ValidationError(
        "Structure has no chains: " + path,
        "Provide a structure file with at least one protein chain"
    );
}

// ============================================================================
// Dimension/Shape Errors
// ============================================================================

inline PFalignError embedding_dimension_mismatch(int dim1, int dim2,
                                                 const std::string& param1,
                                                 const std::string& param2) {
    return DimensionError(
        param1 + " vs " + param2,
        std::to_string(dim1),
        std::to_string(dim2)
    );
}

inline PFalignError invalid_array_shape(const std::string& param_name,
                                       const std::string& actual_shape,
                                       const std::string& expected_shape) {
    return DimensionError(param_name, actual_shape, expected_shape);
}

inline PFalignError array_must_be_2d(const std::string& param_name,
                                    int actual_dims) {
    return DimensionError(
        param_name,
        std::to_string(actual_dims) + "D",
        "2D array (N, D)"
    );
}

inline PFalignError empty_array(const std::string& param_name) {
    return ValidationError(
        param_name + " cannot be empty",
        "Provide a non-empty array with at least one element"
    );
}

// ============================================================================
// Validation Errors
// ============================================================================

inline PFalignError parameter_out_of_range(const std::string& param_name,
                                          const std::string& value,
                                          const std::string& min,
                                          const std::string& max) {
    return ValidationError(
        param_name,
        value,
        "value in range [" + min + ", " + max + "]"
    );
}

inline PFalignError parameter_must_be_positive(const std::string& param_name,
                                              const std::string& value) {
    return ValidationError(
        param_name,
        value,
        "positive value (> 0)"
    );
}

inline PFalignError parameter_must_be_non_negative(const std::string& param_name,
                                                  const std::string& value) {
    return ValidationError(
        param_name,
        value,
        "non-negative value (>= 0)"
    );
}

inline PFalignError invalid_enum_value(const std::string& param_name,
                                      const std::string& value,
                                      const std::vector<std::string>& valid_values) {
    std::string valid_str;
    for (size_t i = 0; i < valid_values.size(); ++i) {
        if (i > 0) valid_str += ", ";
        valid_str += "'" + valid_values[i] + "'";
    }
    return ValidationError(
        param_name,
        value,
        "one of: " + valid_str
    );
}

// ============================================================================
// MSA/Alignment Errors
// ============================================================================

inline PFalignError insufficient_sequences_for_msa(int num_sequences) {
    return ValidationError(
        "MSA requires at least 2 sequences, got " + std::to_string(num_sequences),
        "Provide at least 2 input structures or embeddings"
    );
}

inline PFalignError unknown_tree_method(const std::string& method) {
    return invalid_enum_value(
        "method",
        method,
        {"upgma", "nj", "bionj", "mst"}
    );
}

inline PFalignError alignment_failed(const std::string& reason) {
    return AlgorithmError(
        "Alignment",
        reason,
        "Check input sequences are valid and non-empty"
    );
}

inline PFalignError tree_building_failed(const std::string& method,
                                        const std::string& reason) {
    return AlgorithmError(
        "Tree building (" + method + ")",
        reason,
        "Check distance matrix is valid and symmetric"
    );
}

// ============================================================================
// MPNN Encoding Errors
// ============================================================================

inline PFalignError mpnn_encoding_failed(const std::string& path,
                                        const std::string& reason) {
    return AlgorithmError(
        "MPNN encoding",
        "Failed to encode " + path + ": " + reason,
        "Check structure file is valid and contains protein atoms"
    );
}

inline PFalignError weights_loading_failed(const std::string& reason) {
    return PFalignError(
        ErrorCategory::Resource,
        "Failed to load embedded MPNN weights: " + reason,
        "This is a build-time error - please report this issue"
    );
}

// ============================================================================
// Structure Errors
// ============================================================================

inline PFalignError empty_structure(const std::string& path) {
    return ValidationError(
        "Structure contains no atoms: " + path,
        "Provide a structure file with protein atoms"
    );
}

inline PFalignError invalid_coordinate_dimensions(const std::string& path,
                                                  int num_points,
                                                  const std::string& expected) {
    return DimensionError(
        "coordinates for " + path,
        "(" + std::to_string(num_points) + ", ?)",
        expected + " points"
    );
}

// ============================================================================
// Thread/Resource Errors
// ============================================================================

inline PFalignError invalid_thread_count(int requested, int max_available) {
    return ValidationError(
        "threads",
        std::to_string(requested),
        "value in range [0, " + std::to_string(max_available) + "] (0 = auto)"
    );
}

// ============================================================================
// Coordinate Array Errors (for pybind bindings)
// ============================================================================

inline PFalignError coordinate_array_ndim_error(const std::string& param_name) {
    return FormatError(
        param_name + " must be a 2D NumPy array",
        "Provide an array with shape (N, 3) where N is the number of atoms"
    );
}

inline PFalignError coordinate_array_shape_error(const std::string& param_name,
                                                  int actual_cols) {
    return FormatError(
        param_name + " must have shape (N, 3), got (N, " + std::to_string(actual_cols) + ")",
        "Provide coordinates as (N, 3) array: N atoms * 3 coordinates (x,y,z)"
    );
}

inline PFalignError array_size_mismatch(const std::string& param1,
                                        const std::string& param2,
                                        int size1,
                                        int size2) {
    return DimensionError(
        param1 + " vs " + param2,
        std::to_string(size1) + " elements",
        std::to_string(size2) + " elements (must match)"
    );
}

inline PFalignError sequence_length_mismatch(int len1, int len2) {
    return ValidationError(
        "Aligned sequences must have equal length",
        "Got lengths " + std::to_string(len1) + " and " + std::to_string(len2) +
        ". Use sequences from the same alignment output."
    );
}

}  // namespace messages
}  // namespace errors
}  // namespace pfalign
