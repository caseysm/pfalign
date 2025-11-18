#pragma once

#include "pfalign_error.h"
#include <string>
#include <vector>

namespace pfalign {

// Forward declarations
namespace cli { struct StructureRecord; }

namespace validation {

/**
 * Centralized validation functions with consistent error messages.
 *
 * These functions throw specific error types (FileNotFoundError,
 * ValidationError, etc.) with helpful context and suggestions.
 */

/**
 * Validate that a file exists and is readable.
 *
 * @param path Path to the file
 * @param file_type Description of the file type for error messages
 * @throws FileNotFoundError if file doesn't exist or isn't readable
 */
void validate_file_exists(const std::string& path,
                          const std::string& file_type = "file");

/**
 * Validate that a file has one of the supported extensions.
 *
 * @param path Path to the file
 * @param valid_extensions List of valid extensions (e.g., {".pdb", ".cif"})
 * @throws FormatError if file extension is not in the list
 */
void validate_file_format(const std::string& path,
                         const std::vector<std::string>& valid_extensions);

/**
 * Validate that a directory exists.
 *
 * @param path Path to the directory
 * @throws FileNotFoundError if directory doesn't exist
 */
void validate_directory_exists(const std::string& path);

/**
 * Validate that output directory exists (creating parent if needed).
 *
 * @param path Path to output file
 * @throws FileWriteError if parent directory doesn't exist and can't be created
 */
void validate_output_path(const std::string& path);

/**
 * Validate that a chain index is valid for the given structure.
 *
 * @param num_chains Number of chains in the structure
 * @param chain_index Chain index to validate
 * @param structure_path Path to structure file (for error messages)
 * @throws ValidationError if chain index is out of range
 */
void validate_chain_index(int num_chains, int chain_index,
                         const std::string& structure_path);

/**
 * Validate that a chain ID exists in the structure.
 *
 * @param available_chains List of available chain IDs
 * @param chain_id Requested chain ID
 * @param structure_path Path to structure file (for error messages)
 * @throws ChainNotFoundError if chain ID is not in the list
 */
void validate_chain_id(const std::vector<std::string>& available_chains,
                      const std::string& chain_id,
                      const std::string& structure_path);

/**
 * Validate that a value is positive.
 *
 * @param value Value to check
 * @param param_name Parameter name for error messages
 * @throws ValidationError if value <= 0
 */
void validate_positive(int value, const std::string& param_name);
void validate_positive(float value, const std::string& param_name);

/**
 * Validate that a value is non-negative.
 *
 * @param value Value to check
 * @param param_name Parameter name for error messages
 * @throws ValidationError if value < 0
 */
void validate_non_negative(int value, const std::string& param_name);
void validate_non_negative(float value, const std::string& param_name);

/**
 * Validate that a value is in a specific range [min, max].
 *
 * @param value Value to check
 * @param min Minimum allowed value (inclusive)
 * @param max Maximum allowed value (inclusive)
 * @param param_name Parameter name for error messages
 * @throws ValidationError if value is out of range
 */
void validate_range(int value, int min, int max, const std::string& param_name);
void validate_range(float value, float min, float max, const std::string& param_name);

/**
 * Validate array/tensor shape.
 *
 * @param actual_shape Actual shape as string (e.g., "(100, 3)")
 * @param expected_shape Expected shape as string
 * @param param_name Parameter name for error messages
 * @throws DimensionError if shapes don't match
 */
void validate_shape(const std::string& actual_shape,
                   const std::string& expected_shape,
                   const std::string& param_name);

/**
 * Validate that two dimensions match.
 *
 * @param dim1 First dimension
 * @param dim2 Second dimension
 * @param param1_name Name of first parameter
 * @param param2_name Name of second parameter
 * @throws DimensionError if dimensions don't match
 */
void validate_dimensions_match(int dim1, int dim2,
                              const std::string& param1_name,
                              const std::string& param2_name);

}  // namespace validation
}  // namespace pfalign
