#pragma once

#include "error_categories.h"
#include <string>
#include <vector>

namespace pfalign {
namespace errors {

/**
 * Utilities for consistent error message formatting across all components.
 */

/**
 * Format a generic error message with optional suggestion and context.
 */
std::string format_error(const std::string& message,
                        const std::string& suggestion = "",
                        const std::string& context = "");

/**
 * Format file not found error with helpful suggestions.
 */
std::string format_file_not_found(const std::string& path,
                                  const std::string& file_type = "file");

/**
 * Format chain not found error with list of available chains.
 */
std::string format_chain_not_found(const std::string& chain_id,
                                   const std::string& path,
                                   const std::vector<std::string>& available);

/**
 * Format validation error showing actual vs expected value.
 */
std::string format_validation_error(const std::string& param_name,
                                    const std::string& value,
                                    const std::string& expected);

/**
 * Format format/parse error with supported formats.
 */
std::string format_format_error(const std::string& path,
                               const std::string& reason,
                               const std::vector<std::string>& supported);

/**
 * Format dimension mismatch error.
 */
std::string format_dimension_error(const std::string& param_name,
                                   const std::string& actual,
                                   const std::string& expected);

/**
 * Format range validation error.
 */
std::string format_range_error(const std::string& param_name,
                              const std::string& value,
                              const std::string& min,
                              const std::string& max);

/**
 * Join strings with delimiter.
 */
std::string join(const std::vector<std::string>& strings, const std::string& delimiter);

}  // namespace errors
}  // namespace pfalign
