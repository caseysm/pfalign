#pragma once

#include "error_categories.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace pfalign {
namespace errors {

/**
 * Base exception class for all pfalign errors.
 *
 * Provides structured error information with category, message,
 * suggestion, and context for helpful error reporting.
 */
class PFalignError : public std::runtime_error {
protected:
    ErrorCategory category_;
    std::string message_;
    std::string suggestion_;
    std::string context_;

public:
    PFalignError(ErrorCategory category, const std::string& message,
                 const std::string& suggestion = "",
                 const std::string& context = "");

    PFalignError(const ErrorInfo& info);

    virtual ~PFalignError() = default;

    ErrorCategory category() const { return category_; }
    const std::string& message() const { return message_; }
    const std::string& suggestion() const { return suggestion_; }
    const std::string& context() const { return context_; }

    /**
     * Get fully formatted error message for display.
     * Format:
     *   [ERROR] {category}: {message}
     *   Context: {context}
     *   Suggestion: {suggestion}
     */
    std::string formatted() const;
};

/**
 * File I/O error - file not found, cannot read/write, permission denied.
 */
class FileNotFoundError : public PFalignError {
public:
    FileNotFoundError(const std::string& path,
                     const std::string& file_type = "file");
};

/**
 * File cannot be written.
 */
class FileWriteError : public PFalignError {
public:
    FileWriteError(const std::string& path,
                   const std::string& reason = "");
};

/**
 * Validation error - parameter out of range, invalid value.
 */
class ValidationError : public PFalignError {
public:
    ValidationError(const std::string& param_name,
                   const std::string& value,
                   const std::string& expected);

    ValidationError(const std::string& message,
                   const std::string& suggestion = "");
};

/**
 * File format error - unsupported format, parse failure.
 */
class FormatError : public PFalignError {
public:
    FormatError(const std::string& path,
               const std::string& reason,
               const std::vector<std::string>& supported_formats = {});
};

/**
 * Chain not found in structure.
 */
class ChainNotFoundError : public PFalignError {
public:
    ChainNotFoundError(const std::string& chain_id,
                      const std::string& structure_path,
                      const std::vector<std::string>& available_chains);
};

/**
 * Array/tensor shape or dimension error.
 */
class DimensionError : public PFalignError {
public:
    DimensionError(const std::string& param_name,
                  const std::string& actual_shape,
                  const std::string& expected_shape);
};

/**
 * Algorithm error - convergence failure, invalid state.
 */
class AlgorithmError : public PFalignError {
public:
    AlgorithmError(const std::string& algorithm_name,
                  const std::string& error_message,
                  const std::string& suggestion = "");
};

}  // namespace errors
}  // namespace pfalign
