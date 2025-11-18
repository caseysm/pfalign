#pragma once

#include <string>

namespace pfalign {
namespace errors {

/**
 * Error categories for unified error handling across all components.
 *
 * These categories help classify errors and provide appropriate
 * suggestions to users.
 */
enum class ErrorCategory {
    FileIO,          // File not found, read/write errors, permission errors
    Validation,      // Invalid parameters, out of range values
    Format,          // Unsupported file format, parse errors
    Algorithm,       // MSA/alignment algorithm errors, convergence issues
    Resource,        // Memory allocation, thread errors
    UserError,       // Bad input from user (chain not found, etc.)
};

/**
 * Get string representation of error category for display.
 */
inline std::string category_to_string(ErrorCategory category) {
    switch (category) {
        case ErrorCategory::FileIO:
            return "File I/O";
        case ErrorCategory::Validation:
            return "Validation";
        case ErrorCategory::Format:
            return "Format";
        case ErrorCategory::Algorithm:
            return "Algorithm";
        case ErrorCategory::Resource:
            return "Resource";
        case ErrorCategory::UserError:
            return "User Error";
        default:
            return "Unknown";
    }
}

/**
 * Structured error information for consistent error reporting.
 */
struct ErrorInfo {
    ErrorCategory category;
    std::string message;
    std::string suggestion;  // What the user should do to fix the error
    std::string context;     // Additional context about the error

    ErrorInfo(ErrorCategory cat, std::string msg,
              std::string sug = "", std::string ctx = "")
        : category(cat), message(std::move(msg)),
          suggestion(std::move(sug)), context(std::move(ctx)) {}
};

}  // namespace errors
}  // namespace pfalign
