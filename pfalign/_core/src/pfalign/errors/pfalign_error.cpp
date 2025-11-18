#include "pfalign_error.h"
#include <sstream>

namespace pfalign {
namespace errors {

PFalignError::PFalignError(ErrorCategory category, const std::string& message,
                           const std::string& suggestion, const std::string& context)
    : std::runtime_error(message),
      category_(category),
      message_(message),
      suggestion_(suggestion),
      context_(context) {}

PFalignError::PFalignError(const ErrorInfo& info)
    : PFalignError(info.category, info.message, info.suggestion, info.context) {}

std::string PFalignError::formatted() const {
    std::ostringstream oss;
    oss << "[ERROR] " << category_to_string(category_) << ": " << message_;

    if (!context_.empty()) {
        oss << "\n  Context: " << context_;
    }

    if (!suggestion_.empty()) {
        oss << "\n  Suggestion: " << suggestion_;
    }

    return oss.str();
}

FileNotFoundError::FileNotFoundError(const std::string& path, const std::string& file_type)
    : PFalignError(
        ErrorCategory::FileIO,
        file_type + " not found: " + path,
        "Check that the file path exists and is readable",
        "") {}

FileWriteError::FileWriteError(const std::string& path, const std::string& reason)
    : PFalignError(
        ErrorCategory::FileIO,
        "Cannot write to file: " + path,
        "Check that the directory exists and you have write permissions",
        reason.empty() ? "" : "Reason: " + reason) {}

ValidationError::ValidationError(const std::string& param_name,
                                const std::string& value,
                                const std::string& expected)
    : PFalignError(
        ErrorCategory::Validation,
        "Invalid value for " + param_name + ": " + value,
        "Expected: " + expected,
        "") {}

ValidationError::ValidationError(const std::string& message, const std::string& suggestion)
    : PFalignError(ErrorCategory::Validation, message, suggestion, "") {}

FormatError::FormatError(const std::string& path,
                        const std::string& reason,
                        const std::vector<std::string>& supported_formats)
    : PFalignError(
        ErrorCategory::Format,
        "Format error in " + path + ": " + reason,
        supported_formats.empty() ? "" :
            "Supported formats: " + [&]() {
                std::string result;
                for (size_t i = 0; i < supported_formats.size(); ++i) {
                    if (i > 0) result += ", ";
                    result += supported_formats[i];
                }
                return result;
            }(),
        "") {}

ChainNotFoundError::ChainNotFoundError(const std::string& chain_id,
                                      const std::string& structure_path,
                                      const std::vector<std::string>& available_chains)
    : PFalignError(
        ErrorCategory::UserError,
        "Chain '" + chain_id + "' not found in " + structure_path,
        [&]() {
            if (available_chains.empty()) {
                return std::string("Structure has no chains");
            }
            std::string result = "Available chains: ";
            for (size_t i = 0; i < available_chains.size(); ++i) {
                if (i > 0) result += ", ";
                result += available_chains[i];
            }
            return result;
        }(),
        "Use chain ID (A, B, C, ...) or index (0, 1, 2, ...)") {}

DimensionError::DimensionError(const std::string& param_name,
                              const std::string& actual_shape,
                              const std::string& expected_shape)
    : PFalignError(
        ErrorCategory::Validation,
        "Invalid shape for " + param_name + ": " + actual_shape,
        "Expected shape: " + expected_shape,
        "") {}

AlgorithmError::AlgorithmError(const std::string& algorithm_name,
                              const std::string& error_message,
                              const std::string& suggestion)
    : PFalignError(
        ErrorCategory::Algorithm,
        algorithm_name + " failed: " + error_message,
        suggestion.empty() ? "Try adjusting algorithm parameters or input data" : suggestion,
        "") {}

}  // namespace errors
}  // namespace pfalign
