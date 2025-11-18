#include "formatters.h"
#include <sstream>

namespace pfalign {
namespace errors {

std::string format_error(const std::string& message,
                        const std::string& suggestion,
                        const std::string& context) {
    std::ostringstream oss;
    oss << "[ERROR] " << message;

    if (!context.empty()) {
        oss << "\n  Context: " << context;
    }

    if (!suggestion.empty()) {
        oss << "\n  Suggestion: " << suggestion;
    }

    return oss.str();
}

std::string format_file_not_found(const std::string& path,
                                  const std::string& file_type) {
    return format_error(
        file_type + " not found: " + path,
        "Check that the file path exists and is readable"
    );
}

std::string format_chain_not_found(const std::string& chain_id,
                                   const std::string& path,
                                   const std::vector<std::string>& available) {
    std::string avail_str = join(available, ", ");
    return format_error(
        "Chain '" + chain_id + "' not found in " + path,
        available.empty() ? "Structure has no chains" : "Available chains: " + avail_str,
        "Use chain ID (A, B, C, ...) or index (0, 1, 2, ...)"
    );
}

std::string format_validation_error(const std::string& param_name,
                                    const std::string& value,
                                    const std::string& expected) {
    return format_error(
        "Invalid value for " + param_name + ": " + value,
        "Expected: " + expected
    );
}

std::string format_format_error(const std::string& path,
                               const std::string& reason,
                               const std::vector<std::string>& supported) {
    std::string supp_str = supported.empty() ? "" : "Supported formats: " + join(supported, ", ");
    return format_error(
        "Format error in " + path + ": " + reason,
        supp_str
    );
}

std::string format_dimension_error(const std::string& param_name,
                                   const std::string& actual,
                                   const std::string& expected) {
    return format_error(
        "Invalid shape for " + param_name + ": " + actual,
        "Expected shape: " + expected
    );
}

std::string format_range_error(const std::string& param_name,
                              const std::string& value,
                              const std::string& min,
                              const std::string& max) {
    return format_error(
        "Value for " + param_name + " out of range: " + value,
        "Expected value in range [" + min + ", " + max + "]"
    );
}

std::string join(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) return "";

    std::ostringstream oss;
    oss << strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        oss << delimiter << strings[i];
    }
    return oss.str();
}

}  // namespace errors
}  // namespace pfalign
