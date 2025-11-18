#include "validators.h"
#include <fstream>
#include <sstream>

namespace pfalign {
namespace cli {

Validator ExistingFile() {
    return Validator(
        [](const std::string& filename) -> std::string {
            std::ifstream file(filename);
            if (!file.good()) {
                return "File does not exist: " + filename;
            }
            return "";  // Valid
        },
        "FILE(existing)");
}

Validator Range(double min, double max) {
    return Validator(
        [min, max](const std::string& value) -> std::string {
            try {
                double num = std::stod(value);
                if (num < min || num > max) {
                    std::ostringstream oss;
                    oss << "Value " << num << " not in range [" << min << ", " << max << "]";
                    return oss.str();
                }
                return "";  // Valid
            } catch (...) {
                return "Not a valid number: " + value;
            }
        },
        "FLOAT in [" + std::to_string(min) + ", " + std::to_string(max) + "]");
}

}  // namespace cli
}  // namespace pfalign
