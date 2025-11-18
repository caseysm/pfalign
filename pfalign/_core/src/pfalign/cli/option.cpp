#include "option.h"
#include "errors.h"
#include <algorithm>
#include <sstream>

namespace pfalign {
namespace cli {

Option::Option(const std::string& names, const std::string& desc)
    : names_(names), description_(desc) {
    parse_names();
}

void Option::parse_names() {
    // Parse comma-separated names: "-f,--file,input"
    // Short names start with single dash: -f
    // Long names start with double dash: --file
    // Positional names have no dashes: input

    std::string current;
    std::istringstream stream(names_);

    while (std::getline(stream, current, ',')) {
        // Trim whitespace
        current.erase(0, current.find_first_not_of(" \t"));
        current.erase(current.find_last_not_of(" \t") + 1);

        if (current.empty())
            continue;

        if (current.size() >= 2 && current[0] == '-' && current[1] == '-') {
            // Long name: --file
            long_names_.push_back(current);
        } else if (current.size() >= 2 && current[0] == '-') {
            // Short name: -f
            short_names_.push_back(current);
        } else {
            // Positional: input
            positional_names_.push_back(current);
        }
    }
}

Option* Option::required(bool value) {
    required_ = value;
    return this;
}

Option* Option::check(const Validator& validator) {
    validators_.push_back(validator);
    return this;
}

Option* Option::consume_remaining(bool value) {
    consume_remaining_ = value;
    return this;
}

bool Option::parse(const std::string& input) {
    // Run validators first
    for (const auto& validator : validators_) {
        std::string error = validator(input);
        if (!error.empty()) {
            throw ValidationError(error);
        }
    }

    // Parse the value
    if (!value_) {
        throw ParseError("Option not bound to a variable: " + names_);
    }

    if (!value_->parse(input)) {
        throw ParseError("Failed to parse value '" + input + "' for option " + names_);
    }

    ++actual_count_;
    return true;
}

bool Option::matches(const std::string& arg) const {
    // Check short names
    for (const auto& s : short_names_) {
        if (arg == s)
            return true;
    }

    // Check long names
    for (const auto& l : long_names_) {
        if (arg == l)
            return true;
        // Handle --flag=value format
        if (arg.find('=') != std::string::npos) {
            auto equals_pos = arg.find('=');
            if (arg.substr(0, equals_pos) == l) {
                return true;
            }
        }
    }

    return false;
}

std::string Option::extract_value(const std::string& arg) const {
    // Extract value from --flag=value format
    auto equals_pos = arg.find('=');
    if (equals_pos != std::string::npos) {
        return arg.substr(equals_pos + 1);
    }
    return "";
}

bool Option::is_satisfied() const {
    if (!required_)
        return true;
    return actual_count_ > 0;
}

std::string Option::help_text() const {
    std::ostringstream oss;

    // Format names
    if (!short_names_.empty() || !long_names_.empty()) {
        bool first = true;
        for (const auto& s : short_names_) {
            if (!first)
                oss << ",";
            oss << s;
            first = false;
        }
        for (const auto& l : long_names_) {
            if (!first)
                oss << ",";
            oss << l;
            first = false;
        }
    } else if (!positional_names_.empty()) {
        oss << positional_names_[0];
    }

    // Add type hint
    if (value_) {
        oss << " " << value_->type_name();
    }

    // Add validator descriptions
    if (!validators_.empty()) {
        oss << " (";
        for (size_t i = 0; i < validators_.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << validators_[i].description();
        }
        oss << ")";
    }

    // Add required marker
    if (required_) {
        oss << " [REQUIRED]";
    }

    // Add description on new line with indentation
    if (!description_.empty()) {
        oss << "\n    " << description_;
    }

    return oss.str();
}

}  // namespace cli
}  // namespace pfalign
