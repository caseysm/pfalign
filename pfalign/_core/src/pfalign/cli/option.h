#pragma once

#include "types.h"
#include "validators.h"
#include <memory>
#include <vector>
#include <string>

namespace pfalign {
namespace cli {

// Represents a command-line option (flag or positional argument)
class Option {
    std::string names_;  // e.g., "-f,--file" or "input"
    std::string description_;
    std::unique_ptr<TypedValue> value_;
    std::vector<Validator> validators_;
    bool required_{false};
    [[maybe_unused]] bool is_flag_{false};
    [[maybe_unused]] int expected_count_{1};  // 0 for flags, 1+ for options
    int actual_count_{0};
    bool consume_remaining_{false};

    // Parsed name variants
    std::vector<std::string> short_names_;       // e.g., ["-f"]
    std::vector<std::string> long_names_;        // e.g., ["--file"]
    std::vector<std::string> positional_names_;  // e.g., ["input"]

    void parse_names();

public:
    Option(const std::string& names, const std::string& desc);

    // Fluent setters
    Option* required(bool value = true);
    Option* check(const Validator& validator);
    Option* consume_remaining(bool value = true);

    // Type binding
    template <typename T>
    Option* bind(T* ptr) {
        value_ = std::make_unique<TypedValueImpl<T>>(ptr);
        return this;
    }

    // Parsing
    bool parse(const std::string& input);
    [[nodiscard]] bool matches(const std::string& arg) const;
    [[nodiscard]] std::string extract_value(const std::string& arg) const;  // For --flag=value
    [[nodiscard]] bool consumes_remaining() const {
        return consume_remaining_;
    }

    // Getters
    [[nodiscard]] bool is_required() const {
        return required_;
    }
    [[nodiscard]] bool is_satisfied() const;
    [[nodiscard]] int count() const {
        return actual_count_;
    }
    [[nodiscard]] std::string help_text() const;
    [[nodiscard]] const std::string& names() const {
        return names_;
    }
    [[nodiscard]] const std::string& description() const {
        return description_;
    }
    [[nodiscard]] bool is_positional() const {
        return !positional_names_.empty() && short_names_.empty() && long_names_.empty();
    }
};

}  // namespace cli
}  // namespace pfalign
