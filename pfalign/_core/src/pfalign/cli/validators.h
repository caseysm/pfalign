#pragma once

#include <string>
#include <functional>

namespace pfalign {
namespace cli {

// Validator functor that checks a value and returns error message (empty if valid)
class Validator {
    std::function<std::string(const std::string&)> func_;
    std::string description_;

public:
    Validator(std::function<std::string(const std::string&)> func, const std::string& desc)
        : func_(std::move(func)), description_(desc) {
    }

    // Apply validator to a value
    std::string operator()(const std::string& value) const {
        return func_(value);
    }

    const std::string& description() const {
        return description_;
    }
};

// Built-in validators
Validator ExistingFile();
Validator Range(double min, double max);

}  // namespace cli
}  // namespace pfalign
