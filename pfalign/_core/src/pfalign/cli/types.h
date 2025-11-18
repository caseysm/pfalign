#pragma once

#include <cctype>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace pfalign {
namespace cli {

// Generic type conversion from string
template <typename T>
bool parse_value(const std::string& input, T& output) {
    std::istringstream ss(input);
    ss >> output;
    return !ss.fail() && ss.eof();
}

// Specialization for std::string (no conversion needed)
template <>
inline bool parse_value<std::string>(const std::string& input, std::string& output) {
    output = input;
    return true;
}

// Specialization for bool (handle true/false, yes/no, 1/0)
template <>
inline bool parse_value<bool>(const std::string& input, bool& output) {
    std::string lower = input;
    for (auto& c : lower)
        c = std::tolower(c);

    if (lower == "true" || lower == "yes" || lower == "1" || lower == "on") {
        output = true;
        return true;
    } else if (lower == "false" || lower == "no" || lower == "0" || lower == "off") {
        output = false;
        return true;
    }
    return false;
}

// Type-erased storage interface
class TypedValue {
public:
    virtual ~TypedValue() = default;
    virtual bool parse(const std::string& input) = 0;
    virtual std::string type_name() const = 0;
    virtual std::unique_ptr<TypedValue> clone() const = 0;
};

// Concrete implementation for specific types
template <typename T>
class TypedValueImpl : public TypedValue {
    T* ptr_;

public:
    explicit TypedValueImpl(T* ptr) : ptr_(ptr) {
    }

    bool parse(const std::string& input) override {
        return parse_value(input, *ptr_);
    }

    std::string type_name() const override {
        if (std::is_same<T, std::string>::value)
            return "TEXT";
        if (std::is_same<T, int>::value)
            return "INT";
        if (std::is_same<T, float>::value)
            return "FLOAT";
        if (std::is_same<T, double>::value)
            return "FLOAT";
        if (std::is_same<T, bool>::value)
            return "BOOL";
        return "VALUE";
    }

    std::unique_ptr<TypedValue> clone() const override {
        return std::make_unique<TypedValueImpl<T>>(ptr_);
    }
};

// Specialization for std::vector<std::string> to support multi-value positionals
template <>
inline bool parse_value<std::vector<std::string>>(const std::string& input,
                                                  std::vector<std::string>& output) {
    output.push_back(input);
    return true;
}

}  // namespace cli
}  // namespace pfalign
