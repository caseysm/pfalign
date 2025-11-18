#pragma once

#include <stdexcept>
#include <string>

namespace pfalign {
namespace cli {

// Base error class for all CLI errors
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {
    }
    virtual int get_exit_code() const {
        return 1;
    }
};

// Parse errors - general argument parsing failures
class ParseError : public Error {
public:
    explicit ParseError(const std::string& msg) : Error(msg) {
    }
};

// Validation errors - argument failed validation
class ValidationError : public ParseError {
public:
    explicit ValidationError(const std::string& msg) : ParseError(msg) {
    }
};

// Missing argument errors
class MissingArgument : public ParseError {
public:
    explicit MissingArgument(const std::string& msg) : ParseError(msg) {
    }
};

// Help requested (not an error, exit 0)
class CallForHelp : public Error {
public:
    CallForHelp() : Error("Help requested") {
    }
    int get_exit_code() const override {
        return 0;
    }
};

}  // namespace cli
}  // namespace pfalign
