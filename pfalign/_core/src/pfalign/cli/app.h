#pragma once

#include "option.h"
#include "errors.h"
#include <vector>
#include <memory>
#include <string>

namespace pfalign {
namespace cli {

// Main application class supporting subcommands
class App {
    std::string name_;
    std::string description_;
    std::vector<std::unique_ptr<Option>> options_;
    std::vector<std::unique_ptr<Option>> positionals_;
    std::vector<std::unique_ptr<App>> subcommands_;
    App* active_subcommand_{nullptr};
    App* parent_{nullptr};
    bool require_subcommand_{false};
    [[maybe_unused]] bool help_flag_{false};

public:
    App(const std::string& name, const std::string& description = "");

    // Subcommands
    App* add_subcommand(const std::string& name, const std::string& desc);
    void require_subcommand(bool value = true);
    [[nodiscard]] App* get_subcommand(const std::string& name);
    [[nodiscard]] const App* get_subcommand(const std::string& name) const;

    // Options (flags and named arguments)
    template <typename T>
    Option* add_option(const std::string& names, T& variable, const std::string& description = "") {
        auto opt = std::make_unique<Option>(names, description);
        opt->bind(&variable);
        auto* ptr = opt.get();
        options_.push_back(std::move(opt));
        return ptr;
    }

    template <typename T>
    Option* add_flag(const std::string& names, T& variable, const std::string& description = "") {
        auto opt = std::make_unique<Option>(names, description);
        opt->bind(&variable);
        auto* ptr = opt.get();
        options_.push_back(std::move(opt));
        return ptr;
    }

    // Positional arguments
    template <typename T>
    Option* add_positional(const std::string& name, T& variable,
                           const std::string& description = "") {
        auto opt = std::make_unique<Option>(name, description);
        opt->bind(&variable)->required(true);
        auto* ptr = opt.get();
        positionals_.push_back(std::move(opt));
        return ptr;
    }

    // Parsing
    void parse(int argc, char** argv);
    void parse(const std::vector<std::string>& args);

    // Help
    [[nodiscard]] std::string help() const;
    int exit(const Error& e) const;

    // Getters
    [[nodiscard]] const std::string& name() const {
        return name_;
    }
    [[nodiscard]] const std::string& description() const {
        return description_;
    }
    [[nodiscard]] bool got_subcommand() const {
        return active_subcommand_ != nullptr;
    }
    [[nodiscard]] App* get_active_subcommand() {
        return active_subcommand_;
    }
    [[nodiscard]] const App* get_active_subcommand() const {
        return active_subcommand_;
    }
    [[nodiscard]] const std::vector<std::unique_ptr<Option>>& get_options() const {
        return options_;
    }
    [[nodiscard]] const std::vector<std::unique_ptr<Option>>& get_positionals() const {
        return positionals_;
    }
    [[nodiscard]] const std::vector<std::unique_ptr<App>>& get_subcommands() const {
        return subcommands_;
    }
};

}  // namespace cli
}  // namespace pfalign
