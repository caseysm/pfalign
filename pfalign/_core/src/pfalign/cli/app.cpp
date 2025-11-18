#include "app.h"
#include "formatter.h"
#include <iostream>
#include <algorithm>

namespace pfalign {
namespace cli {

App::App(const std::string& name, const std::string& description)
    : name_(name), description_(description) {
}

App* App::add_subcommand(const std::string& name, const std::string& desc) {
    auto sub = std::make_unique<App>(name, desc);
    sub->parent_ = this;
    auto* ptr = sub.get();
    subcommands_.push_back(std::move(sub));
    return ptr;
}

void App::require_subcommand(bool value) {
    require_subcommand_ = value;
}

App* App::get_subcommand(const std::string& name) {
    for (auto& sub : subcommands_) {
        if (sub->name_ == name) {
            return sub.get();
        }
    }
    return nullptr;
}

const App* App::get_subcommand(const std::string& name) const {
    for (const auto& sub : subcommands_) {
        if (sub->name_ == name) {
            return sub.get();
        }
    }
    return nullptr;
}

void App::parse(int argc, char** argv) {
    // Convert to vector of strings
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {  // Skip program name (argv[0])
        args.push_back(argv[i]);
    }
    parse(args);
}

void App::parse(const std::vector<std::string>& args) {
    size_t i = 0;
    size_t positional_index = 0;

    while (i < args.size()) {
        const std::string& arg = args[i];

        // Check for help flag
        if (arg == "-h" || arg == "--help") {
            throw CallForHelp();
        }

        // Check for subcommand
        if (auto* sub = get_subcommand(arg)) {
            active_subcommand_ = sub;
            // Pass remaining args to subcommand
            std::vector<std::string> remaining(args.begin() + i + 1, args.end());
            sub->parse(remaining);
            return;  // Subcommand handles the rest
        }

        // Check for option
        bool matched = false;
        for (auto& opt : options_) {
            if (opt->matches(arg)) {
                // Check if value is embedded (--flag=value)
                std::string embedded_value = opt->extract_value(arg);
                if (!embedded_value.empty()) {
                    opt->parse(embedded_value);
                } else {
                    // Next arg should be the value
                    if (i + 1 >= args.size()) {
                        throw MissingArgument("Option " + arg + " requires a value");
                    }
                    opt->parse(args[++i]);
                }
                matched = true;
                break;
            }
        }

        if (matched) {
            ++i;
            continue;
        }

        // Treat as positional argument
        if (positional_index < positionals_.size()) {
            auto* positional = positionals_[positional_index].get();
            positional->parse(arg);
            if (!positional->consumes_remaining()) {
                ++positional_index;
            }
            ++i;
        } else {
            throw ParseError("Unexpected argument: " + arg);
        }
    }

    // Validate all required options are satisfied
    for (const auto& opt : options_) {
        if (!opt->is_satisfied()) {
            throw MissingArgument("Required option missing: " + opt->names());
        }
    }

    // Validate all positionals are satisfied
    for (const auto& pos : positionals_) {
        if (!pos->is_satisfied()) {
            throw MissingArgument("Required argument missing: " + pos->names());
        }
    }

    // Check if subcommand is required
    if (require_subcommand_ && !active_subcommand_) {
        throw ParseError("Subcommand required. Use --help to see available subcommands.");
    }
}

std::string App::help() const {
    return HelpFormatter::format(*this);
}

int App::exit(const Error& e) const {
    if (dynamic_cast<const CallForHelp*>(&e)) {
        // Help requested - print help and exit 0
        std::cout << help() << std::endl;
    } else {
        // Error occurred - print error message and help
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << help() << std::endl;
    }
    return e.get_exit_code();
}

}  // namespace cli
}  // namespace pfalign
