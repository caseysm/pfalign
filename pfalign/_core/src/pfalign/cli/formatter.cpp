#include "formatter.h"
#include "app.h"
#include "option.h"
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace pfalign {
namespace cli {

std::string HelpFormatter::format(const App& app) {
    std::ostringstream oss;

    // Application name and description
    oss << app.name();
    if (!app.description().empty()) {
        oss << " - " << app.description();
    }
    oss << "\n\n";

    // Usage
    oss << format_usage(app) << "\n\n";

    // Subcommands (if any)
    if (!app.get_subcommands().empty()) {
        std::vector<const App*> subcommands;
        for (const auto& sub : app.get_subcommands()) {
            subcommands.push_back(sub.get());
        }
        oss << format_subcommands(subcommands) << "\n\n";
    }

    // Positional arguments (if any)
    if (!app.get_positionals().empty()) {
        std::vector<const Option*> positionals;
        for (const auto& pos : app.get_positionals()) {
            positionals.push_back(pos.get());
        }
        oss << format_positionals(positionals) << "\n\n";
    }

    // Options (if any)
    if (!app.get_options().empty()) {
        std::vector<const Option*> options;
        for (const auto& opt : app.get_options()) {
            options.push_back(opt.get());
        }
        oss << format_options(options) << "\n\n";
    }

    // Generic help option (always available)
    oss << "OPTIONS:\n";
    oss << "  -h,--help                Show this help message and exit\n";

    return oss.str();
}

std::string HelpFormatter::format_usage(const App& app) {
    std::ostringstream oss;
    oss << "USAGE:\n  " << app.name();

    // Add subcommand placeholder
    if (!app.get_subcommands().empty()) {
        oss << " <subcommand>";
    }

    // Add positionals
    for (const auto& pos : app.get_positionals()) {
        oss << " <" << pos->names() << ">";
    }

    // Add options placeholder
    if (!app.get_options().empty()) {
        oss << " [options]";
    }

    return oss.str();
}

std::string HelpFormatter::format_description(const App& app) {
    if (app.description().empty()) {
        return "";
    }
    return "DESCRIPTION:\n  " + app.description() + "\n";
}

std::string HelpFormatter::format_positionals(const std::vector<const Option*>& positionals) {
    if (positionals.empty()) {
        return "";
    }

    std::ostringstream oss;
    oss << "ARGUMENTS:\n";

    for (const auto* pos : positionals) {
        oss << "  " << std::left << std::setw(24) << pos->names();
        oss << pos->description();
        if (pos->is_required()) {
            oss << " [REQUIRED]";
        }
        oss << "\n";
    }

    return oss.str();
}

std::string HelpFormatter::format_options(const std::vector<const Option*>& options) {
    if (options.empty()) {
        return "";
    }

    std::ostringstream oss;
    oss << "OPTIONS:\n";

    for (const auto* opt : options) {
        // Build the names string
        std::string names_str = opt->names();

        oss << "  " << std::left << std::setw(24) << names_str;
        oss << opt->description();
        if (opt->is_required()) {
            oss << " [REQUIRED]";
        }
        oss << "\n";
    }

    return oss.str();
}

std::string HelpFormatter::format_subcommands(const std::vector<const App*>& subcommands) {
    if (subcommands.empty()) {
        return "";
    }

    std::ostringstream oss;
    oss << "SUBCOMMANDS:\n";

    // Find max name length for alignment
    size_t max_len = 0;
    for (const auto* sub : subcommands) {
        max_len = std::max(max_len, sub->name().size());
    }

    for (const auto* sub : subcommands) {
        oss << "  " << std::left << std::setw(max_len + 4) << sub->name();
        oss << sub->description() << "\n";
    }

    oss << "\nUse \"" << subcommands[0]->name()
        << " <subcommand> --help\" for more information about a subcommand.";

    return oss.str();
}

std::string HelpFormatter::wrap_text(const std::string& text, size_t width, size_t indent) {
    // Simple word wrap implementation
    std::ostringstream oss;
    std::string indent_str(indent, ' ');
    [[maybe_unused]] size_t pos = 0;
    size_t line_len = 0;

    std::istringstream words(text);
    std::string word;

    while (words >> word) {
        if (line_len > 0 && line_len + word.size() + 1 > width) {
            oss << "\n" << indent_str;
            line_len = 0;
        }

        if (line_len > 0) {
            oss << " ";
            ++line_len;
        }

        oss << word;
        line_len += word.size();
    }

    return oss.str();
}

}  // namespace cli
}  // namespace pfalign
