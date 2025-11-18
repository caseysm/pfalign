#pragma once

#include <string>
#include <vector>

namespace pfalign {
namespace cli {

class App;
class Option;

// Generates help text for applications
class HelpFormatter {
public:
    static std::string format(const App& app);

private:
    static std::string format_usage(const App& app);
    static std::string format_description(const App& app);
    static std::string format_positionals(const std::vector<const Option*>& positionals);
    static std::string format_options(const std::vector<const Option*>& options);
    static std::string format_subcommands(const std::vector<const App*>& subcommands);

    // Helper to wrap text to terminal width
    static std::string wrap_text(const std::string& text, size_t width = 80, size_t indent = 0);
};

}  // namespace cli
}  // namespace pfalign
