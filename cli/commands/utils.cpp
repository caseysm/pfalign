#include "commands.h"
#include <iostream>
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"

namespace pfalign {
namespace commands {

int reformat(const std::string& input_path, const std::string& output_path,
             const std::string& input_format, const std::string& output_format,
             const std::string& match_mode, int gap_threshold, bool remove_inserts,
             int remove_gapped, bool uppercase, bool lowercase, bool remove_ss) {
    // Format conversion is best handled by Python CLI with its rich MSA I/O
    (void)input_path; (void)output_path; (void)input_format; (void)output_format;
    (void)match_mode; (void)gap_threshold; (void)remove_inserts; (void)remove_gapped;
    (void)uppercase; (void)lowercase; (void)remove_ss;

    std::cerr << "Note: Format conversion is best handled by the Python CLI\n";
    std::cerr << "      Use: python -m pfalign reformat " << input_path << " --output " << output_path << "\n";
    return 1;
}

int info(const std::string& path, bool show_chains) {
    // Structure inspection is best handled by Python CLI
    (void)path; (void)show_chains;

    std::cerr << "Note: Structure inspection is best handled by the Python CLI\n";
    std::cerr << "      Use: python -m pfalign info " << path;
    if (show_chains) {
        std::cerr << " --chains";
    }
    std::cerr << "\n";
    return 1;
}

int version() {
    // Basic version information
    std::cout << "pfalign 0.0.1a0" << std::endl;
    std::cout << "Build: C++ CLI (development)" << std::endl;
    std::cout << "Platform: " << __VERSION__ << std::endl;
    std::cout << "\nFor full version details: python -m pfalign version" << std::endl;
    return 0;
}

}  // namespace commands
}  // namespace pfalign
