#include "commands.h"
#include <iostream>
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"

namespace pfalign {
namespace commands {

int full(const std::string& weights_path, const std::string& pdb1_path,
         const std::string& pdb2_path, const std::string& output_path,
         const std::string& fasta_path, const std::string& mode, float gap_open, float gap_extend) {
    std::cout << "===========================================\n";
    std::cout << "  PFalign Full Pipeline Command\n";
    std::cout << "===========================================\n\n";
    std::cout << "Weights:       " << weights_path << "\n";
    std::cout << "PDB 1:         " << pdb1_path << "\n";
    std::cout << "PDB 2:         " << pdb2_path << "\n";
    std::cout << "Output:        " << output_path << "\n";
    if (!fasta_path.empty()) {
        std::cout << "FASTA output:  " << fasta_path << "\n";
    }
    std::cout << "Mode:          " << mode << "\n";
    std::cout << "Gap open:      " << gap_open << "\n";
    std::cout << "Gap extend:    " << gap_extend << "\n\n";

    std::cout << "⚠️  This command is not yet fully implemented.\n";
    std::cout << "   The CLI framework is working correctly!\n";
    std::cout << "   Implementation will be added in Phase 7.\n\n";

    std::cout << "Pipeline steps:\n";
    std::cout << "  1. Parse PDB files\n";
    std::cout << "  2. Load MPNN weights\n";
    std::cout << "  3. Compute embeddings for both proteins\n";
    std::cout << "  4. Compute similarity matrix\n";
    std::cout << "  5. Run Smith-Waterman alignment\n";
    std::cout << "  6. Save alignment posteriors\n";
    if (!fasta_path.empty()) {
        std::cout << "  7. Save FASTA alignment\n";
    }
    std::cout << "\n";

    return 0;
}

}  // namespace commands
}  // namespace pfalign
