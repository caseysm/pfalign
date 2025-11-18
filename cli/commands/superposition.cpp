#include "commands.h"
#include "input_utils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"
#include "pfalign/io/protein_structure.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/primitives/structural_metrics/distance_matrix.h"

namespace pfalign {
namespace commands {

int kabsch(const std::string& struct1_path, const std::string& struct2_path,
           const std::string& output_path, int chain1, int chain2, bool print_rmsd) {
    try {
        // Load structures
        io::Protein protein1 = LoadStructureFile(struct1_path);
        io::Protein protein2 = LoadStructureFile(struct2_path);

        // Validate chain indices
        validation::validate_chain_index(
            static_cast<int>(protein1.num_chains()), chain1, struct1_path);
        validation::validate_chain_index(
            static_cast<int>(protein2.num_chains()), chain2, struct2_path);

        // Get backbone coordinates
        std::vector<float> coords1 = protein1.get_backbone_coords(chain1);
        std::vector<float> coords2 = protein2.get_backbone_coords(chain2);

        int L1 = protein1.get_chain(chain1).size();
        int L2 = protein2.get_chain(chain2).size();

        if (L1 != L2) {
            throw errors::DimensionError(
                "structure lengths",
                std::to_string(L1) + " vs " + std::to_string(L2),
                "equal lengths for superposition");
        }

        // Extract CA atoms
        std::vector<float> ca1(L1 * 3);
        std::vector<float> ca2(L2 * 3);
        structural_metrics::extract_ca_atoms(coords1.data(), L1, ca1.data());
        structural_metrics::extract_ca_atoms(coords2.data(), L2, ca2.data());

        // Compute Kabsch transformation
        std::vector<float> R(9);
        std::vector<float> t(3);
        float rmsd;
        pfalign::kabsch::kabsch_align<ScalarBackend>(ca1.data(), ca2.data(), L1, R.data(),
                                                     t.data(), &rmsd, nullptr, nullptr);

        // Write JSON output
        std::ofstream out(output_path);
        if (!out.is_open()) {
            throw errors::FileWriteError(output_path);
        }

        out << "{\n";
        out << "  \"rotation\": [\n";
        out << "    [" << R[0] << ", " << R[1] << ", " << R[2] << "],\n";
        out << "    [" << R[3] << ", " << R[4] << ", " << R[5] << "],\n";
        out << "    [" << R[6] << ", " << R[7] << ", " << R[8] << "]\n";
        out << "  ],\n";
        out << "  \"translation\": [" << t[0] << ", " << t[1] << ", " << t[2] << "],\n";
        out << "  \"rmsd\": " << rmsd << "\n";
        out << "}\n";
        out.close();

        if (print_rmsd) {
            std::cout << "RMSD: " << rmsd << "\n";
        }

        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int superpose(const std::string& /* mobile_path */, const std::string& /* reference_path */,
              const std::string& /* output_path */, int /* chain1 */, int /* chain2 */,
              const std::string& /* transform_path */, const std::string& /* metrics_path */) {
    std::cerr << "Error: superpose command not yet implemented\n";
    std::cerr << "  This command requires PDB writing functionality which is complex\n";
    std::cerr << "  Use the kabsch command to compute the transformation matrix\n";
    return 1;
}

}  // namespace commands
}  // namespace pfalign
