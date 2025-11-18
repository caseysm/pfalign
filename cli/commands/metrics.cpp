#include "commands.h"
#include "input_utils.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "pfalign/dispatch/backend_traits.h"
#include "pfalign/errors/messages.h"
#include "pfalign/errors/validators.h"
#include "pfalign/io/protein_structure.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/primitives/structural_metrics/distance_matrix.h"
#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"

namespace pfalign {
namespace commands {

int compute_rmsd(const std::string& struct1_path, const std::string& struct2_path, int chain1,
                 int chain2, bool aligned) {
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
                "equal lengths for alignment");
        }

        // Extract CA atoms
        std::vector<float> ca1(L1 * 3);
        std::vector<float> ca2(L2 * 3);
        structural_metrics::extract_ca_atoms(coords1.data(), L1, ca1.data());
        structural_metrics::extract_ca_atoms(coords2.data(), L2, ca2.data());

        float rmsd;
        if (aligned) {
            // Already aligned, compute RMSD manually
            float sum_sq = 0.0f;
            for (int i = 0; i < L1; i++) {
                float dx = ca1[i*3 + 0] - ca2[i*3 + 0];
                float dy = ca1[i*3 + 1] - ca2[i*3 + 1];
                float dz = ca1[i*3 + 2] - ca2[i*3 + 2];
                sum_sq += dx*dx + dy*dy + dz*dz;
            }
            rmsd = std::sqrt(sum_sq / L1);
        } else {
            // Perform Kabsch alignment first
            std::vector<float> R(9);
            std::vector<float> t(3);
            kabsch::kabsch_align<ScalarBackend>(ca1.data(), ca2.data(), L1, R.data(), t.data(),
                                                &rmsd, nullptr, nullptr);
        }

        std::cout << rmsd << "\n";
        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int compute_tm_score(const std::string& struct1_path, const std::string& struct2_path,
                     int chain1, int chain2, bool aligned) {
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
                "equal lengths for alignment");
        }

        // Extract CA atoms
        std::vector<float> ca1(L1 * 3);
        std::vector<float> ca2(L2 * 3);
        structural_metrics::extract_ca_atoms(coords1.data(), L1, ca1.data());
        structural_metrics::extract_ca_atoms(coords2.data(), L2, ca2.data());

        // Perform Kabsch alignment if needed
        if (!aligned) {
            std::vector<float> R(9);
            std::vector<float> t(3);
            kabsch::kabsch_align<ScalarBackend>(ca1.data(), ca2.data(), L1, R.data(), t.data(),
                                                nullptr, nullptr, nullptr);

            // Apply transformation to ca1
            for (int i = 0; i < L1; i++) {
                float x = ca1[i * 3 + 0];
                float y = ca1[i * 3 + 1];
                float z = ca1[i * 3 + 2];
                ca1[i * 3 + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
                ca1[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
                ca1[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
            }
        }

        float tm_score = structural_metrics::compute_tm_score<ScalarBackend>(ca1.data(), ca2.data(),
                                                                              L1, L1);
        std::cout << tm_score << "\n";
        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int compute_gdt(const std::string& struct1_path, const std::string& struct2_path, int chain1,
                int chain2, bool aligned) {
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
                "equal lengths for alignment");
        }

        // Extract CA atoms
        std::vector<float> ca1(L1 * 3);
        std::vector<float> ca2(L2 * 3);
        structural_metrics::extract_ca_atoms(coords1.data(), L1, ca1.data());
        structural_metrics::extract_ca_atoms(coords2.data(), L2, ca2.data());

        // Perform Kabsch alignment if needed
        if (!aligned) {
            std::vector<float> R(9);
            std::vector<float> t(3);
            kabsch::kabsch_align<ScalarBackend>(ca1.data(), ca2.data(), L1, R.data(), t.data(),
                                                nullptr, nullptr, nullptr);

            // Apply transformation to ca1
            for (int i = 0; i < L1; i++) {
                float x = ca1[i * 3 + 0];
                float y = ca1[i * 3 + 1];
                float z = ca1[i * 3 + 2];
                ca1[i * 3 + 0] = R[0] * x + R[1] * y + R[2] * z + t[0];
                ca1[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + t[1];
                ca1[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + t[2];
            }
        }

        float gdt_ts, gdt_ha;
        structural_metrics::compute_gdt<ScalarBackend>(ca1.data(), ca2.data(), L1, &gdt_ts,
                                                       &gdt_ha);

        std::cout << "GDT-TS: " << gdt_ts << "\n";
        std::cout << "GDT-HA: " << gdt_ha << "\n";
        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int compute_identity(const std::string& alignment_path, bool ignore_gaps) {
    try {
        // Read FASTA file
        std::ifstream file(alignment_path);
        if (!file.is_open()) {
            throw errors::FileNotFoundError(alignment_path, "alignment file");
        }

        std::string seq1, seq2;
        std::string line;
        int seq_count = 0;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (line[0] == '>') {
                seq_count++;
                continue;
            }

            if (seq_count == 1) {
                seq1 += line;
            } else if (seq_count == 2) {
                seq2 += line;
            }
        }

        if (seq1.empty() || seq2.empty()) {
            throw errors::ValidationError(
                "Failed to read sequences from alignment file",
                "Ensure file is in valid FASTA format with at least 2 sequences");
        }

        if (seq1.length() != seq2.length()) {
            throw errors::DimensionError(
                "aligned sequences",
                std::to_string(seq1.length()) + " vs " + std::to_string(seq2.length()),
                "equal lengths");
        }

        // Compute identity manually
        int matches = 0;
        int total = 0;

        for (size_t i = 0; i < seq1.length(); i++) {
            char c1 = seq1[i];
            char c2 = seq2[i];

            if (ignore_gaps && (c1 == '-' || c2 == '-')) {
                continue;  // Skip gap positions
            }

            total++;
            if (c1 == c2) {
                matches++;
            }
        }

        float identity = (total > 0) ? (static_cast<float>(matches) / total) : 0.0f;
        std::cout << identity << "\n";
        return 0;

    } catch (const errors::PFalignError& e) {
        std::cerr << e.formatted() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int compute_ecs(const std::string& /* msa_path */, float /* temperature */) {
    std::cerr << "Error: compute-ecs command not yet implemented\n";
    std::cerr << "  This command requires MSA parsing which is not yet available in the CLI\n";
    return 1;
}

}  // namespace commands
}  // namespace pfalign
