/**
 * Quick debug: Print V2 intermediate values at key stages.
 *
 * This is a simpler alternative to full intermediate dumping.
 * Prints first 5 values at each stage for manual comparison with V1.
 *
 * Usage:
 *   ./quick_debug_v2 <input.pdb>
 */

#include "pfalign/tools/weights/embedded_weights_loader.h"
#include "pfalign/io/pdb_parser.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/primitives/knn/knn_impl.h"
#include "pfalign/primitives/rbf/rbf_impl.h"
#include "pfalign/primitives/gemm/gemm_impl.h"
#include "pfalign/primitives/layer_norm/layer_norm_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>


using namespace pfalign;

#define PRINT_ARRAY(name, arr, n) do { \
    std::cout << name << "[0:5]: "; \
    for (int i = 0; i < std::min(n, 5); i++) { \
        std::cout << std::setw(12) << std::fixed << std::setprecision(6) << arr[i] << " "; \
    } \
    std::cout << std::endl; \
} while(0)

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input.pdb>" << std::endl;
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);

    // Load weights
    mpnn::MPNNWeights weights(3);
    mpnn::MPNNConfig config;

    try {
        auto [loaded_weights, loaded_config, sw_defaults] = weights::load_embedded_mpnn_weights();
        weights = std::move(loaded_weights);
        config = loaded_config;
        (void)sw_defaults;
    } catch (const std::exception& e) {
        std::cerr << "SKIP: Could not load embedded MPNN weights (" << e.what() << ")" << std::endl;
        return 0;
    }

    std::cout << "Config: hidden=" << config.hidden_dim
              << " layers=" << config.num_layers
              << " rbf=" << config.num_rbf << std::endl;

    // Load PDB
    io::PDBParser parser;
    io::Protein protein = parser.parse_file(argv[1]);
    int L = protein.get_chain(0).size();
    auto coords = protein.get_backbone_coords(0);

    std::cout << "Protein: L=" << L << std::endl;
    std::cout << "\n========================================\n";

    // Manual forward pass with prints
    mpnn::MPNNWorkspace ws(L, config.k_neighbors, config.hidden_dim);

    // Step 1: Extract atoms
    for (int i = 0; i < L; i++) {
        const float* residue = coords.data() + i * 4 * 3;
        // Ca
        ws.Ca[i * 3 + 0] = residue[1 * 3 + 0];
        ws.Ca[i * 3 + 1] = residue[1 * 3 + 1];
        ws.Ca[i * 3 + 2] = residue[1 * 3 + 2];
    }

    std::cout << "STAGE: Input Ca coordinates" << std::endl;
    PRINT_ARRAY("Ca_x", ws.Ca, L);
    PRINT_ARRAY("Ca_y", ws.Ca + 1, L);
    PRINT_ARRAY("Ca_z", ws.Ca + 2, L);

    // Step 2: KNN
    pfalign::knn::knn_search<ScalarBackend>(
        ws.Ca, L, config.k_neighbors,
        ws.neighbor_indices, ws.neighbor_distances_sq
    );

    std::cout << "\nSTAGE: KNN indices" << std::endl;
    PRINT_ARRAY("KNN[0]", ws.neighbor_indices, config.k_neighbors);

    // Skip RBF computation for brevity... would need full atom extraction

    std::cout << "\n========================================" << std::endl;
    std::cout << "Note: Full debug requires implementing all stages" << std::endl;
    std::cout << "See DEBUG_PLAN.md for complete approach" << std::endl;

    return 0;
}
