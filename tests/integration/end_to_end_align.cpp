/**
 * End-to-end protein alignment test.
 *
 * Pipeline:
 * 1. Parse two PDB files
 * 2. Compute MPNN embeddings for each protein
 * 3. Compute similarity matrix (dot product)
 * 4. Run Smith-Waterman alignment
 * 5. Save alignment matrix as numpy file
 *
 * Usage:
 *   ./end_to_end_align \
 *       <pdb1.pdb> \
 *       <pdb2.pdb> \
 *       <output.npy> \
 *       [--mode <sw_mode>] \
 *       [--gap <penalty>] \
 *       [--gap-open <penalty>] \
 *       [--gap-extend <penalty>] \
 *       [--temperature <temp>]
 *
 * SW Modes:
 *   direct_regular           Standard SW with single gap penalty
 *   direct_affine            Standard SW with affine gaps
 *   direct_affine_flexible   Standard SW with flexible affine gaps
 *   jax_regular              JAX-compatible SW with single gap penalty
 *   jax_affine               JAX-compatible SW with affine gaps
 *   jax_affine_flexible      JAX-compatible SW with flexible affine gaps (default)
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cmath>

// Modules
#include "pfalign/modules/mpnn/mpnn_encoder.h"
#include "pfalign/modules/similarity/similarity.h"
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/primitives/smith_waterman/smith_waterman.h"

// I/O
#include "pfalign/io/pdb_parser.h"
#include "pfalign/io/sequence_utils.h"
#include "pfalign/tools/weights/save_npy.h"
#include "pfalign/tools/weights/embedded_weights_loader.h"

// Primitives
#include "pfalign/primitives/alignment_decode/alignment_decode.h"
#include "pfalign/primitives/kabsch/kabsch_impl.h"
#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"

// Backends
#include "pfalign/dispatch/backend_traits.h"

// Profiling
#include "pfalign/common/profiling.h"
#include "pfalign/common/perf_timer.h"

using namespace pfalign;
using namespace pfalign::mpnn;
using namespace pfalign::io;
using pfalign::pairwise::AlignmentResult;

struct Args {
    std::string pdb1_path;
    std::string pdb2_path;
    std::string output_path;
    std::string sw_mode = "jax_affine_flexible";

    // Optional overrides (nullptr means use model defaults)
    float* gap_override = nullptr;
    float* gap_open_override = nullptr;
    float* gap_extend_override = nullptr;
    float* temperature_override = nullptr;

    // Profiling options
    bool profile_console = false;
    std::string profile_json_path;
    std::string profile_csv_path;

    ~Args() {
        delete gap_override;
        delete gap_open_override;
        delete gap_extend_override;
        delete temperature_override;
    }
};

bool parse_args(int argc, char** argv, Args& args) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <pdb1> <pdb2> <output.npy> [options]\n";
        std::cerr << "\nOptions:\n";
        std::cerr << "  --mode <sw_mode>         Smith-Waterman mode (default: jax_affine_flexible)\n";
        std::cerr << "  --gap <penalty>          Gap penalty (default: -0.1)\n";
        std::cerr << "  --gap-open <penalty>     Gap open penalty (default: -1.0)\n";
        std::cerr << "  --gap-extend <penalty>   Gap extend penalty (default: -0.1)\n";
        std::cerr << "  --temperature <temp>     Temperature (default: 1.0)\n";
        std::cerr << "\nProfiling Options:\n";
        std::cerr << "  --profile-console        Print profiling report to console\n";
        std::cerr << "  --profile-json <file>    Write profiling report to JSON file\n";
        std::cerr << "  --profile-csv <file>     Write profiling report to CSV file\n";
        std::cerr << "  --profile-all <prefix>   Write all formats (prefix.json, prefix.csv)\n";
        std::cerr << "\nSW Modes:\n";
        std::cerr << "  direct_regular\n";
        std::cerr << "  direct_affine\n";
        std::cerr << "  direct_affine_flexible\n";
        std::cerr << "  jax_regular\n";
        std::cerr << "  jax_affine\n";
        std::cerr << "  jax_affine_flexible (default)\n";
        return false;
    }

    args.pdb1_path = argv[1];
    args.pdb2_path = argv[2];
    args.output_path = argv[3];

    // Parse optional arguments
    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            args.sw_mode = argv[++i];
        } else if (arg == "--gap" && i + 1 < argc) {
            args.gap_override = new float(std::atof(argv[++i]));
        } else if (arg == "--gap-open" && i + 1 < argc) {
            args.gap_open_override = new float(std::atof(argv[++i]));
        } else if (arg == "--gap-extend" && i + 1 < argc) {
            args.gap_extend_override = new float(std::atof(argv[++i]));
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature_override = new float(std::atof(argv[++i]));
        } else if (arg == "--profile-console") {
            args.profile_console = true;
        } else if (arg == "--profile-json" && i + 1 < argc) {
            args.profile_json_path = argv[++i];
        } else if (arg == "--profile-csv" && i + 1 < argc) {
            args.profile_csv_path = argv[++i];
        } else if (arg == "--profile-all" && i + 1 < argc) {
            std::string prefix = argv[++i];
            args.profile_console = true;
            args.profile_json_path = prefix + ".json";
            args.profile_csv_path = prefix + ".csv";
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    pfalign::perf::PerfTimer perf_timer("end_to_end_align");
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "  End-to-End Protein Alignment\n";
    std::cout << "========================================\n\n";

    // ========================================================================
    // Step 1: Load MPNN weights and SW parameters
    // ========================================================================
    std::cout << "Step 1: Loading MPNN weights and SW parameters...\n";

    MPNNWeights weights(0);
    MPNNConfig config;
    weights::SWParams sw_params;

    {
        PROFILE_SCOPE("Load_weights");
        try {
            auto [w, c, p] = weights::load_embedded_mpnn_weights();
            weights = std::move(w);
            config = c;
            sw_params = p;
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to load embedded weights\n";
            std::cerr << "  Error: " << e.what() << "\n";
            return 0;
        }
    }

    // Apply command-line overrides
    if (args.gap_override) sw_params.gap = *args.gap_override;
    if (args.gap_open_override) sw_params.gap_open = *args.gap_open_override;
    if (args.gap_extend_override) sw_params.gap = *args.gap_extend_override;  // gap_extend maps to gap
    if (args.temperature_override) sw_params.temperature = *args.temperature_override;

    std::cout << "✓ Loaded weights\n";
    std::cout << "  Hidden dim: " << config.hidden_dim << "\n";
    std::cout << "  Num layers: " << config.num_layers << "\n";
    std::cout << "\n✓ Loaded SW parameters\n";
    std::cout << "  Gap extend:  " << sw_params.gap << (args.gap_extend_override ? " (overridden)" : " (from model)") << "\n";
    std::cout << "  Gap open:    " << sw_params.gap_open << (args.gap_open_override ? " (overridden)" : " (from model)") << "\n";
    std::cout << "  Temperature: " << sw_params.temperature << (args.temperature_override ? " (overridden)" : " (from model)") << "\n\n";

    // ========================================================================
    // Step 2: Parse PDB files
    // ========================================================================
    std::cout << "Step 2: Parsing PDB files...\n";

    PDBParser parser;
    Protein prot1, prot2;

    {
        PROFILE_SCOPE("Parse_PDB");
        try {
            prot1 = parser.parse_file(args.pdb1_path);
            std::cout << "✓ Parsed " << args.pdb1_path << "\n";
            std::cout << "  Chains: " << prot1.num_chains() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to parse " << args.pdb1_path << ": " << e.what() << "\n";
            return 1;
        }

        try {
            prot2 = parser.parse_file(args.pdb2_path);
            std::cout << "✓ Parsed " << args.pdb2_path << "\n";
            std::cout << "  Chains: " << prot2.num_chains() << "\n\n";
        } catch (const std::exception& e) {
            std::cerr << "✗ Failed to parse " << args.pdb2_path << ": " << e.what() << "\n";
            return 1;
        }
    }

    // Use first chain of each protein
    int L1 = prot1.get_chain(0).size();
    int L2 = prot2.get_chain(0).size();

    std::cout << "  Protein 1: " << L1 << " residues\n";
    std::cout << "  Protein 2: " << L2 << " residues\n\n";

    auto coords1 = prot1.get_backbone_coords(0);
    auto coords2 = prot2.get_backbone_coords(0);

    // Extract amino acid sequences (for FASTA output)
    std::string seq1, seq2;
    const auto& chain1 = prot1.get_chain(0);
    const auto& chain2 = prot2.get_chain(0);

    for (size_t i = 0; i < L1; i++) {
        char aa = three_to_one(chain1.residues[i].resn);
        seq1 += aa;
    }
    for (size_t i = 0; i < L2; i++) {
        char aa = three_to_one(chain2.residues[i].resn);
        seq2 += aa;
    }

    // ========================================================================
    // Step 3: Compute MPNN embeddings
    // ========================================================================
    std::cout << "Step 3: Computing MPNN embeddings...\n";

    // Allocate embedding buffers
    float* emb1 = new float[L1 * config.hidden_dim];
    float* emb2 = new float[L2 * config.hidden_dim];

    // Create workspaces
    MPNNWorkspace workspace1(L1, config.k_neighbors, config.hidden_dim);
    MPNNWorkspace workspace2(L2, config.k_neighbors, config.hidden_dim);

    // Initialize residue indices and chain labels
    for (int i = 0; i < L1; i++) {
        workspace1.residue_idx[i] = i;
        workspace1.chain_labels[i] = 0;
    }
    for (int i = 0; i < L2; i++) {
        workspace2.residue_idx[i] = i;
        workspace2.chain_labels[i] = 0;
    }

    // Encode protein 1
    std::cout << "  Encoding protein 1...\n";
    {
        PROFILE_SCOPE("MPNN_protein1");
        mpnn_forward<ScalarBackend>(
            coords1.data(),
            L1,
            weights,
            config,
            emb1,
            &workspace1
        );
    }

    // Encode protein 2
    std::cout << "  Encoding protein 2...\n";
    {
        PROFILE_SCOPE("MPNN_protein2");
        mpnn_forward<ScalarBackend>(
            coords2.data(),
            L2,
            weights,
            config,
            emb2,
            &workspace2
        );
    }

    std::cout << "✓ Computed embeddings\n";
    std::cout << "  Embedding 1: [" << L1 << " * " << config.hidden_dim << "]\n";
    std::cout << "  Embedding 2: [" << L2 << " * " << config.hidden_dim << "]\n\n";

    // Save node features for JAX comparison (protein 1 only, since self-alignment uses same)
    if (args.pdb1_path == args.pdb2_path) {
        std::vector<size_t> emb_shape = {static_cast<size_t>(L1), static_cast<size_t>(config.hidden_dim)};
        save_npy_float32("/tmp/crambin_node_features.npy", emb1, emb_shape);
        std::cout << "  Saved node features to /tmp/crambin_node_features.npy\n\n";
    }

    // ========================================================================
    // Step 4: Compute similarity matrix
    // ========================================================================
    std::cout << "Step 4: Computing similarity matrix...\n";

    float* similarity = new float[L1 * L2];

    {
        PROFILE_SCOPE("Similarity");
        similarity::compute_similarity<ScalarBackend>(
            emb1,
            emb2,
            similarity,
            L1,
            L2,
            config.hidden_dim
        );
    }

    std::cout << "✓ Computed similarity\n";
    std::cout << "  Shape: [" << L1 << " * " << L2 << "]\n";
    std::cout << "  Sample values [0,0:5]: ";
    for (int j = 0; j < std::min(5, L2); j++) {
        std::cout << similarity[j] << " ";
    }
    std::cout << "\n\n";

    // ========================================================================
    // Step 5: Run Smith-Waterman alignment (Forward + Backward)
    // ========================================================================
    std::cout << "Step 5: Running Smith-Waterman alignment...\n";
    std::cout << "  Mode: " << args.sw_mode << "\n";

    // Configure SW (using loaded parameters)
    smith_waterman::SWConfig sw_config;
    sw_config.gap = sw_params.gap;           // Gap extension
    sw_config.gap_open = sw_params.gap_open; // Gap opening
    sw_config.gap_extend = sw_params.gap;    // Gap extension (same as gap)
    sw_config.temperature = sw_params.temperature;

    std::cout << "  Gap extend:  " << sw_config.gap_extend << "\n";
    std::cout << "  Gap open:    " << sw_config.gap_open << "\n";
    std::cout << "  Temperature: " << sw_config.temperature << "\n";

    // Allocate buffers
    float* forward_dp = nullptr;  // Forward DP values
    float* posteriors = new float[L1 * L2];  // Posterior alignment matrix (always 2D)
    float partition = 0.0f;

    {
        PROFILE_SCOPE("SW_alignment");
        // Run forward + backward pass for the selected mode
        if (args.sw_mode == "direct_regular") {
        // Forward: alpha [L1+1 * L2+1]
        forward_dp = new float[(L1 + 1) * (L2 + 1)];
        smith_waterman::smith_waterman_direct_regular<ScalarBackend>(
            similarity, L1, L2, sw_config, forward_dp, &partition
        );

        // Backward: posteriors [L1 * L2]
        smith_waterman::smith_waterman_direct_regular_backward<ScalarBackend>(
            forward_dp, similarity, L1, L2, sw_config, partition, posteriors
        );

    } else if (args.sw_mode == "direct_affine") {
        // Forward: alpha [L1+1 * L2+1 * 3]
        forward_dp = new float[(L1 + 1) * (L2 + 1) * 3];
        smith_waterman::smith_waterman_direct_affine<ScalarBackend>(
            similarity, L1, L2, sw_config, forward_dp, &partition
        );

        // Backward: posteriors [L1 * L2]
        smith_waterman::smith_waterman_direct_affine_backward<ScalarBackend>(
            forward_dp, similarity, L1, L2, sw_config, partition, posteriors
        );

    } else if (args.sw_mode == "direct_affine_flexible") {
        // Forward: alpha [L1+1 * L2+1 * 3]
        forward_dp = new float[(L1 + 1) * (L2 + 1) * 3];
        smith_waterman::smith_waterman_direct_affine_flexible<ScalarBackend>(
            similarity, L1, L2, sw_config, forward_dp, &partition
        );

        // Backward: posteriors [L1 * L2]
        smith_waterman::smith_waterman_direct_affine_flexible_backward<ScalarBackend>(
            forward_dp, similarity, L1, L2, sw_config, partition, posteriors
        );

    } else if (args.sw_mode == "jax_regular") {
        // Forward: hij [L1 * L2]
        forward_dp = new float[L1 * L2];
        smith_waterman::smith_waterman_jax_regular<ScalarBackend>(
            similarity, L1, L2, sw_config, forward_dp, &partition
        );

        // Backward: posteriors [L1 * L2]
        smith_waterman::smith_waterman_jax_regular_backward<ScalarBackend>(
            forward_dp, similarity, L1, L2, sw_config, partition, posteriors
        );

    } else if (args.sw_mode == "jax_affine") {
        // Forward: hij [L1 * L2 * 3]
        forward_dp = new float[L1 * L2 * 3];
        smith_waterman::smith_waterman_jax_affine<ScalarBackend>(
            similarity, L1, L2, sw_config, forward_dp, &partition
        );

        // Backward: posteriors [L1 * L2]
        smith_waterman::smith_waterman_jax_affine_backward<ScalarBackend>(
            forward_dp, similarity, L1, L2, sw_config, partition, posteriors
        );

    } else if (args.sw_mode == "jax_affine_flexible") {
        // Forward: hij [L1 * L2 * 3]
        forward_dp = new float[L1 * L2 * 3];
        smith_waterman::smith_waterman_jax_affine_flexible<ScalarBackend>(
            similarity, L1, L2, sw_config, forward_dp, &partition
        );

        // Backward: posteriors [L1 * L2]
        smith_waterman::smith_waterman_jax_affine_flexible_backward<ScalarBackend>(
            forward_dp, similarity, L1, L2, sw_config, partition, posteriors
        );

        } else {
            std::cerr << "✗ Unknown SW mode: " << args.sw_mode << "\n";
            delete[] emb1;
            delete[] emb2;
            delete[] similarity;
            delete[] posteriors;
            return 1;
        }
    }  // End PROFILE_SCOPE("SW_alignment")

    std::cout << "✓ Alignment complete\n";
    std::cout << "  Partition function: " << partition << "\n";
    std::cout << "  Posteriors shape: [" << L1 << " * " << L2 << "]\n\n";

    // Compute alignment score (cosine similarity)
    float score = 0.0f;
    {
        PROFILE_SCOPE("Compute_score");
        // First compute norms for each embedding
        float* norms1 = new float[L1];
        float* norms2 = new float[L2];

        for (int i = 0; i < L1; i++) {
            float norm_sq = 0.0f;
        for (int d = 0; d < config.hidden_dim; d++) {
            float val = emb1[i * config.hidden_dim + d];
            norm_sq += val * val;
        }
        norms1[i] = std::sqrt(norm_sq);
    }

    for (int j = 0; j < L2; j++) {
        float norm_sq = 0.0f;
        for (int d = 0; d < config.hidden_dim; d++) {
            float val = emb2[j * config.hidden_dim + d];
            norm_sq += val * val;
        }
        norms2[j] = std::sqrt(norm_sq);
    }

        // Compute cosine similarity score: Sigma P[i,j] * (S[i,j] / ||e1[i]|| * ||e2[j]||) / Sigma P[i,j]
        float posterior_sum = 0.0f;

        for (int i = 0; i < L1; i++) {
            for (int j = 0; j < L2; j++) {
                int idx = i * L2 + j;
                float norm_product = norms1[i] * norms2[j];
                float cosine = (norm_product > 1e-10f) ? (similarity[idx] / norm_product) : 0.0f;
                score += posteriors[idx] * cosine;
                posterior_sum += posteriors[idx];
            }
        }

        if (posterior_sum > 0.0f) {
            score /= posterior_sum;
        }

        delete[] norms1;
        delete[] norms2;
    }  // End PROFILE_SCOPE("Compute_score")

    std::cout << "  Alignment score:    " << score << " (cosine similarity)\n\n";

    // ========================================================================
    // Step 6a: Decode alignment path from posteriors
    // ========================================================================
    std::cout << "Step 6a: Decoding alignment path...\n";

    int max_path_length = L1 + L2;
    AlignmentPair* alignment_path = new AlignmentPair[max_path_length];
    float* dp_score = new float[(L1 + 1) * (L2 + 1)];
    uint8_t* dp_traceback = new uint8_t[(L1 + 1) * (L2 + 1)];

    int path_length;
    {
        PROFILE_SCOPE("Decode_alignment");
        path_length = alignment_decode::decode_alignment<ScalarBackend>(
            posteriors, L1, L2,
            -2.0f,  // gap_penalty
            alignment_path, max_path_length,
            dp_score, dp_traceback
        );
    }

    if (path_length < 0) {
        std::cerr << "✗ Alignment decoding failed\n";
        delete[] alignment_path;
        delete[] dp_score;
        delete[] dp_traceback;
        delete[] emb1;
        delete[] emb2;
        delete[] similarity;
        delete[] forward_dp;
        delete[] posteriors;
        return 1;
    }

    std::cout << "✓ Decoded alignment path: " << path_length << " positions\n\n";

    // Clean up temporary decode buffers
    delete[] dp_score;
    delete[] dp_traceback;

    // ========================================================================
    // Step 6b: Save posterior alignment matrix
    // ========================================================================
    std::cout << "Step 6b: Saving posterior alignment matrix...\n";

    std::vector<size_t> shape = {static_cast<size_t>(L1), static_cast<size_t>(L2)};
    bool success = save_npy_float32(args.output_path, posteriors, shape);

    if (!success) {
        std::cerr << "✗ Failed to save output to " << args.output_path << "\n";
        delete[] emb1;
        delete[] emb2;
        delete[] similarity;
        delete[] forward_dp;
        delete[] posteriors;
        return 1;
    }

    std::cout << "✓ Saved posterior alignment matrix\n";
    std::cout << "  File: " << args.output_path << "\n";
    std::cout << "  Shape: [" << L1 << " * " << L2 << "]\n\n";

    // Prepare reusable AlignmentResult wrapper for downstream outputs
    AlignmentResult alignment_result;
    alignment_result.partition = partition;
    alignment_result.score = score;
    alignment_result.L1 = L1;
    alignment_result.L2 = L2;
    alignment_result.path_length = path_length;
    alignment_result.max_path_length = max_path_length;
    alignment_result.posteriors = posteriors;
    alignment_result.alignment_path = alignment_path;
    alignment_result.coords1 = coords1.data();
    alignment_result.coords2 = coords2.data();
    alignment_result.protein1 = &prot1;
    alignment_result.protein2 = &prot2;

    // ========================================================================
    // Step 6c: Write FASTA alignment
    // ========================================================================
    std::cout << "Step 6c: Writing FASTA alignment...\n";

    // Generate FASTA filename (replace .npy with .fasta)
    std::string fasta_path = args.output_path;
    size_t ext_pos = fasta_path.rfind(".npy");
    if (ext_pos != std::string::npos) {
        fasta_path.replace(ext_pos, 4, ".fasta");
    } else {
        fasta_path += ".fasta";
    }

    // Extract protein names from PDB paths (just the filename)
    std::string name1 = args.pdb1_path;
    std::string name2 = args.pdb2_path;
    size_t pos1 = name1.find_last_of("/\\");
    size_t pos2 = name2.find_last_of("/\\");
    if (pos1 != std::string::npos) name1 = name1.substr(pos1 + 1);
    if (pos2 != std::string::npos) name2 = name2.substr(pos2 + 1);

    alignment_result.id1 = name1;
    alignment_result.id2 = name2;

    bool fasta_success;
    {
        PROFILE_SCOPE("Write_FASTA");
        fasta_success = alignment_result.write_fasta(
            fasta_path,
            seq1,
            seq2
        );
    }

    if (!fasta_success) {
        std::cerr << "✗ Failed to write FASTA to " << fasta_path << "\n";
        // Non-fatal error - continue with cleanup
    } else {
        std::cout << "✓ Saved FASTA alignment\n";
        std::cout << "  File: " << fasta_path << "\n\n";
    }

    // ========================================================================
    // Step 7: Structural Alignment (Kabsch + TM-score + GDT + PDB output)
    // ========================================================================
    std::cout << "Step 7: Computing structural alignment...\n";

    // Extract aligned CA coordinates from alignment_path
    std::vector<float> ca1_aligned, ca2_aligned;
    std::vector<std::pair<int,int>> match_pairs;

    for (int k = 0; k < path_length; k++) {
        if (alignment_path[k].i >= 0 && alignment_path[k].j >= 0) {
            int i = alignment_path[k].i;
            int j = alignment_path[k].j;
            match_pairs.push_back({i, j});

            // Extract CA coordinates (atom index 1, coords at offset 3-5 in flattened layout)
            // Layout: [residue*12 + atom*3 + coord]
            ca1_aligned.push_back(coords1[i * 12 + 3]);  // CA_x
            ca1_aligned.push_back(coords1[i * 12 + 4]);  // CA_y
            ca1_aligned.push_back(coords1[i * 12 + 5]);  // CA_z

            ca2_aligned.push_back(coords2[j * 12 + 3]);  // CA_x
            ca2_aligned.push_back(coords2[j * 12 + 4]);  // CA_y
            ca2_aligned.push_back(coords2[j * 12 + 5]);  // CA_z
        }
    }

    int N_aligned = match_pairs.size();
    std::cout << "  Aligned pairs: " << N_aligned << " CA atoms\n";

    // Kabsch alignment
    float R[9], t[3], rmsd;
    {
        PROFILE_SCOPE("Kabsch_alignment");
        kabsch::kabsch_align<ScalarBackend>(
            ca1_aligned.data(), ca2_aligned.data(), N_aligned,
            R, t, &rmsd
        );
    }

    // Apply transformation to full structure 1
    std::vector<float> coords1_full_aligned(L1 * 4 * 3);
    {
        PROFILE_SCOPE("Kabsch_apply_transformation");
        kabsch::apply_transformation<ScalarBackend>(
            R, t, coords1.data(), coords1_full_aligned.data(), L1
        );
    }

    // Extract transformed aligned CA coordinates for metrics
    std::vector<float> ca1_transformed;
    for (const auto& [i, j] : match_pairs) {
        ca1_transformed.push_back(coords1_full_aligned[i * 12 + 3]);  // CA_x
        ca1_transformed.push_back(coords1_full_aligned[i * 12 + 4]);  // CA_y
        ca1_transformed.push_back(coords1_full_aligned[i * 12 + 5]);  // CA_z
    }

    // Compute structural metrics using TRANSFORMED coordinates
    float tm1, tm2;
    {
        PROFILE_SCOPE("Compute_TM_score");
        tm1 = structural_metrics::compute_tm_score<ScalarBackend>(
            ca1_transformed.data(), ca2_aligned.data(), N_aligned, L1
        );
        tm2 = structural_metrics::compute_tm_score<ScalarBackend>(
            ca1_transformed.data(), ca2_aligned.data(), N_aligned, L2
        );
    }

    float gdt_ts, gdt_ha;
    {
        PROFILE_SCOPE("Compute_GDT");
        structural_metrics::compute_gdt<ScalarBackend>(
            ca1_transformed.data(), ca2_aligned.data(), N_aligned,
            &gdt_ts, &gdt_ha
        );
    }

    // Print TMalign-style output
    std::cout << "\n=== Structural Alignment Metrics ===\n";
    std::cout << "Length of Structure 1: " << L1 << " residues\n";
    std::cout << "Length of Structure 2: " << L2 << " residues\n";
    std::cout << "Aligned length:        " << N_aligned << " CA atoms\n";
    std::cout << "RMSD:                  " << std::fixed << std::setprecision(3) << rmsd << " Å\n";
    std::cout << "TM-score (norm by L1): " << std::fixed << std::setprecision(5) << tm1 << " (L=" << L1 << ")\n";
    std::cout << "TM-score (norm by L2): " << std::fixed << std::setprecision(5) << tm2 << " (L=" << L2 << ")\n";
    std::cout << "GDT-TS:                " << std::fixed << std::setprecision(5) << gdt_ts << "\n";
    std::cout << "GDT-HA:                " << std::fixed << std::setprecision(5) << gdt_ha << "\n\n";

    // Write aligned PDB
    std::string pdb_path = args.output_path;
    size_t pdb_ext_pos = pdb_path.rfind(".npy");
    if (pdb_ext_pos != std::string::npos) {
        pdb_path.replace(pdb_ext_pos, 4, "_aligned.pdb");
    } else {
        pdb_path += "_aligned.pdb";
    }

    bool pdb_success = alignment_result.write_superposed_pdb(
        pdb_path,
        /*reference=*/1  // Align protein 1 onto protein 2 (match legacy output)
    );

    if (pdb_success) {
        std::cout << "✓ Wrote aligned structures to " << pdb_path << "\n\n";
    } else {
        std::cerr << "✗ Failed to write PDB output\n\n";
    }

    // Clean up alignment path
    delete[] alignment_path;

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "========================================\n";
    std::cout << "  Summary\n";
    std::cout << "========================================\n";
    std::cout << "Protein 1:    " << args.pdb1_path << " (" << L1 << " residues)\n";
    std::cout << "Protein 2:    " << args.pdb2_path << " (" << L2 << " residues)\n";
    std::cout << "SW Mode:      " << args.sw_mode << "\n";
    std::cout << "Gap extend:   " << sw_config.gap_extend << "\n";
    std::cout << "Gap open:     " << sw_config.gap_open << "\n";
    std::cout << "Temperature:  " << sw_config.temperature << "\n";
    std::cout << "Partition:    " << partition << "\n";
    std::cout << "Output shape: [" << L1 << " * " << L2 << "] (posteriors)\n";
    std::cout << "Posteriors:   " << args.output_path << "\n";
    std::cout << "FASTA:        " << fasta_path << " (aligned sequences with gaps)\n";
    std::cout << "========================================\n";

    // ========================================================================
    // Profiling output (if enabled)
    // ========================================================================
    if (args.profile_console) {
        std::cout << "\n";
        PROFILE_PRINT();
    }
    if (!args.profile_json_path.empty()) {
        PROFILE_WRITE_JSON(args.profile_json_path);
        std::cout << "Profiling JSON written to: " << args.profile_json_path << "\n";
    }
    if (!args.profile_csv_path.empty()) {
        PROFILE_WRITE_CSV(args.profile_csv_path);
        std::cout << "Profiling CSV written to: " << args.profile_csv_path << "\n";
    }

    // Cleanup
    delete[] emb1;
    delete[] emb2;
    delete[] similarity;
    delete[] forward_dp;
    delete[] posteriors;

    return 0;
}
