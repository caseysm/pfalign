#include "pfalign/cli/cli.h"
#include "commands/commands.h"
#include <iostream>

using namespace pfalign::cli;

int main(int argc, char** argv) {
    App app("pfalign", "Protein structure alignment using MPNN embeddings");

    // ========== Global Flags ==========
    bool flag_quiet = false;
    bool flag_progress = false;
    bool flag_stats = false;

    app.add_flag("--quiet", flag_quiet, "Suppress informational output (only show errors)");
    app.add_flag("--progress", flag_progress, "Show progress bars for long-running operations");
    app.add_flag("--stats", flag_stats, "Show detailed statistics after completion");

    // ========== Encode Subcommand ==========
    App* encode_cmd = app.add_subcommand("encode", "Encode PDB to MPNN embeddings");

    std::string encode_pdb, encode_output;
    int encode_k_neighbors = 30;
    int encode_chain = 0;
    encode_cmd->add_positional("input", encode_pdb, "Input PDB file")->check(ExistingFile());
    encode_cmd->add_positional("output", encode_output, "Output embeddings (.npy)");
    encode_cmd
        ->add_option("--k-neighbors", encode_k_neighbors,
                     "Number of nearest neighbors for MPNN (default: 30)")
        ->check(Range(1, 100));
    encode_cmd->add_option("--chain", encode_chain, "Chain index to encode (default: 0)")
        ->check(Range(0, 100));

    // ========== Encode Batch Subcommand ==========
    App* encode_batch_cmd = app.add_subcommand("encode-batch", "Batch encode multiple structures");

    std::vector<std::string> encode_batch_inputs;
    std::string encode_batch_input_list;
    std::string encode_batch_input_dir;
    std::string encode_batch_output_dir;
    int encode_batch_k_neighbors = 30;
    int encode_batch_chain = 0;

    auto* encode_batch_positional = encode_batch_cmd->add_positional("inputs", encode_batch_inputs, "Input paths (optional)");
    encode_batch_positional->required(false)->consume_remaining();
    encode_batch_cmd->add_option("--input-list", encode_batch_input_list, "File containing list of inputs")
        ->check(ExistingFile());
    encode_batch_cmd->add_option("--input-dir", encode_batch_input_dir, "Directory containing inputs");
    encode_batch_cmd->add_option("--output-dir", encode_batch_output_dir, "Output directory for embeddings")->required(true);
    encode_batch_cmd->add_option("--k-neighbors", encode_batch_k_neighbors,
                                 "Number of nearest neighbors for MPNN (default: 30)")
        ->check(Range(1, 100));
    encode_batch_cmd->add_option("--chain", encode_batch_chain, "Chain index to encode (default: 0)")
        ->check(Range(0, 100));

    // ========== Similarity Subcommand ==========
    App* sim_cmd = app.add_subcommand("similarity", "Compute similarity matrix between embeddings");

    std::string sim_emb1, sim_emb2, sim_output;
    sim_cmd->add_positional("emb1", sim_emb1, "First embedding (.npy)")->check(ExistingFile());
    sim_cmd->add_positional("emb2", sim_emb2, "Second embedding (.npy)")->check(ExistingFile());
    sim_cmd->add_positional("output", sim_output, "Output similarity matrix (.npy)");

    // ========== Compute Distances Subcommand ==========
    App* distances_cmd = app.add_subcommand("compute-distances", "Compute distance matrix between embeddings");

    std::vector<std::string> distances_inputs;
    std::string distances_input_list;
    std::string distances_input_dir;
    std::string distances_output;
    float distances_gap_open = -2.544f;  // Trained model default
    float distances_gap_extend = 0.194f;  // Trained model default
    float distances_temperature = 1.0f;

    auto* distances_positional = distances_cmd->add_positional("inputs", distances_inputs, "Input embedding paths (optional)");
    distances_positional->required(false)->consume_remaining();
    distances_cmd->add_option("--input-list", distances_input_list, "File containing list of embeddings")
        ->check(ExistingFile());
    distances_cmd->add_option("--input-dir", distances_input_dir, "Directory containing embeddings");
    distances_cmd->add_option("--output", distances_output, "Output distance matrix (.npy)")->required(true);
    distances_cmd->add_option("--gap-open", distances_gap_open, "Gap opening penalty (default: -2.544, trained model)");
    distances_cmd->add_option("--gap-extend", distances_gap_extend, "Gap extension penalty (default: 0.194, trained model)");
    distances_cmd->add_option("--temperature", distances_temperature, "Alignment temperature (default: 1.0)")
        ->check(Range(0.1, 10.0));

    // ========== Pairwise Subcommand ==========
    App* pairwise_cmd =
        app.add_subcommand("pairwise", "Pairwise alignment (structures or embeddings)");

    std::string pair_input1, pair_input2, pair_fasta, pair_posterior, pair_superpose, pair_metrics;
    std::string pair_format = "";  // Auto-detect from extension if empty
    float pair_gap_open = -2.544f;  // Trained model default
    float pair_gap_extend = 0.194f;  // Trained model default
    float pair_temperature = 1.0f;
    int pair_k_neighbors = 30;
    int pair_chain1 = 0;
    int pair_chain2 = 0;
    bool pair_disable_parallel_mpnn = false;

    pairwise_cmd->add_positional("input1", pair_input1, "First input (.pdb/.cif/.npy)")
        ->check(ExistingFile());
    pairwise_cmd->add_positional("input2", pair_input2, "Second input (.pdb/.cif/.npy)")
        ->check(ExistingFile());
    pairwise_cmd->add_option("--output", pair_fasta, "Output FASTA path (A3M/FASTA)");
    pairwise_cmd->add_option("--format", pair_format, "Output format (auto-detect from extension if not specified). Options: fas, a2m, a3m, sto, psi, clu");
    pairwise_cmd->add_option("--posterior", pair_posterior, "Optional posterior matrix (.npy)");
    pairwise_cmd->add_option("--superpose", pair_superpose, "Optional superposed PDB output");
    pairwise_cmd->add_option("--metrics", pair_metrics, "Optional metrics text output");
    pairwise_cmd->add_option("--gap-open", pair_gap_open, "Gap opening penalty (default: -2.544, trained model)");
    pairwise_cmd->add_option("--gap-extend", pair_gap_extend, "Gap extension penalty (default: 0.194, trained model)");
    pairwise_cmd
        ->add_option("--temperature", pair_temperature, "Alignment temperature (default: 1.0)")
        ->check(Range(0.1, 10.0));
    pairwise_cmd
        ->add_option("--k-neighbors", pair_k_neighbors, "MPNN neighbors (structures, default: 30)")
        ->check(Range(1, 100));
    pairwise_cmd
        ->add_option("--chain1", pair_chain1, "Chain index for first structure (default: 0)")
        ->check(Range(0, 100));
    pairwise_cmd
        ->add_option("--chain2", pair_chain2, "Chain index for second structure (default: 0)")
        ->check(Range(0, 100));
    pairwise_cmd->add_flag("--disable-mpnn-parallel", pair_disable_parallel_mpnn,
                           "Disable parallel MPNN encoding for pairwise alignment");

    // ========== MSA Subcommand ==========
    App* msa_cmd =
        app.add_subcommand("msa", "Multiple sequence alignment (structures or embeddings)");

    std::vector<std::string> msa_inputs;
    std::string msa_input_list;
    std::string msa_input_dir;
    std::string msa_output;
    std::string msa_format = "";  // Auto-detect from extension if empty
    std::string msa_superpose;
    std::string msa_metrics;
    std::string msa_method = "upgma";
    float msa_gap_open = -2.544f;  // Trained model default
    float msa_gap_extend = 0.194f;  // Trained model default
    float msa_temperature = 1.0f;
    float msa_ecs_temperature = 5.0f;
    int msa_k_neighbors = 30;
    int msa_arena_size_mb = 200;
    int msa_threads = 0;

    auto* msa_positional = msa_cmd->add_positional("inputs", msa_inputs, "Input paths (optional)");
    msa_positional->required(false)->consume_remaining();
    msa_cmd->add_option("--input-list", msa_input_list, "File containing list of inputs")
        ->check(ExistingFile());
    msa_cmd->add_option("--input-dir", msa_input_dir, "Directory containing inputs");
    msa_cmd->add_option("--output", msa_output, "Output FASTA path")->required(true);
    msa_cmd->add_option("--format", msa_format, "Output format (auto-detect from extension if not specified). Options: fas, a2m, a3m, sto, psi, clu");
    msa_cmd->add_option("--method", msa_method, "Guide tree method (upgma|nj|bionj|mst)");
    msa_cmd->add_option("--gap-open", msa_gap_open, "Gap opening penalty (default: -2.544, trained model)");
    msa_cmd->add_option("--gap-extend", msa_gap_extend, "Gap extension penalty (default: 0.194, trained model)");
    msa_cmd->add_option("--temperature", msa_temperature, "Alignment temperature (default: 1.0)")
        ->check(Range(0.1, 10.0));
    msa_cmd
        ->add_option("--ecs-temperature", msa_ecs_temperature,
                     "ECS weighting temperature (default: 5.0)")
        ->check(Range(0.1, 100.0));
    msa_cmd
        ->add_option("--k-neighbors", msa_k_neighbors, "MPNN neighbors (structures, default: 30)")
        ->check(Range(1, 100));
    msa_cmd->add_option("--arena-size-mb", msa_arena_size_mb, "Arena size in MB (default: 200)")
        ->check(Range(50, 4096));
    msa_cmd->add_option("--threads", msa_threads, "Worker threads (default: auto)")
        ->check(Range(0, 1024));
    msa_cmd->add_option("--superpose", msa_superpose, "Optional superposed PDB output");
    msa_cmd->add_option("--metrics", msa_metrics, "Optional metrics text output");

    // ========== METRICS Command ==========
    App* metrics_cmd = app.add_subcommand("metrics", "Compute quality metrics for alignments and structures");
    metrics_cmd->require_subcommand(true);

    // Metrics: RMSD subcommand
    App* metrics_rmsd_cmd = metrics_cmd->add_subcommand("rmsd", "Compute RMSD between structures");
    std::string metrics_rmsd_struct1, metrics_rmsd_struct2;
    int metrics_rmsd_chain1 = 0, metrics_rmsd_chain2 = 0;
    bool metrics_rmsd_aligned = false;
    metrics_rmsd_cmd->add_positional("struct1", metrics_rmsd_struct1, "First structure")->check(ExistingFile());
    metrics_rmsd_cmd->add_positional("struct2", metrics_rmsd_struct2, "Second structure")->check(ExistingFile());
    metrics_rmsd_cmd->add_option("--chain1", metrics_rmsd_chain1, "Chain index for first structure (default: 0)");
    metrics_rmsd_cmd->add_option("--chain2", metrics_rmsd_chain2, "Chain index for second structure (default: 0)");
    metrics_rmsd_cmd->add_flag("--aligned", metrics_rmsd_aligned, "Assume structures are already superposed");

    // Metrics: TM-score subcommand
    App* metrics_tm_cmd = metrics_cmd->add_subcommand("tm-score", "Compute TM-score between structures");
    std::string metrics_tm_struct1, metrics_tm_struct2;
    int metrics_tm_chain1 = 0, metrics_tm_chain2 = 0;
    bool metrics_tm_aligned = false;
    metrics_tm_cmd->add_positional("struct1", metrics_tm_struct1, "First structure")->check(ExistingFile());
    metrics_tm_cmd->add_positional("struct2", metrics_tm_struct2, "Second structure")->check(ExistingFile());
    metrics_tm_cmd->add_option("--chain1", metrics_tm_chain1, "Chain index for first structure (default: 0)");
    metrics_tm_cmd->add_option("--chain2", metrics_tm_chain2, "Chain index for second structure (default: 0)");
    metrics_tm_cmd->add_flag("--aligned", metrics_tm_aligned, "Assume structures are already superposed");

    // Metrics: GDT subcommand
    App* metrics_gdt_cmd = metrics_cmd->add_subcommand("gdt", "Compute GDT-TS and GDT-HA scores");
    std::string metrics_gdt_struct1, metrics_gdt_struct2;
    int metrics_gdt_chain1 = 0, metrics_gdt_chain2 = 0;
    bool metrics_gdt_aligned = false;
    metrics_gdt_cmd->add_positional("struct1", metrics_gdt_struct1, "First structure")->check(ExistingFile());
    metrics_gdt_cmd->add_positional("struct2", metrics_gdt_struct2, "Second structure")->check(ExistingFile());
    metrics_gdt_cmd->add_option("--chain1", metrics_gdt_chain1, "Chain index for first structure (default: 0)");
    metrics_gdt_cmd->add_option("--chain2", metrics_gdt_chain2, "Chain index for second structure (default: 0)");
    metrics_gdt_cmd->add_flag("--aligned", metrics_gdt_aligned, "Assume structures are already superposed");

    // Metrics: Identity subcommand
    App* metrics_identity_cmd = metrics_cmd->add_subcommand("identity", "Compute sequence identity from alignment");
    std::string metrics_identity_path;
    bool metrics_identity_ignore_gaps = true;
    metrics_identity_cmd->add_positional("alignment", metrics_identity_path, "Alignment file (FASTA)")->check(ExistingFile());
    metrics_identity_cmd->add_flag("--ignore-gaps", metrics_identity_ignore_gaps, "Ignore gaps in identity calculation (default: true)");

    // Metrics: ECS subcommand
    App* metrics_ecs_cmd = metrics_cmd->add_subcommand("ecs", "Compute ECS score from MSA");
    std::string metrics_ecs_path;
    float metrics_ecs_temperature = 5.0f;
    metrics_ecs_cmd->add_positional("msa", metrics_ecs_path, "MSA file (FASTA)")->check(ExistingFile());
    metrics_ecs_cmd->add_option("--temperature", metrics_ecs_temperature, "ECS temperature (default: 5.0)");

    // ========== STRUCTURE Command ==========
    App* structure_cmd = app.add_subcommand("structure", "Structural superposition and alignment");
    structure_cmd->require_subcommand(true);

    // Structure: Superpose subcommand
    App* structure_superpose_cmd = structure_cmd->add_subcommand("superpose", "Superpose two structures");
    std::string structure_superpose_mobile, structure_superpose_ref, structure_superpose_output;
    std::string structure_superpose_transform, structure_superpose_metrics;
    int structure_superpose_chain1 = 0, structure_superpose_chain2 = 0;
    structure_superpose_cmd->add_positional("mobile", structure_superpose_mobile, "Mobile structure")->check(ExistingFile());
    structure_superpose_cmd->add_positional("reference", structure_superpose_ref, "Reference structure")->check(ExistingFile());
    structure_superpose_cmd->add_option("--output", structure_superpose_output, "Output superposed PDB")->required(true);
    structure_superpose_cmd->add_option("--chain1", structure_superpose_chain1, "Chain index for mobile (default: 0)");
    structure_superpose_cmd->add_option("--chain2", structure_superpose_chain2, "Chain index for reference (default: 0)");
    structure_superpose_cmd->add_option("--transform", structure_superpose_transform, "Optional: save transformation to JSON");
    structure_superpose_cmd->add_option("--metrics", structure_superpose_metrics, "Optional: save metrics to file");

    // Structure: Kabsch subcommand
    App* structure_kabsch_cmd = structure_cmd->add_subcommand("kabsch", "Compute Kabsch transformation matrix");
    std::string structure_kabsch_struct1, structure_kabsch_struct2, structure_kabsch_output;
    int structure_kabsch_chain1 = 0, structure_kabsch_chain2 = 0;
    bool structure_kabsch_print_rmsd = false;
    structure_kabsch_cmd->add_positional("struct1", structure_kabsch_struct1, "First structure")->check(ExistingFile());
    structure_kabsch_cmd->add_positional("struct2", structure_kabsch_struct2, "Second structure")->check(ExistingFile());
    structure_kabsch_cmd->add_option("--output", structure_kabsch_output, "Output JSON file")->required(true);
    structure_kabsch_cmd->add_option("--chain1", structure_kabsch_chain1, "Chain index for first structure (default: 0)");
    structure_kabsch_cmd->add_option("--chain2", structure_kabsch_chain2, "Chain index for second structure (default: 0)");
    structure_kabsch_cmd->add_flag("--print-rmsd", structure_kabsch_print_rmsd, "Print RMSD to stdout");

    // ========== BATCH Command ==========
    App* batch_cmd = app.add_subcommand("batch", "Batch processing commands");
    batch_cmd->require_subcommand(true);

    // Batch: Encode subcommand
    App* batch_encode_cmd = batch_cmd->add_subcommand("encode", "Batch encode multiple structures");
    std::vector<std::string> batch_encode_inputs;
    std::string batch_encode_input_list, batch_encode_input_dir, batch_encode_output_dir;
    int batch_encode_k_neighbors = 30;
    int batch_encode_chain = 0;
    auto* batch_encode_positional = batch_encode_cmd->add_positional("inputs", batch_encode_inputs, "Input paths (optional)");
    batch_encode_positional->required(false)->consume_remaining();
    batch_encode_cmd->add_option("--input-list", batch_encode_input_list, "File containing list of inputs")->check(ExistingFile());
    batch_encode_cmd->add_option("--input-dir", batch_encode_input_dir, "Directory containing inputs");
    batch_encode_cmd->add_option("--output-dir", batch_encode_output_dir, "Output directory for embeddings")->required(true);
    batch_encode_cmd->add_option("--k-neighbors", batch_encode_k_neighbors, "Number of nearest neighbors for MPNN (default: 30)")->check(Range(1, 100));
    batch_encode_cmd->add_option("--chain", batch_encode_chain, "Chain index to encode (default: 0)")->check(Range(0, 100));

    // ========== REFORMAT Command ==========
    App* reformat_cmd = app.add_subcommand("reformat", "Convert between alignment formats (FASTA, A2M, A3M, Stockholm, PSI-BLAST, Clustal)");
    std::string reformat_input, reformat_output;
    std::string reformat_input_format = "";
    std::string reformat_output_format = "";
    std::string reformat_match_mode = "";
    int reformat_gap_threshold = 50;
    bool reformat_remove_inserts = false;
    int reformat_remove_gapped = 0;
    bool reformat_uppercase = false;
    bool reformat_lowercase = false;
    bool reformat_remove_ss = false;
    reformat_cmd->add_positional("input", reformat_input, "Input alignment file")->check(ExistingFile());
    reformat_cmd->add_option("--output", reformat_output, "Output alignment file")->required(true);
    reformat_cmd->add_option("--input-format", reformat_input_format, "Input format (auto-detect if not specified). Options: fas, a2m, a3m, sto, psi, clu");
    reformat_cmd->add_option("--output-format", reformat_output_format, "Output format (auto-detect if not specified). Options: fas, a2m, a3m, sto, psi, clu");
    reformat_cmd->add_option("--match-mode", reformat_match_mode, "Match state assignment: 'first' (use first sequence), 'gap' (use gap threshold)");
    reformat_cmd->add_option("--gap-threshold", reformat_gap_threshold, "Gap percentage threshold for match states (0-100, default: 50)")->check(Range(0, 100));
    reformat_cmd->add_flag("--remove-inserts", reformat_remove_inserts, "Remove insert states (lowercase residues)");
    reformat_cmd->add_option("--remove-gapped", reformat_remove_gapped, "Remove columns with >=N% gaps (0-100, 0=disabled, default: 0)")->check(Range(0, 100));
    reformat_cmd->add_flag("--uppercase", reformat_uppercase, "Convert all residues to uppercase");
    reformat_cmd->add_flag("--lowercase", reformat_lowercase, "Convert all residues to lowercase");
    reformat_cmd->add_flag("--remove-secondary-structure", reformat_remove_ss, "Remove secondary structure annotations");

    // ========== INFO Command ==========
    App* info_cmd = app.add_subcommand("info", "Inspect structure files or directories");
    std::string info_path;
    bool info_chains = false;
    info_cmd->add_positional("path", info_path, "Structure file or directory path");
    info_cmd->add_flag("--chains", info_chains, "Show detailed chain information");

    // ========== VERSION Command ==========
    App* version_cmd = app.add_subcommand("version", "Show detailed version information");

    // ========== ALIGNMENT Command (low-level primitives) ==========
    App* alignment_cmd = app.add_subcommand("alignment", "Low-level alignment primitives for research");
    alignment_cmd->require_subcommand(true);

    // Alignment: Forward subcommand
    App* alignment_forward_cmd = alignment_cmd->add_subcommand("forward", "Smith-Waterman forward pass only");
    std::string alignment_forward_similarity, alignment_forward_output;
    float alignment_forward_gap_open = -2.544f;
    float alignment_forward_gap_extend = 0.194f;
    float alignment_forward_temperature = 1.0f;
    alignment_forward_cmd->add_positional("similarity", alignment_forward_similarity, "Input similarity matrix (.npy)")->check(ExistingFile());
    alignment_forward_cmd->add_option("--output", alignment_forward_output, "Output forward scores (.npy)")->required(true);
    alignment_forward_cmd->add_option("--gap-open", alignment_forward_gap_open, "Gap opening penalty (default: -2.544)");
    alignment_forward_cmd->add_option("--gap-extend", alignment_forward_gap_extend, "Gap extension penalty (default: 0.194)");
    alignment_forward_cmd->add_option("--temperature", alignment_forward_temperature, "Softmax temperature (default: 1.0)")->check(Range(0.1, 10.0));

    // Alignment: Backward subcommand
    App* alignment_backward_cmd = alignment_cmd->add_subcommand("backward", "Smith-Waterman backward pass only");
    std::string alignment_backward_forward, alignment_backward_similarity, alignment_backward_output;
    float alignment_backward_partition = 0.0f;
    float alignment_backward_gap_open = -2.544f;
    float alignment_backward_gap_extend = 0.194f;
    float alignment_backward_temperature = 1.0f;
    alignment_backward_cmd->add_positional("forward_scores", alignment_backward_forward, "Input forward scores (.npy from forward command)")->check(ExistingFile());
    alignment_backward_cmd->add_positional("similarity", alignment_backward_similarity, "Original similarity matrix (.npy)")->check(ExistingFile());
    alignment_backward_cmd->add_option("--partition", alignment_backward_partition, "Partition function (from forward pass)")->required(true);
    alignment_backward_cmd->add_option("--output", alignment_backward_output, "Output posterior matrix (.npy)")->required(true);
    alignment_backward_cmd->add_option("--gap-open", alignment_backward_gap_open, "Gap opening penalty (must match forward pass)");
    alignment_backward_cmd->add_option("--gap-extend", alignment_backward_gap_extend, "Gap extension penalty (must match forward pass)");
    alignment_backward_cmd->add_option("--temperature", alignment_backward_temperature, "Softmax temperature (must match forward pass)")->check(Range(0.1, 10.0));

    // Alignment: Decode subcommand
    App* alignment_decode_cmd = alignment_cmd->add_subcommand("decode", "Viterbi decode alignment from posterior");
    std::string alignment_decode_posterior, alignment_decode_seq1, alignment_decode_seq2, alignment_decode_output;
    float alignment_decode_gap_penalty = -5.0f;
    alignment_decode_cmd->add_positional("posterior", alignment_decode_posterior, "Input posterior matrix (.npy)")->check(ExistingFile());
    alignment_decode_cmd->add_positional("seq1", alignment_decode_seq1, "First sequence string or file");
    alignment_decode_cmd->add_positional("seq2", alignment_decode_seq2, "Second sequence string or file");
    alignment_decode_cmd->add_option("--output", alignment_decode_output, "Output FASTA alignment file")->required(true);
    alignment_decode_cmd->add_option("--gap-penalty", alignment_decode_gap_penalty, "Gap penalty for Viterbi decoding (default: -5.0)");

    // ========== TREE Command ==========
    App* tree_cmd = app.add_subcommand("tree", "Build phylogenetic guide trees");
    tree_cmd->require_subcommand(true);

    // Tree: Build subcommand (generic with method selection)
    App* tree_build_cmd = tree_cmd->add_subcommand("build", "Build tree with specified method");
    std::string tree_build_distances, tree_build_output;
    std::string tree_build_method = "upgma";
    std::vector<std::string> tree_build_labels;
    tree_build_cmd->add_positional("distances", tree_build_distances, "Distance matrix (.npy)")->check(ExistingFile());
    tree_build_cmd->add_option("--output", tree_build_output, "Output Newick file")->required(true);
    tree_build_cmd->add_option("--method", tree_build_method, "Tree building method (upgma|nj|bionj|mst, default: upgma)");
    tree_build_cmd->add_option("--labels", tree_build_labels, "Sequence labels (default: seq0, seq1, ...)");

    // Tree: UPGMA subcommand
    App* tree_upgma_cmd = tree_cmd->add_subcommand("upgma", "Build tree using UPGMA");
    std::string tree_upgma_distances, tree_upgma_output;
    std::vector<std::string> tree_upgma_labels;
    tree_upgma_cmd->add_positional("distances", tree_upgma_distances, "Distance matrix (.npy)")->check(ExistingFile());
    tree_upgma_cmd->add_option("--output", tree_upgma_output, "Output Newick file")->required(true);
    tree_upgma_cmd->add_option("--labels", tree_upgma_labels, "Sequence labels (default: seq0, seq1, ...)");

    // Tree: NJ subcommand
    App* tree_nj_cmd = tree_cmd->add_subcommand("nj", "Build tree using Neighbor-Joining");
    std::string tree_nj_distances, tree_nj_output;
    std::vector<std::string> tree_nj_labels;
    tree_nj_cmd->add_positional("distances", tree_nj_distances, "Distance matrix (.npy)")->check(ExistingFile());
    tree_nj_cmd->add_option("--output", tree_nj_output, "Output Newick file")->required(true);
    tree_nj_cmd->add_option("--labels", tree_nj_labels, "Sequence labels (default: seq0, seq1, ...)");

    // Tree: BioNJ subcommand
    App* tree_bionj_cmd = tree_cmd->add_subcommand("bionj", "Build tree using BioNJ");
    std::string tree_bionj_distances, tree_bionj_output;
    std::vector<std::string> tree_bionj_labels;
    tree_bionj_cmd->add_positional("distances", tree_bionj_distances, "Distance matrix (.npy)")->check(ExistingFile());
    tree_bionj_cmd->add_option("--output", tree_bionj_output, "Output Newick file")->required(true);
    tree_bionj_cmd->add_option("--labels", tree_bionj_labels, "Sequence labels (default: seq0, seq1, ...)");

    // Tree: MST subcommand
    App* tree_mst_cmd = tree_cmd->add_subcommand("mst", "Build tree using Minimum Spanning Tree");
    std::string tree_mst_distances, tree_mst_output;
    std::vector<std::string> tree_mst_labels;
    tree_mst_cmd->add_positional("distances", tree_mst_distances, "Distance matrix (.npy)")->check(ExistingFile());
    tree_mst_cmd->add_option("--output", tree_mst_output, "Output Newick file")->required(true);
    tree_mst_cmd->add_option("--labels", tree_mst_labels, "Sequence labels (default: seq0, seq1, ...)");

    // ========== OLD Metrics Subcommands (TO BE REMOVED) ==========
    App* rmsd_cmd = app.add_subcommand("compute-rmsd", "Compute RMSD between structures");
    std::string rmsd_struct1, rmsd_struct2;
    int rmsd_chain1 = 0, rmsd_chain2 = 0;
    bool rmsd_aligned = false;
    rmsd_cmd->add_positional("struct1", rmsd_struct1, "First structure")->check(ExistingFile());
    rmsd_cmd->add_positional("struct2", rmsd_struct2, "Second structure")->check(ExistingFile());
    rmsd_cmd->add_option("--chain1", rmsd_chain1, "Chain index for first structure (default: 0)");
    rmsd_cmd->add_option("--chain2", rmsd_chain2, "Chain index for second structure (default: 0)");
    rmsd_cmd->add_flag("--aligned", rmsd_aligned, "Assume structures are already superposed");

    App* tm_cmd = app.add_subcommand("compute-tm-score", "Compute TM-score between structures");
    std::string tm_struct1, tm_struct2;
    int tm_chain1 = 0, tm_chain2 = 0;
    bool tm_aligned = false;
    tm_cmd->add_positional("struct1", tm_struct1, "First structure")->check(ExistingFile());
    tm_cmd->add_positional("struct2", tm_struct2, "Second structure")->check(ExistingFile());
    tm_cmd->add_option("--chain1", tm_chain1, "Chain index for first structure (default: 0)");
    tm_cmd->add_option("--chain2", tm_chain2, "Chain index for second structure (default: 0)");
    tm_cmd->add_flag("--aligned", tm_aligned, "Assume structures are already superposed");

    App* gdt_cmd = app.add_subcommand("compute-gdt", "Compute GDT-TS and GDT-HA scores");
    std::string gdt_struct1, gdt_struct2;
    int gdt_chain1 = 0, gdt_chain2 = 0;
    bool gdt_aligned = false;
    gdt_cmd->add_positional("struct1", gdt_struct1, "First structure")->check(ExistingFile());
    gdt_cmd->add_positional("struct2", gdt_struct2, "Second structure")->check(ExistingFile());
    gdt_cmd->add_option("--chain1", gdt_chain1, "Chain index for first structure (default: 0)");
    gdt_cmd->add_option("--chain2", gdt_chain2, "Chain index for second structure (default: 0)");
    gdt_cmd->add_flag("--aligned", gdt_aligned, "Assume structures are already superposed");

    App* identity_cmd = app.add_subcommand("compute-identity", "Compute sequence identity from alignment");
    std::string identity_path;
    bool identity_ignore_gaps = true;
    identity_cmd->add_positional("alignment", identity_path, "Alignment file (FASTA)")->check(ExistingFile());
    identity_cmd->add_flag("--ignore-gaps", identity_ignore_gaps, "Ignore gaps in identity calculation (default: true)");

    App* ecs_cmd = app.add_subcommand("compute-ecs", "Compute ECS score from MSA");
    std::string ecs_path;
    float ecs_temperature = 5.0f;
    ecs_cmd->add_positional("msa", ecs_path, "MSA file (FASTA)")->check(ExistingFile());
    ecs_cmd->add_option("--temperature", ecs_temperature, "ECS temperature (default: 5.0)");

    // ========== Superposition Subcommands ==========
    App* kabsch_cmd = app.add_subcommand("kabsch", "Compute Kabsch transformation matrix");
    std::string kabsch_struct1, kabsch_struct2, kabsch_output;
    int kabsch_chain1 = 0, kabsch_chain2 = 0;
    bool kabsch_print_rmsd = false;
    kabsch_cmd->add_positional("struct1", kabsch_struct1, "First structure")->check(ExistingFile());
    kabsch_cmd->add_positional("struct2", kabsch_struct2, "Second structure")->check(ExistingFile());
    kabsch_cmd->add_option("--output", kabsch_output, "Output JSON file")->required(true);
    kabsch_cmd->add_option("--chain1", kabsch_chain1, "Chain index for first structure (default: 0)");
    kabsch_cmd->add_option("--chain2", kabsch_chain2, "Chain index for second structure (default: 0)");
    kabsch_cmd->add_flag("--print-rmsd", kabsch_print_rmsd, "Print RMSD to stdout");

    App* superpose_cmd = app.add_subcommand("superpose", "Superpose two structures");
    std::string superpose_mobile, superpose_ref, superpose_output, superpose_transform, superpose_metrics;
    int superpose_chain1 = 0, superpose_chain2 = 0;
    superpose_cmd->add_positional("mobile", superpose_mobile, "Mobile structure")->check(ExistingFile());
    superpose_cmd->add_positional("reference", superpose_ref, "Reference structure")->check(ExistingFile());
    superpose_cmd->add_option("--output", superpose_output, "Output superposed PDB")->required(true);
    superpose_cmd->add_option("--chain1", superpose_chain1, "Chain index for mobile (default: 0)");
    superpose_cmd->add_option("--chain2", superpose_chain2, "Chain index for reference (default: 0)");
    superpose_cmd->add_option("--transform", superpose_transform, "Optional: save transformation to JSON");
    superpose_cmd->add_option("--metrics", superpose_metrics, "Optional: save metrics to file");

    // Require a subcommand
    app.require_subcommand(true);

    // ========== Parse Arguments ==========
    PFALIGN_PARSE(app, argc, argv);

    // ========== Dispatch to Subcommand ==========
    if (app.get_active_subcommand() == encode_cmd) {
        return pfalign::commands::encode(encode_pdb, encode_output, encode_k_neighbors,
                                         encode_chain);
    } else if (app.get_active_subcommand() == encode_batch_cmd) {
        // Parse inputs using same logic as MSA command
        std::vector<std::string> all_paths;
        if (!encode_batch_inputs.empty()) {
            all_paths = encode_batch_inputs;
        }
        // Note: input_list and input_dir parsing should be handled in the command function
        // For now, pass the parameters directly
        if (!encode_batch_input_list.empty() || !encode_batch_input_dir.empty()) {
            std::cerr << "Error: --input-list and --input-dir not yet supported for encode-batch\n";
            std::cerr << "       Please use positional arguments for now\n";
            return 1;
        }
        return pfalign::commands::encode_batch(all_paths, encode_batch_output_dir,
                                               encode_batch_k_neighbors, encode_batch_chain);
    } else if (app.get_active_subcommand() == sim_cmd) {
        return pfalign::commands::similarity(sim_emb1, sim_emb2, sim_output);
    } else if (app.get_active_subcommand() == distances_cmd) {
        // Parse inputs using same logic as MSA command
        std::vector<std::string> all_paths;
        if (!distances_inputs.empty()) {
            all_paths = distances_inputs;
        }
        // Note: input_list and input_dir parsing should be handled in the command function
        // For now, pass the parameters directly
        if (!distances_input_list.empty() || !distances_input_dir.empty()) {
            std::cerr << "Error: --input-list and --input-dir not yet supported for compute-distances\n";
            std::cerr << "       Please use positional arguments for now\n";
            return 1;
        }
        return pfalign::commands::compute_distances(all_paths, distances_output,
                                                    distances_gap_open, distances_gap_extend,
                                                    distances_temperature);
    } else if (app.get_active_subcommand() == pairwise_cmd) {
        if (pair_fasta.empty() && pair_posterior.empty() && pair_superpose.empty() &&
            pair_metrics.empty()) {
            std::cerr << "Error: Provide --output (FASTA), --posterior (.npy), --superpose "
                         "(.pdb), or --metrics for pairwise results.\n";
            return 1;
        }
        return pfalign::commands::pairwise(pair_input1, pair_input2, pair_posterior, pair_fasta,
                                           pair_superpose, pair_metrics, pair_gap_open,
                                           pair_gap_extend, pair_temperature, pair_k_neighbors,
                                           pair_chain1, pair_chain2, pair_disable_parallel_mpnn);
    } else if (app.get_active_subcommand() == msa_cmd) {
        return pfalign::commands::msa(msa_inputs, msa_input_list, msa_input_dir, msa_output,
                                      msa_superpose, msa_metrics, msa_method, msa_gap_open,
                                      msa_gap_extend, msa_temperature, msa_ecs_temperature,
                                      msa_k_neighbors, msa_arena_size_mb, msa_threads);

    // ========== METRICS Subcommands ==========
    } else if (app.get_active_subcommand() == metrics_cmd) {
        if (metrics_cmd->get_active_subcommand() == metrics_rmsd_cmd) {
            return pfalign::commands::compute_rmsd(metrics_rmsd_struct1, metrics_rmsd_struct2,
                                                   metrics_rmsd_chain1, metrics_rmsd_chain2,
                                                   metrics_rmsd_aligned);
        } else if (metrics_cmd->get_active_subcommand() == metrics_tm_cmd) {
            return pfalign::commands::compute_tm_score(metrics_tm_struct1, metrics_tm_struct2,
                                                       metrics_tm_chain1, metrics_tm_chain2,
                                                       metrics_tm_aligned);
        } else if (metrics_cmd->get_active_subcommand() == metrics_gdt_cmd) {
            return pfalign::commands::compute_gdt(metrics_gdt_struct1, metrics_gdt_struct2,
                                                  metrics_gdt_chain1, metrics_gdt_chain2,
                                                  metrics_gdt_aligned);
        } else if (metrics_cmd->get_active_subcommand() == metrics_identity_cmd) {
            return pfalign::commands::compute_identity(metrics_identity_path,
                                                       metrics_identity_ignore_gaps);
        } else if (metrics_cmd->get_active_subcommand() == metrics_ecs_cmd) {
            return pfalign::commands::compute_ecs(metrics_ecs_path, metrics_ecs_temperature);
        }

    // ========== STRUCTURE Subcommands ==========
    } else if (app.get_active_subcommand() == structure_cmd) {
        if (structure_cmd->get_active_subcommand() == structure_superpose_cmd) {
            return pfalign::commands::superpose(structure_superpose_mobile,
                                               structure_superpose_ref,
                                               structure_superpose_output,
                                               structure_superpose_chain1,
                                               structure_superpose_chain2,
                                               structure_superpose_transform,
                                               structure_superpose_metrics);
        } else if (structure_cmd->get_active_subcommand() == structure_kabsch_cmd) {
            return pfalign::commands::kabsch(structure_kabsch_struct1,
                                            structure_kabsch_struct2,
                                            structure_kabsch_output,
                                            structure_kabsch_chain1,
                                            structure_kabsch_chain2,
                                            structure_kabsch_print_rmsd);
        }

    // ========== BATCH Subcommands ==========
    } else if (app.get_active_subcommand() == batch_cmd) {
        if (batch_cmd->get_active_subcommand() == batch_encode_cmd) {
            // Use same logic as encode-batch
            std::vector<std::string> all_paths;
            if (!batch_encode_inputs.empty()) {
                all_paths = batch_encode_inputs;
            }
            if (!batch_encode_input_list.empty() || !batch_encode_input_dir.empty()) {
                std::cerr << "Error: --input-list and --input-dir not yet supported for batch encode\n";
                std::cerr << "       Please use positional arguments for now\n";
                return 1;
            }
            return pfalign::commands::encode_batch(all_paths, batch_encode_output_dir,
                                                  batch_encode_k_neighbors, batch_encode_chain);
        }

    // ========== REFORMAT Command ==========
    } else if (app.get_active_subcommand() == reformat_cmd) {
        return pfalign::commands::reformat(reformat_input, reformat_output,
                                          reformat_input_format, reformat_output_format,
                                          reformat_match_mode, reformat_gap_threshold,
                                          reformat_remove_inserts, reformat_remove_gapped,
                                          reformat_uppercase, reformat_lowercase,
                                          reformat_remove_ss);

    // ========== INFO Command ==========
    } else if (app.get_active_subcommand() == info_cmd) {
        return pfalign::commands::info(info_path, info_chains);

    // ========== VERSION Command ==========
    } else if (app.get_active_subcommand() == version_cmd) {
        return pfalign::commands::version();

    // ========== ALIGNMENT Subcommands ==========
    } else if (app.get_active_subcommand() == alignment_cmd) {
        if (alignment_cmd->get_active_subcommand() == alignment_forward_cmd) {
            return pfalign::commands::alignment_forward(alignment_forward_similarity,
                                                       alignment_forward_output,
                                                       alignment_forward_gap_open,
                                                       alignment_forward_gap_extend,
                                                       alignment_forward_temperature);
        } else if (alignment_cmd->get_active_subcommand() == alignment_backward_cmd) {
            return pfalign::commands::alignment_backward(alignment_backward_forward,
                                                        alignment_backward_similarity,
                                                        alignment_backward_partition,
                                                        alignment_backward_output,
                                                        alignment_backward_gap_open,
                                                        alignment_backward_gap_extend,
                                                        alignment_backward_temperature);
        } else if (alignment_cmd->get_active_subcommand() == alignment_decode_cmd) {
            return pfalign::commands::alignment_decode(alignment_decode_posterior,
                                                      alignment_decode_seq1,
                                                      alignment_decode_seq2,
                                                      alignment_decode_output,
                                                      alignment_decode_gap_penalty);
        }

    // ========== TREE Subcommands ==========
    } else if (app.get_active_subcommand() == tree_cmd) {
        if (tree_cmd->get_active_subcommand() == tree_build_cmd) {
            return pfalign::commands::tree_build(tree_build_distances, tree_build_output,
                                                tree_build_method, tree_build_labels);
        } else if (tree_cmd->get_active_subcommand() == tree_upgma_cmd) {
            return pfalign::commands::tree_build(tree_upgma_distances, tree_upgma_output,
                                                "upgma", tree_upgma_labels);
        } else if (tree_cmd->get_active_subcommand() == tree_nj_cmd) {
            return pfalign::commands::tree_build(tree_nj_distances, tree_nj_output,
                                                "nj", tree_nj_labels);
        } else if (tree_cmd->get_active_subcommand() == tree_bionj_cmd) {
            return pfalign::commands::tree_build(tree_bionj_distances, tree_bionj_output,
                                                "bionj", tree_bionj_labels);
        } else if (tree_cmd->get_active_subcommand() == tree_mst_cmd) {
            return pfalign::commands::tree_build(tree_mst_distances, tree_mst_output,
                                                "mst", tree_mst_labels);
        }

    // ========== OLD Commands (for backwards compatibility) ==========
    } else if (app.get_active_subcommand() == rmsd_cmd) {
        return pfalign::commands::compute_rmsd(rmsd_struct1, rmsd_struct2, rmsd_chain1,
                                               rmsd_chain2, rmsd_aligned);
    } else if (app.get_active_subcommand() == tm_cmd) {
        return pfalign::commands::compute_tm_score(tm_struct1, tm_struct2, tm_chain1,
                                                   tm_chain2, tm_aligned);
    } else if (app.get_active_subcommand() == gdt_cmd) {
        return pfalign::commands::compute_gdt(gdt_struct1, gdt_struct2, gdt_chain1,
                                              gdt_chain2, gdt_aligned);
    } else if (app.get_active_subcommand() == identity_cmd) {
        return pfalign::commands::compute_identity(identity_path, identity_ignore_gaps);
    } else if (app.get_active_subcommand() == ecs_cmd) {
        return pfalign::commands::compute_ecs(ecs_path, ecs_temperature);
    } else if (app.get_active_subcommand() == kabsch_cmd) {
        return pfalign::commands::kabsch(kabsch_struct1, kabsch_struct2, kabsch_output,
                                         kabsch_chain1, kabsch_chain2, kabsch_print_rmsd);
    } else if (app.get_active_subcommand() == superpose_cmd) {
        return pfalign::commands::superpose(superpose_mobile, superpose_ref, superpose_output,
                                            superpose_chain1, superpose_chain2,
                                            superpose_transform, superpose_metrics);
    }

    // Should never reach here
    std::cerr << "Error: No subcommand selected" << std::endl;
    return 1;
}
