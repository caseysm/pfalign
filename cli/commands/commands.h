#pragma once

#include <string>
#include <vector>
#include <iostream>

namespace pfalign {
namespace commands {

// Global flags shared across all commands
struct GlobalFlags {
    bool quiet = false;      // Suppress informational output
    bool progress = false;   // Show progress bars
    bool stats = false;      // Show detailed statistics
};

// Helper functions for output
inline void print_info(const std::string& message, bool quiet) {
    if (!quiet) {
        std::cout << message << std::endl;
    }
}

inline void print_success(const std::string& message, bool quiet) {
    if (!quiet) {
        std::cout << "[OK] " << message << std::endl;
    }
}

inline void print_field(const std::string& name, const std::string& value, bool quiet) {
    if (!quiet) {
        std::cout << "  " << name << ": " << value << std::endl;
    }
}

template<typename T>
inline void print_field(const std::string& name, const T& value, bool quiet) {
    if (!quiet) {
        std::cout << "  " << name << ": " << value << std::endl;
    }
}

// Encode PDB to MPNN embeddings
// Usage: pfalign encode <input.pdb> <output.npy> [--k-neighbors K] [--chain C]
int encode(const std::string& pdb_path, const std::string& output_path, int k_neighbors = 30,
           int chain = 0);

// Batch encode multiple structures to embeddings
// Usage: pfalign encode-batch <input-dir|input-list> --output-dir <dir> [--k-neighbors K] [--chain C]
int encode_batch(const std::vector<std::string>& input_paths, const std::string& output_dir,
                 int k_neighbors = 30, int chain = 0);

// Pairwise alignment command (structures or embeddings)
// Usage: pfalign pairwise <input1> <input2> <output.npy> [--fasta path]
int pairwise(const std::string& input1, const std::string& input2,
             const std::string& posterior_path, const std::string& fasta_path,
             const std::string& superpose_path, const std::string& metrics_path, float gap_open,
             float gap_extend, float temperature, int k_neighbors, int chain1, int chain2,
             bool disable_parallel_mpnn);

// Multiple sequence alignment command
int msa(const std::vector<std::string>& inputs, const std::string& input_list,
        const std::string& input_dir, const std::string& output_path,
        const std::string& superpose_path, const std::string& metrics_path,
        const std::string& method, float gap_open, float gap_extend, float temperature,
        float ecs_temperature, int k_neighbors, int arena_size_mb, int thread_count);

// Compute similarity matrix between embeddings
// Usage: pfalign similarity <emb1.npy> <emb2.npy> <output.npy>
int similarity(const std::string& emb1_path, const std::string& emb2_path,
               const std::string& output_path);

// Compute distance matrix between embeddings
// Usage: pfalign compute-distances <input-dir|input-list> --output <distances.npy> [options]
int compute_distances(const std::vector<std::string>& embedding_paths,
                     const std::string& output_path, float gap_open, float gap_extend,
                     float temperature);

// ========== Metrics Commands ==========

// Compute RMSD between two structures
// Usage: pfalign compute-rmsd <struct1.pdb> <struct2.pdb> [--chain1 C1] [--chain2 C2] [--aligned]
int compute_rmsd(const std::string& struct1_path, const std::string& struct2_path, int chain1,
                 int chain2, bool aligned);

// Compute TM-score between two structures
// Usage: pfalign compute-tm-score <struct1.pdb> <struct2.pdb> [--chain1 C1] [--chain2 C2] [--aligned]
int compute_tm_score(const std::string& struct1_path, const std::string& struct2_path,
                     int chain1, int chain2, bool aligned);

// Compute GDT-TS and GDT-HA scores
// Usage: pfalign compute-gdt <struct1.pdb> <struct2.pdb> [--chain1 C1] [--chain2 C2] [--aligned]
int compute_gdt(const std::string& struct1_path, const std::string& struct2_path, int chain1,
                int chain2, bool aligned);

// Compute sequence identity from alignment
// Usage: pfalign compute-identity <alignment.fasta> [--ignore-gaps]
int compute_identity(const std::string& alignment_path, bool ignore_gaps);

// Compute ECS score from MSA
// Usage: pfalign compute-ecs <msa.fasta> [--temperature T]
int compute_ecs(const std::string& msa_path, float temperature);

// ========== Superposition Commands ==========

// Compute Kabsch transformation
// Usage: pfalign kabsch <struct1.pdb> <struct2.pdb> --output <transform.json> [--chain1 C1] [--chain2 C2]
int kabsch(const std::string& struct1_path, const std::string& struct2_path,
           const std::string& output_path, int chain1, int chain2, bool print_rmsd);

// Superpose two structures
// Usage: pfalign superpose <mobile.pdb> <reference.pdb> --output <superposed.pdb> [options]
int superpose(const std::string& mobile_path, const std::string& reference_path,
              const std::string& output_path, int chain1, int chain2,
              const std::string& transform_path, const std::string& metrics_path);

// ========== Alignment Primitives ==========

// Smith-Waterman forward pass
// Usage: pfalign alignment forward <similarity.npy> --output <forward.npy> [options]
int alignment_forward(const std::string& similarity_path, const std::string& output_path,
                     float gap_open, float gap_extend, float temperature);

// Smith-Waterman backward pass
// Usage: pfalign alignment backward <forward.npy> <similarity.npy> --partition P --output <posterior.npy> [options]
int alignment_backward(const std::string& forward_path, const std::string& similarity_path,
                      float partition, const std::string& output_path,
                      float gap_open, float gap_extend, float temperature);

// Viterbi decode alignment from posterior
// Usage: pfalign alignment decode <posterior.npy> <seq1> <seq2> --output <alignment.fasta> [options]
int alignment_decode(const std::string& posterior_path, const std::string& seq1,
                    const std::string& seq2, const std::string& output_path,
                    float gap_penalty);

// ========== Tree Building ==========

// Build phylogenetic tree from distance matrix
// Usage: pfalign tree build <distances.npy> --output <tree.nwk> [--method upgma|nj|bionj|mst] [--labels ...]
int tree_build(const std::string& distances_path, const std::string& output_path,
              const std::string& method, const std::vector<std::string>& labels);

// ========== Utility Commands ==========

// Convert between alignment formats
// Usage: pfalign reformat <input> --output <output> [options]
int reformat(const std::string& input_path, const std::string& output_path,
             const std::string& input_format, const std::string& output_format,
             const std::string& match_mode, int gap_threshold, bool remove_inserts,
             int remove_gapped, bool uppercase, bool lowercase, bool remove_ss);

// Inspect structure files
// Usage: pfalign info <path> [--chains]
int info(const std::string& path, bool show_chains);

// Show version information
// Usage: pfalign version
int version();

}  // namespace commands
}  // namespace pfalign
