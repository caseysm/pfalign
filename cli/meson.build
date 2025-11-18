# PFalign - Standalone CLI Binary

# Build the pfalign executable
pfalign_exe = executable('pfalign',
  files(
    'main.cpp',
    'commands/input_utils.cpp',
    'commands/encode.cpp',
    'commands/similarity.cpp',
    'commands/pairwise.cpp',
    'commands/msa.cpp',
    'commands/metrics.cpp',
    'commands/superposition.cpp',
    'commands/utils.cpp',
    'commands/align.cpp',
    'commands/tree.cpp',
  ),
  include_directories: [
    inc_pfalign,
    include_directories('.'),  # For commands/input_utils.h
  ],
  dependencies: [
    threads_dep,

    # CLI framework
    pfalign_cli_dep,

    # Error handling
    errors_dep,

    # Result utilities
    result_types_dep,

    # High-level modules (same as Python bindings)
    pairwise_dep,
    progressive_msa_dep,
    distance_metrics_dep,
    mpnn_dep,
    similarity_dep,
    alignment_dep,
    decode_dep,
    tree_builder_dep,

    # Data structures
    types_dep,
    adapters_dep,

    # Modules
    msa_dep,

    # Primitives
    smith_waterman_dep,
    alignment_decode_dep,
    gemm_dep,
    layer_norm_dep,
    gather_dep,
    reduce_dep,
    rbf_dep,
    knn_dep,
    kabsch_dep,
    structural_metrics_dep,
    svd3x3_dep,

    # Infrastructure
    dispatch_dep,
    fasta_writer_dep,
    pdb_writer_dep,
  ],
  install: true,
  install_dir: get_option('bindir'),
)
