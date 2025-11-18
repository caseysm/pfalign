# PFalign Python Bindings
# PyBind11 extension module (_align_cpp)

# Python extension module
py.extension_module('_align_cpp',
  files(
    'pybind_main.cpp',
    '../../cli/commands/input_utils.cpp',  # Shared with CLI
  ),
  include_directories: [
    inc_pfalign,
    include_directories('.'),          # For numpy_utils.h
    include_directories('../../cli'),  # For commands/input_utils.h
  ],
  dependencies: [
    pybind11_dep,
    threads_dep,

    # Error handling
    errors_dep,

    # Result types
    result_types_dep,

    # High-level modules
    pairwise_dep,
    progressive_msa_dep,
    mpnn_dep,
    similarity_dep,
    alignment_dep,
    decode_dep,
    distance_metrics_dep,    # Distance matrix computation
    tree_builder_dep,         # Guide tree builders (UPGMA, NJ, BioNJ, MST)

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
    alignment_formats_dep,
  ],
  install: true,
  subdir: 'pfalign',
  cpp_args: [
    '-DPYBIND11_DETAILED_ERROR_MESSAGES',
    '-fvisibility=hidden',
    '-Wno-gnu-zero-variadic-macro-arguments',  # PyBind11 macros trigger this on GCC
    '-Wno-pedantic',  # GCC 13.3 treats PYBIND11_MODULE as pedantic error
  ],
)

# Note: meson-python automatically packages the entire pfalign/ directory
# It DOES NOT package:
#   - pfalign/_core/src/ (C++ source - build artifacts only)
#   - /tests/ (C++ tests - outside package)
#   - /cli/ (standalone binary - outside package)
