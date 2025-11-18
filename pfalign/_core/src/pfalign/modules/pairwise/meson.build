# Pairwise Alignment Pipeline Module

pairwise_scalar_lib = static_library('pairwise_scalar',
  'pairwise_scalar.cpp',
  'pairwise_output.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    mpnn_scalar_dep,
    similarity_scalar_dep,
    smith_waterman_scalar_dep,
    alignment_scalar_dep,
    reduce_scalar_dep,
    dispatch_dep,
    alignment_decode_dep,      # decode_alignment function
    kabsch_dep,                 # Kabsch alignment for superposition
    structural_metrics_dep,     # TM-score, GDT
    fasta_writer_dep,           # FASTA output
    pdb_writer_dep,             # PDB output
  ],
  install: false,
)

pairwise_scalar_dep = declare_dependency(
  link_with: pairwise_scalar_lib,
  dependencies: [
    mpnn_scalar_dep,
    similarity_scalar_dep,
    smith_waterman_scalar_dep,
    alignment_scalar_dep,
    reduce_scalar_dep,
    dispatch_dep,
    alignment_decode_dep,
    kabsch_dep,
    structural_metrics_dep,
    fasta_writer_dep,
    pdb_writer_dep,
  ],
  include_directories: [inc_pfalign],
)

# Combined pairwise library
pairwise_dep = declare_dependency(
  dependencies: [pairwise_scalar_dep],
)
