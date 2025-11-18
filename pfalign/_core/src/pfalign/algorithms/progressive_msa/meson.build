# Progressive MSA implementation

progressive_msa_lib = static_library('progressive_msa',
  'progressive_msa.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    common_dep,
    types_dep,               # GuideTree
    adapters_dep,            # Alignment adapters
    mpnn_dep,                # SequenceCache
    msa_dep,
    similarity_scalar_dep,
    smith_waterman_scalar_dep,
    alignment_decode_dep,    # Viterbi posterior decoding
    kabsch_dep,              # Kabsch alignment for superposition
  ],
  install: false,
)

progressive_msa_dep = declare_dependency(
  link_with: progressive_msa_lib,
  dependencies: [
    common_dep,
    types_dep,
    adapters_dep,
    mpnn_dep,
    msa_dep,
    similarity_scalar_dep,
    smith_waterman_scalar_dep,
    alignment_decode_dep,
    kabsch_dep,
  ],
  include_directories: [inc_pfalign],
)
