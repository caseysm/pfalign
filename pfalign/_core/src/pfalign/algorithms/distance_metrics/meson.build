# Distance metrics computation library

distance_metrics_lib = static_library('distance_metrics',
  'distance_matrix.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    mpnn_dep,             # For SequenceCache type
    pairwise_scalar_dep,  # Calls pairwise alignment (Tier 4 → Tier 3, correct downward!)
    similarity_scalar_dep,  # Similarity computation
  ],
  install: false,
)

distance_metrics_dep = declare_dependency(
  link_with: distance_metrics_lib,
  dependencies: [
    mpnn_dep,
    pairwise_scalar_dep,
    similarity_scalar_dep,
  ],
  include_directories: [inc_pfalign],
)
