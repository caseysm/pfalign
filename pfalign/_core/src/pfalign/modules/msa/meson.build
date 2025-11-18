# MSA (Multiple Sequence Alignment) Module
# Consistency-based progressive alignment using MPNN embeddings

msa_lib = static_library('msa',
  'profile.cpp',
  include_directories: [inc_pfalign],
  dependencies: [common_dep],
  install: false,
)

msa_dep = declare_dependency(
  link_with: msa_lib,
  include_directories: [inc_pfalign],
)
