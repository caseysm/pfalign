# Adapters
# Format conversion utilities for different alignment representations.
# Contains:
# - alignment_adapters: Convert between pairwise edge lists and MSA column format

adapters_lib = static_library('adapters',
  'alignment_adapters.cpp',
  include_directories: [inc_pfalign],
  dependencies: [common_dep],
  install: false,
)

adapters_dep = declare_dependency(
  link_with: adapters_lib,
  include_directories: [inc_pfalign],
)
