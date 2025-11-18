# Types library - Pure data structures (Layer 1)
# Contains fundamental data types used across multiple layers.
# Zero dependencies on algorithms or modules.
# Only depends on Layer 0 (common, dispatch).

types_lib = static_library('types',
  'guide_tree_types.cpp',
  include_directories: [inc_pfalign],
  dependencies: [common_dep],
  install: false,
)

types_dep = declare_dependency(
  link_with: types_lib,
  include_directories: [inc_pfalign],
)
