# Alignment module
# Wraps smith_waterman primitive with high-level interface

alignment_scalar_lib = static_library('alignment_scalar',
  'alignment_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    dispatch_dep,
    primitives_dep,
  ],
  install: false,
)

alignment_scalar_dep = declare_dependency(
  link_with: alignment_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined alignment library
alignment_dep = declare_dependency(
  dependencies: [alignment_scalar_dep],
)
