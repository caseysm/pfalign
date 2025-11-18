# Alignment decoding (MAP from posteriors to discrete path)

alignment_decode_scalar_lib = static_library('alignment_decode_scalar',
  'alignment_decode_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep],
  install: false,
)

alignment_decode_scalar_dep = declare_dependency(
  link_with: alignment_decode_scalar_lib,
  dependencies: [dispatch_dep],  # Propagate transitive deps
  include_directories: [inc_pfalign],
)

# Combined alignment_decode library
alignment_decode_dep = declare_dependency(
  dependencies: [alignment_decode_scalar_dep],
)
