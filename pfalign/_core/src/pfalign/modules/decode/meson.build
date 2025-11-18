# Decode module - wraps alignment_decode primitive

decode_scalar_lib = static_library('decode_scalar',
  'decode_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    alignment_decode_dep,
    dispatch_dep,
    common_dep,
  ],
  install: false,
)

decode_scalar_dep = declare_dependency(
  link_with: decode_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined decode library
decode_dep = declare_dependency(
  dependencies: [decode_scalar_dep],
)
