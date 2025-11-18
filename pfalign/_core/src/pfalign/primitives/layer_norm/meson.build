# Layer Normalization primitive

layer_norm_scalar_lib = static_library('layer_norm_scalar',
  'layer_norm_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, common_dep],
  install: false,
)

layer_norm_scalar_dep = declare_dependency(
  link_with: layer_norm_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined layer_norm library (scalar only)
layer_norm_dep = declare_dependency(
  dependencies: [layer_norm_scalar_dep],
)
