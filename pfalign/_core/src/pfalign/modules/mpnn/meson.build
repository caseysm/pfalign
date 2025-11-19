# MPNN Encoder Module

mpnn_scalar_lib = static_library('mpnn_scalar',
  'mpnn_scalar.cpp',
  'mpnn_cache_adapter.cpp',
  'sequence_cache.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    dispatch_dep,
    common_dep,
    primitives_dep,  # Includes: gemm, rbf, knn, layer_norm, gather
  ],
  cpp_args: get_option('buildtype') != 'debug' ? ['-DNDEBUG'] : [],
  install: false,
)

mpnn_scalar_dep = declare_dependency(
  link_with: mpnn_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined MPNN library (scalar only on this branch)
mpnn_dep = declare_dependency(
  dependencies: [mpnn_scalar_dep],
)
