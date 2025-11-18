# GEMM (General Matrix Multiply) primitive

gemm_scalar_lib = static_library('gemm_scalar',
  'gemm_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep],
  install: false,
)

gemm_scalar_dep = declare_dependency(
  link_with: gemm_scalar_lib,
  include_directories: [inc_pfalign],
)

# Dispatch wrapper
gemm_dispatch_lib = static_library('gemm_dispatch',
  'gemm_dispatch.cpp',
  'gemm_profiling.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, gemm_scalar_dep],
  install: false,
)

gemm_dispatch_dep = declare_dependency(
  link_with: gemm_dispatch_lib,
  dependencies: [gemm_scalar_dep],  # Transitively include gemm_scalar
  include_directories: [inc_pfalign],
)

# Combined GEMM library (scalar only on this branch)
gemm_dep = declare_dependency(
  dependencies: [gemm_dispatch_dep],
)
