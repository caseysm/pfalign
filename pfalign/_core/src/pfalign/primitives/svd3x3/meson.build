# 3*3 SVD primitive

svd3x3_scalar_lib = static_library('svd3x3_scalar',
  'svd3x3_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, common_dep],
  install: false,
)

svd3x3_scalar_dep = declare_dependency(
  link_with: svd3x3_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined SVD3x3 library (INTERFACE for future SIMD/CUDA backends)
svd3x3_dep = declare_dependency(
  dependencies: [svd3x3_scalar_dep],
)
