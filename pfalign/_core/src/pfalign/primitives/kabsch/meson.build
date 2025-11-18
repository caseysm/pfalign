# Kabsch algorithm primitive

kabsch_scalar_lib = static_library('kabsch_scalar',
  'kabsch_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [svd3x3_dep, dispatch_dep, common_dep],
  install: false,
)

kabsch_scalar_dep = declare_dependency(
  link_with: kabsch_scalar_lib,
  dependencies: [svd3x3_dep, dispatch_dep, common_dep],  # Propagate transitive deps
  include_directories: [inc_pfalign],
)

# Combined Kabsch library (INTERFACE for future SIMD/CUDA backends)
kabsch_dep = declare_dependency(
  dependencies: [kabsch_scalar_dep],
)
