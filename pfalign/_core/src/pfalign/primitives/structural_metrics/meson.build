# Structural metrics primitive

structural_metrics_scalar_lib = static_library('structural_metrics_scalar',
  'structural_metrics_scalar.cpp',
  'distance_matrix_scalar.cpp',
  'lddt_scalar.cpp',
  'dali_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, common_dep, kabsch_scalar_dep],
  install: false,
)

structural_metrics_scalar_dep = declare_dependency(
  link_with: structural_metrics_scalar_lib,
  dependencies: [dispatch_dep, common_dep, kabsch_scalar_dep],  # Propagate transitive deps
  include_directories: [inc_pfalign],
)

# Combined library (INTERFACE for future SIMD/CUDA)
structural_metrics_dep = declare_dependency(
  dependencies: [structural_metrics_scalar_dep],
)
