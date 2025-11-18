# RBF (Radial Basis Function) kernel primitive

rbf_scalar_lib = static_library('rbf_scalar',
  'rbf_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, common_dep],
  install: false,
)

rbf_scalar_dep = declare_dependency(
  link_with: rbf_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined rbf library (scalar only)
rbf_dep = declare_dependency(
  dependencies: [rbf_scalar_dep],
)
