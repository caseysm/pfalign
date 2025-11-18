# KNN (K-Nearest Neighbors) search primitive

knn_scalar_lib = static_library('knn_scalar',
  'knn_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, common_dep],
  install: false,
)

knn_scalar_dep = declare_dependency(
  link_with: knn_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined knn library (scalar only)
knn_dep = declare_dependency(
  dependencies: [knn_scalar_dep],
)
