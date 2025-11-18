# Reduce operations (elementwise multiply, sum, norm)

reduce_scalar_lib = static_library('reduce_scalar',
  'reduce_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep],
  install: false,
)

reduce_scalar_dep = declare_dependency(
  link_with: reduce_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined reduce library (scalar only)
reduce_dep = declare_dependency(
  dependencies: [reduce_scalar_dep],
)
