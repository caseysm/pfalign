# Gather operations primitive

gather_scalar_lib = static_library('gather_scalar',
  'gather_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep, common_dep],
  install: false,
)

gather_scalar_dep = declare_dependency(
  link_with: gather_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined gather library (scalar only)
gather_dep = declare_dependency(
  dependencies: [gather_scalar_dep],
)
