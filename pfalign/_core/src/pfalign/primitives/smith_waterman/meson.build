# Smith-Waterman Alignment primitive

smith_waterman_scalar_lib = static_library('smith_waterman_scalar',
  'smith_waterman_scalar.cpp',
  include_directories: [inc_pfalign],
  dependencies: [dispatch_dep],
  install: false,
)

smith_waterman_scalar_dep = declare_dependency(
  link_with: smith_waterman_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined Smith-Waterman library
smith_waterman_dep = declare_dependency(
  dependencies: [smith_waterman_scalar_dep],
)
