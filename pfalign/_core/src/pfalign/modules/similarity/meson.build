# Similarity Computation Module

similarity_scalar_lib = static_library('similarity_scalar',
  'similarity_scalar.cpp',
  'profile_similarity.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    dispatch_dep,
    primitives_dep,
    msa_dep,
  ],
  install: false,
)

similarity_scalar_dep = declare_dependency(
  link_with: similarity_scalar_lib,
  include_directories: [inc_pfalign],
)

# Combined similarity library
similarity_dep = declare_dependency(
  dependencies: [similarity_scalar_dep],
)
