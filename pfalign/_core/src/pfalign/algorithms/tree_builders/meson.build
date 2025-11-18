# Tree builder algorithms library

tree_builder_lib = static_library('tree_builder_lib',
  'upgma.cpp',
  'nj.cpp',
  'bionj.cpp',
  'mst.cpp',
  include_directories: [inc_pfalign],
  dependencies: [
    types_dep,  # GuideTree class definition
  ],
  install: false,
)

tree_builder_dep = declare_dependency(
  link_with: tree_builder_lib,
  dependencies: [types_dep],
  include_directories: [inc_pfalign],
)
