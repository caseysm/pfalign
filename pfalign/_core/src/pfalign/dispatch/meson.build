# Dispatch library with device detection and runtime backend selection

dispatch_lib = static_library('dispatch',
  'runtime_dispatch.cpp',
  'execution_policy.cpp',
  include_directories: [inc_pfalign],
  dependencies: [threads_dep],
  install: false,
)

dispatch_dep = declare_dependency(
  link_with: dispatch_lib,
  include_directories: [inc_pfalign],
)
