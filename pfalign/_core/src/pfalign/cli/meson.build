# Minimal CLI library for PFalign

pfalign_cli_lib = static_library('pfalign_cli',
  'validators.cpp',
  'option.cpp',
  'app.cpp',
  'formatter.cpp',
  include_directories: [inc_pfalign],
  install: false,
)

pfalign_cli_dep = declare_dependency(
  link_with: pfalign_cli_lib,
  include_directories: [inc_pfalign],
)

# Note: Most components are header-only (errors.h, types.h)
# All implementation files are now complete!
