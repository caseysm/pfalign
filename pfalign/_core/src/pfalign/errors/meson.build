# Error handling library

errors_sources = files(
  'pfalign_error.cpp',
  'formatters.cpp',
  'validators.cpp',
)

errors_headers = files(
  'error_categories.h',
  'pfalign_error.h',
  'formatters.h',
  'validators.h',
)

errors_lib = static_library(
  'pfalign_errors',
  sources: errors_sources,
  include_directories: inc_pfalign,
  install: false,
)

errors_dep = declare_dependency(
  link_with: errors_lib,
  include_directories: inc_pfalign,
)
