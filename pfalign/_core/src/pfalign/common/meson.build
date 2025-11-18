# Common utilities library
# Contains both header-only utilities and result_types library

# Header-only common utilities (arena_allocator, span, tuning, etc.)
common_dep = declare_dependency(
  include_directories: [inc_pfalign],
)

# Note: result_types library will be built after pairwise_scalar is available
# It will be created in a separate build pass or linked in final targets
