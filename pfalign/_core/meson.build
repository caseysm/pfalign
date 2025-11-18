# PFalign C++ Core Library
# Main build orchestrator for all C++ components

# ============================================================================
# Layer 0: Dispatch and Common Utilities
# ============================================================================
subdir('src/pfalign/dispatch')
subdir('src/pfalign/common')

# ============================================================================
# Error Handling (Foundation Layer)
# ============================================================================
subdir('src/pfalign/errors')

# ============================================================================
# Layer 1: Types (Pure Data Structures)
# ============================================================================
subdir('src/pfalign/types')

# ============================================================================
# Layer 1: I/O Parsers and Writers
# ============================================================================
subdir('src/pfalign/io')

# ============================================================================
# Layer 2: Primitives (Low-level Operations)
# ============================================================================
subdir('src/pfalign/primitives')

# ============================================================================
# Layer 3: Modules (High-level Components)
# ============================================================================
subdir('src/pfalign/modules')

# ============================================================================
# Layer 3: Adapters (Format Conversion)
# ============================================================================
subdir('src/pfalign/adapters')

# ============================================================================
# Layer 4: Algorithms (High-level Algorithms)
# ============================================================================
subdir('src/pfalign/algorithms')

# ============================================================================
# Result Types (built after modules due to pairwise_scalar dependency)
# ============================================================================
# Result types library requires pairwise_scalar, so build it here
result_types_lib = static_library('result_types',
  'src/pfalign/common/result_types.cpp',
  include_directories: [inc_pfalign],
  dependencies: [common_dep, pairwise_scalar_dep],
  install: false,
)

result_types_dep = declare_dependency(
  link_with: result_types_lib,
  include_directories: [inc_pfalign],
)

# ============================================================================
# CLI Library (for standalone binary)
# ============================================================================
subdir('src/pfalign/cli')

# ============================================================================
# Tools (Weight Loading, etc.)
# ============================================================================
# Note: tools/weights has no library to build (tests moved to tests/)
# subdir('src/pfalign/tools/weights')
