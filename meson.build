# PFalign - Protein Structure Alignment with MPNN Embeddings
# Meson build system configuration
#
# This build system supports:
#   - Standalone CLI binary (pfalign executable)
#   - Python bindings (pfalign package with _align_cpp extension)
#   - C++ test suite
#
# Build modes:
#   1. Python package build: meson-python auto-enables bindings
#   2. CLI-only build: meson setup build-release -Dbindings=disabled
#   3. Development build: meson setup build-debug -Dbindings=enabled -Dcli=true -Dtests=true

project('pfalign', 'cpp',
  version: '0.0.1a0',
  license: 'MIT',
  meson_version: '>=1.2.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'werror=true',
    'buildtype=release',
    'b_ndebug=if-release',
  ]
)

# ============================================================================
# Dependencies
# ============================================================================

# Thread support (for manual multithreading with std::thread)
threads_dep = dependency('threads')

# Python (optional, for bindings)
py = import('python').find_installation(pure: false, required: get_option('bindings'))
pybind11_dep = disabler()
if py.found()
  pybind11_dep = dependency('pybind11', required: true)
endif

# ============================================================================
# Global Include Directory
# ============================================================================

# Expose pfalign/_core/src as include root
# This allows all targets to use: #include "pfalign/primitives/foo.h"
# Which resolves to: pfalign/_core/src/pfalign/primitives/foo.h
inc_pfalign = include_directories('pfalign/_core/src')

# ============================================================================
# Profiling Configuration
# ============================================================================

# Add ENABLE_PROFILING define if profiling is enabled
profiling_args = []
if get_option('profiling')
  profiling_args = ['-DENABLE_PROFILING=1']
  message('Profiling instrumentation enabled')
else
  profiling_args = ['-DENABLE_PROFILING=0']
endif

# Add profiling args to all C++ compilations
add_project_arguments(profiling_args, language: 'cpp')

# ============================================================================
# Subdirectories
# ============================================================================

# Build C++ core library (always required)
subdir('pfalign/_core')

# Install Python source files (if enabled)
if py.found()
  subdir('pfalign')
endif

# Build Python bindings (if enabled)
if py.found()
  subdir('pfalign/_bindings')
endif

# Build CLI binary (if enabled)
if get_option('cli')
  subdir('cli')
endif

# Build tests (if enabled)
if get_option('tests')
  subdir('tests')
endif

# Documentation generation (if tools available)
subdir('docs')
