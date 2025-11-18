# PFalign C++ Test Suite
#
# Test categories:
#   - unit/         - Unit tests for primitives and modules
#   - integration/  - Integration tests and parity tests
#   - benchmarks/   - Performance benchmarks
#   - validation/   - Golden data validation

# Test dependencies
test_deps = [
  common_dep,
  primitives_dep,
  modules_dep,
  adapters_dep,
  dispatch_dep,
  types_dep,
  threads_dep,
  tree_builder_dep,      # For tree algorithm tests
  progressive_msa_dep,   # For progressive MSA tests
]

# Common include directories for all tests
test_inc = [
  inc_pfalign,
  include_directories('.'),
  include_directories('unit'),
]

# Relaxed warning flags for test code
# Tests don't need to be as strict as production code
test_warning_args = [
  '-Wno-unused-variable',
  '-Wno-unused-parameter',
  '-Wno-unused-but-set-variable',
  '-Wno-sign-compare',
  '-Wno-unused-lambda-capture',
  '-Wno-deprecated-declarations',
  '-Wno-unused-function',
  '-Wno-vla-cxx-extension',
  '-Wno-c++20-extensions',
]

# ============================================================================
# Minimal smoke tests (always run)
# ============================================================================

test_minimal = executable(
  'test_minimal',
  'unit/test_minimal.cpp',
  dependencies: test_deps,
  cpp_args: test_warning_args,
  include_directories: test_inc,
  install: false,
)

test('minimal arena allocator', test_minimal,
  suite: 'smoke',
  timeout: 30,
)

test_minimal_sw = executable(
  'test_minimal_sw',
  'unit/test_minimal_sw.cpp',
  dependencies: test_deps,
  cpp_args: test_warning_args,
  include_directories: test_inc,
  install: false,
)

test('minimal smith-waterman', test_minimal_sw,
  suite: 'smoke',
  timeout: 30,
)

# ============================================================================
# Primitives tests
# ============================================================================

# Smith-Waterman tests
primitives_sw_tests = [
  'primitives/smith_waterman/test_smith_waterman.cpp',
  'primitives/smith_waterman/test_temperature_scaling.cpp',
]

# Golden data tests (require pre-generated reference data)
primitives_sw_golden_tests = [
  'primitives/smith_waterman/test_smith_waterman_comprehensive.cpp',
  'primitives/smith_waterman/test_sw_golden.cpp',
]

foreach test_src : primitives_sw_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives', 'smith-waterman'],
    timeout: 60,
  )
endforeach

# Golden tests marked as optional (require data generation)
foreach test_src : primitives_sw_golden_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args + ['-DPFALIGN_SOURCE_ROOT="' + meson.project_source_root() + '"'],
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives', 'smith-waterman', 'golden'],
    timeout: 60,
  )
endforeach

# GEMM tests
primitives_gemm_tests = [
  'primitives/gemm/test_gemm.cpp',
]

foreach test_src : primitives_gemm_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives', 'gemm'],
    timeout: 60,
  )
endforeach

# KNN tests
primitives_knn_tests = [
  'primitives/knn/test_knn.cpp',
]

foreach test_src : primitives_knn_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives', 'knn'],
    timeout: 60,
  )
endforeach

# Structural metrics tests
primitives_structural_tests = [
  'primitives/structural_metrics/test_distance_matrix.cpp',
  'primitives/structural_metrics/test_structural_metrics.cpp',
  'primitives/structural_metrics/test_lddt.cpp',
  'primitives/structural_metrics/test_dali.cpp',
  'primitives/structural_metrics/test_msa_metrics.cpp',
]

foreach test_src : primitives_structural_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives', 'structural-metrics'],
    timeout: 60,
  )
endforeach

# Other primitive tests
other_primitive_tests = [
  'primitives/layer_norm/test_layer_norm.cpp',
  'primitives/reduce/test_reduce.cpp',
  'primitives/gather/test_gather.cpp',
  'primitives/svd3x3/test_svd3x3.cpp',
  'primitives/alignment_decode/test_alignment_decode.cpp',
]

foreach test_src : other_primitive_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Data structure tests
# ============================================================================

data_structure_tests = [
  'data_structures/test_guide_tree.cpp',
  'data_structures/test_sequence_cache.cpp',
  'data_structures/test_alignment_adapters.cpp',
]

foreach test_src : data_structure_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['data-structures'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Common/Infrastructure tests
# ============================================================================

common_tests = [
  'common/test_arena.cpp',
  'common/test_arena_alignment.cpp',
  'common/test_growable_arena.cpp',
  'common/test_memory_budget.cpp',
  'common/test_memory_profiler.cpp',
  'common/test_profiling.cpp',
  'common/test_thread_pool.cpp',  # Re-enabled after fixing race conditions
]

foreach test_src : common_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['common'],
    timeout: 60,
  )
endforeach

# ============================================================================
# I/O tests
# ============================================================================

io_tests = [
  'io/test_fasta_writer.cpp',
  'io/test_parsers.cpp',
  'io/test_sequence_utils.cpp',
]

foreach test_src : io_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['io'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Module tests
# ============================================================================

# MPNN tests
module_mpnn_tests = [
  'modules/mpnn/test_mpnn.cpp',
  'modules/mpnn/test_mpnn_comprehensive.cpp',
]

# MPNN golden tests (require reference data)
module_mpnn_golden_tests = [
  'modules/mpnn/test_mpnn_golden.cpp',
]

foreach test_src : module_mpnn_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'mpnn'],
    timeout: 120,
  )
endforeach

foreach test_src : module_mpnn_golden_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args + ['-DPFALIGN_SOURCE_ROOT="' + meson.project_source_root() + '"'],
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'mpnn', 'golden'],
    timeout: 180,
  )
endforeach

# Similarity tests
module_similarity_tests = [
  'modules/similarity/test_similarity.cpp',
  'modules/similarity/test_profile_similarity.cpp',
]

module_similarity_golden_tests = [
  'modules/similarity/test_similarity_golden.cpp',
]

foreach test_src : module_similarity_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'similarity'],
    timeout: 60,
  )
endforeach

foreach test_src : module_similarity_golden_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args + ['-DPFALIGN_SOURCE_ROOT="' + meson.project_source_root() + '"'],
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'similarity', 'golden'],
    timeout: 60,
  )
endforeach

# Pairwise tests
module_pairwise_tests = [
  'modules/pairwise/test_pairwise.cpp',
  'modules/pairwise/test_pairwise_full.cpp',
  'modules/pairwise/test_pairwise_score.cpp',
  'modules/pairwise/test_dual_score_comparison.cpp',
  'modules/pairwise/test_optional_dual_score.cpp',
  'modules/pairwise/test_sw_modes.cpp',
]

foreach test_src : module_pairwise_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'pairwise'],
    timeout: 120,
  )
endforeach

# MSA/Profile tests
module_msa_tests = [
  'modules/msa/test_profile.cpp',
]

module_msa_golden_tests = [
  'modules/msa/test_profile_golden.cpp',
  'modules/msa/test_profile_merge_golden.cpp',
]

foreach test_src : module_msa_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'msa'],
    timeout: 120,
  )
endforeach

foreach test_src : module_msa_golden_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args + ['-DPFALIGN_SOURCE_ROOT="' + meson.project_source_root() + '"'],
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules', 'msa', 'golden'],
    timeout: 120,
  )
endforeach

# Other module tests
module_other_tests = [
  'modules/alignment/test_alignment.cpp',
  'modules/decode/test_decode.cpp',
]

foreach test_src : module_other_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['modules'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Algorithm tests
# ============================================================================

algorithm_tests = [
  'algorithms/progressive_msa/test_progressive_msa.cpp',
  'algorithms/progressive_msa/test_msa_families.cpp',
  'algorithms/tree_builders/test_tree_algorithms.cpp',
]

foreach test_src : algorithm_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['algorithms'],
    timeout: 120,
  )
endforeach

# Note: algorithms/distance_metrics/test_distance_matrix.cpp is duplicate of primitives version

# ============================================================================
# Bindings tests
# ============================================================================

bindings_tests = [
  'bindings/test_bindings.cpp',
]

foreach test_src : bindings_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['bindings'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Tools tests
# ============================================================================

tools_tests = [
  'tools/weights/test_safetensors_loader.cpp',
]

foreach test_src : tools_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args + ['-DPFALIGN_SOURCE_ROOT="' + meson.project_source_root() + '"'],
    install: false,
  )
  test(test_name, test_exe,
    suite: ['tools'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Additional primitive tests (not yet categorized above)
# ============================================================================

additional_primitive_tests = [
  'primitives/kabsch/test_kabsch.cpp',
  'primitives/rbf/test_rbf.cpp',
]

foreach test_src : additional_primitive_tests
  test_name = test_src.split('/')[-1].replace('.cpp', '')
  test_exe = executable(
    test_name,
    'unit' / test_src,
    dependencies: test_deps,
    cpp_args: test_warning_args,
    include_directories: test_inc,
    install: false,
  )
  test(test_name, test_exe,
    suite: ['primitives'],
    timeout: 60,
  )
endforeach

# ============================================================================
# Test Summary
# ============================================================================
# Total: 51 tests (50 unique, 1 duplicate)
# - Smoke: 2
# - Primitives: 17
# - Data structures: 3
# - Common: 3
# - I/O: 3
# - Modules: 17
# - Algorithms: 3
# - Bindings: 1
# - Tools: 1
# - Golden (optional): 7
#
# Run all tests: meson test -C build-release
# Run core tests: meson test -C build-release --no-suite=golden
# Run specific suite: meson test -C build-release --suite=primitives
# Run specific test: meson test -C build-release test_smith_waterman
