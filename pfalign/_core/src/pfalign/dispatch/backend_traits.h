#pragma once

#include <cstddef>
#include <cstdint>

namespace pfalign {

// ============================================================================
// Backend Tag Types (for compile-time dispatch)
// ============================================================================

struct ScalarBackend {};

// ============================================================================
// Backend Traits Interface
// ============================================================================

/**
 * BackendTraits provides a unified interface for backend operations.
 *
 * The scalar backend specializes this template to provide:
 * - SIMD width (always 1 for scalar backend)
 * - Vector type (float for scalar backend)
 * - Operations (load, store, fmadd, etc.)
 *
 * This allows writing generic algorithms using backend traits:
 *
 * template <typename Backend>
 * void dot_product(const float* a, const float* b, int N) {
 *     using Traits = BackendTraits<Backend>;
 *     auto sum = Traits::zero();
 *     for (int i = 0; i < N; i += Traits::simd_width) {
 *         auto va = Traits::load(a + i);
 *         auto vb = Traits::load(b + i);
 *         sum = Traits::fmadd(va, vb, sum);
 *     }
 *     return Traits::reduce_sum(sum);
 * }
 */
template <typename Backend>
struct BackendTraits;

// Forward declaration for scalar specialization
template <>
struct BackendTraits<ScalarBackend>;

}  // namespace pfalign
