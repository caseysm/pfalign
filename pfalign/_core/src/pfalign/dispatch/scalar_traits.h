#pragma once

#include "backend_traits.h"
#include <cmath>

namespace pfalign {

/**
 * Scalar backend: Portable scalar implementation.
 *
 * SIMD width = 1 (process one element at a time).
 * Portable C++ implementation for any CPU architecture.
 */
template <>
struct BackendTraits<ScalarBackend> {
    static constexpr const char* name = "scalar";
    static constexpr size_t simd_width = 1;
    static constexpr bool is_gpu = false;

    using value_type = float;
    using vec_type = float;

    // Load/store operations (trivial for scalar)
    static inline vec_type load(const float* ptr) {
        return *ptr;
    }

    static inline void store(float* ptr, vec_type val) {
        *ptr = val;
    }

    // Arithmetic operations
    static inline vec_type add(vec_type a, vec_type b) {
        return a + b;
    }

    static inline vec_type sub(vec_type a, vec_type b) {
        return a - b;
    }

    static inline vec_type mul(vec_type a, vec_type b) {
        return a * b;
    }

    static inline vec_type div(vec_type a, vec_type b) {
        return a / b;
    }

    // Fused multiply-add: c + a*b
    static inline vec_type fmadd(vec_type a, vec_type b, vec_type c) {
        return c + a * b;
    }

    // Special values
    static inline vec_type zero() {
        return 0.0f;
    }

    static inline vec_type one() {
        return 1.0f;
    }

    // Math functions
    static inline vec_type sqrt(vec_type x) {
        return std::sqrt(x);
    }

    static inline vec_type exp(vec_type x) {
        return std::exp(x);
    }

    static inline vec_type log(vec_type x) {
        return std::log(x);
    }

    // Comparisons
    static inline vec_type max(vec_type a, vec_type b) {
        return (a > b) ? a : b;
    }

    static inline vec_type min(vec_type a, vec_type b) {
        return (a < b) ? a : b;
    }

    // Reductions (trivial for scalar - already reduced)
    static inline float reduce_sum(vec_type v) {
        return v;
    }

    static inline float reduce_max(vec_type v) {
        return v;
    }

    static inline float reduce_min(vec_type v) {
        return v;
    }

    // Broadcast scalar to vector (trivial for scalar)
    static inline vec_type broadcast(float x) {
        return x;
    }
};

}  // namespace pfalign
