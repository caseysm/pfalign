#pragma once

#include <string>

namespace pfalign {

/**
 * Backend types for runtime selection (scalar-only on this branch).
 */
enum class BackendType {
    SCALAR  // Portable scalar implementation
};

/**
 * Detect available backends at runtime.
 */
bool has_scalar_backend();  // Always true

/**
 * Select best available backend automatically.
 *
 * Scalar-only branch: always returns SCALAR
 *
 * Can be overridden with environment variable:
 *   SOFTALIGN_BACKEND=scalar
 */
BackendType select_best_backend();

/**
 * Force a specific backend.
 *
 * Throws std::runtime_error if backend not available.
 */
void set_backend(BackendType backend);
BackendType get_current_backend();

/**
 * Backend name strings (for logging).
 */
const char* backend_name(BackendType backend);

/**
 * Print backend info (for debugging).
 */
void print_backend_info();

}  // namespace pfalign
