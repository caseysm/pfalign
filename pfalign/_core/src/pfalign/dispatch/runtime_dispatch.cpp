#include "runtime_dispatch.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace pfalign {

// Global backend selection (scalar-only)
static BackendType g_current_backend = BackendType::SCALAR;
static bool g_backend_initialized = false;

// ============================================================================
// Backend Detection
// ============================================================================

bool has_scalar_backend() {
    return true;  // Always available
}

// ============================================================================
// Backend Selection
// ============================================================================

BackendType select_best_backend() {
    // Scalar-only branch: always return SCALAR
    const char* env_backend = std::getenv("SOFTALIGN_BACKEND");
    if (env_backend != nullptr && std::strcmp(env_backend, "scalar") != 0) {
        throw std::runtime_error(
            std::string("Only scalar backend is supported on this branch. Got: ") + env_backend);
    }
    return BackendType::SCALAR;
}

void set_backend(BackendType backend) {
    // Scalar-only branch: only SCALAR is valid
    if (backend != BackendType::SCALAR) {
        throw std::runtime_error("Only SCALAR backend is supported on this branch");
    }
    g_current_backend = backend;
    g_backend_initialized = true;
}

BackendType get_current_backend() {
    if (!g_backend_initialized) {
        g_current_backend = select_best_backend();
        g_backend_initialized = true;
    }
    return g_current_backend;
}

// ============================================================================
// Backend Names
// ============================================================================

const char* backend_name([[maybe_unused]] BackendType backend) {
    return "Scalar";  // Only scalar backend on this branch
}

// ============================================================================
// Debug Info
// ============================================================================

void print_backend_info() {
    std::cout << "=== SoftAlign v2 Backend Info ===" << std::endl;
    std::cout << "Backend: Scalar (portable C++ implementation)" << std::endl;
    std::cout << "SIMD optimizations: Not supported on this branch" << std::endl;

    const char* env_override = std::getenv("SOFTALIGN_BACKEND");
    if (env_override != nullptr) {
        std::cout << "Environment variable: " << env_override << std::endl;
    }

    std::cout << "================================" << std::endl;
}

}  // namespace pfalign
