#pragma once

/**
 * Thread utilities for optimal OpenMP configuration.
 *
 * Provides platform-specific thread count detection:
 * - macOS: Detects P-cores only (avoids slow E-cores)
 * - Linux: Uses all available cores
 */

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef __APPLE__
    #include <sys/sysctl.h>
#endif

namespace pfalign {
namespace gemm {

/**
 * Get optimal thread count for GEMM operations.
 *
 * On macOS (Apple Silicon):
 *   - Detects P-core count via sysctl
 *   - E-cores are ~10* slower for compute, so avoid them
 *
 * On Linux (Grace, Graviton, etc.):
 *   - Uses all available cores
 *
 * Returns: Recommended number of threads for OpenMP
 */
inline int get_optimal_thread_count() {
#ifdef _OPENMP

    #ifdef __APPLE__
    // macOS: Detect performance core count only
    int p_cores = 0;
    size_t size = sizeof(p_cores);

    // Query hw.perflevel0.physicalcpu (P-core count)
    if (sysctlbyname("hw.perflevel0.physicalcpu", &p_cores, &size, nullptr, 0) == 0) {
        if (p_cores > 0) {
            return p_cores;  // Use only P-cores
        }
    }

    // Fallback: Assume half are P-cores (conservative estimate)
    int total_cores = omp_get_max_threads();
    return (total_cores > 2) ? (total_cores / 2) : 1;

    #else
    // Linux/Grace: Use all available cores
    return omp_get_max_threads();
    #endif

#else
    // OpenMP not available: single-threaded
    return 1;
#endif
}

/**
 * Set optimal thread count for GEMM operations.
 *
 * Respects OMP_NUM_THREADS environment variable if set,
 * otherwise uses get_optimal_thread_count().
 *
 * Returns: Actual thread count that was set
 */
inline int set_optimal_thread_count() {
#ifdef _OPENMP
    // Check if user explicitly set OMP_NUM_THREADS
    const char* env_threads = std::getenv("OMP_NUM_THREADS");
    int num_threads;

    if (env_threads != nullptr) {
        // User override
        num_threads = std::atoi(env_threads);
        if (num_threads <= 0) {
            num_threads = 1;  // Invalid value, fallback
        }
    } else {
        // Auto-detect optimal count
        num_threads = get_optimal_thread_count();
    }

    omp_set_num_threads(num_threads);
    return num_threads;
#else
    return 1;
#endif
}

}  // namespace gemm
}  // namespace pfalign
