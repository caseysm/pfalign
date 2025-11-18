/**
 * GEMM Profiling Implementation
 *
 * Provides the thread-local profiler instance with proper linkage.
 */

#include "gemm_profiling.h"

#ifdef ENABLE_GEMM_PROFILING

namespace pfalign {
namespace gemm {

// Thread-local profiler instance (shared across all compilation units)
thread_local GEMMProfiler profiler;

}  // namespace gemm
}  // namespace pfalign

#endif  // ENABLE_GEMM_PROFILING
