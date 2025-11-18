#pragma once

#include <cstdlib>
#include <cstring>

namespace pfalign {
namespace tuning {

/**
 * Tunable parameters for GEMM cache blocking.
 *
 * Can be overridden via:
 * 1. Environment variables (highest priority)
 * 2. CMake cache variables (compile-time)
 * 3. Runtime heuristics
 * 4. Default constants
 *
 * Environment variables:
 *   SOFTALIGN_GEMM_MC=64
 *   SOFTALIGN_GEMM_NC=64
 *   SOFTALIGN_GEMM_KC=64
 *   SOFTALIGN_GEMM_MR=4
 *   SOFTALIGN_GEMM_NR=4
 */
struct GEMMTuning {
    int MC;  // Block size for M dimension
    int NC;  // Block size for N dimension
    int KC;  // Block size for K dimension
    int MR;  // Microkernel M (register block)
    int NR;  // Microkernel N (register block)

    /**
     * Get tuned parameters (considers env vars + defaults).
     */
    static GEMMTuning get() {
        GEMMTuning params;

// Defaults (tuned for 32 KB L1 cache)
#ifndef SOFTALIGN_GEMM_MC
        params.MC = 64;
#else
        params.MC = SOFTALIGN_GEMM_MC;
#endif

#ifndef SOFTALIGN_GEMM_NC
        params.NC = 64;
#else
        params.NC = SOFTALIGN_GEMM_NC;
#endif

#ifndef SOFTALIGN_GEMM_KC
        params.KC = 64;
#else
        params.KC = SOFTALIGN_GEMM_KC;
#endif

#ifndef SOFTALIGN_GEMM_MR
        params.MR = 4;
#else
        params.MR = SOFTALIGN_GEMM_MR;
#endif

#ifndef SOFTALIGN_GEMM_NR
        params.NR = 4;
#else
        params.NR = SOFTALIGN_GEMM_NR;
#endif

        // Override with environment variables
        if (const char* env_mc = std::getenv("SOFTALIGN_GEMM_MC")) {
            params.MC = std::atoi(env_mc);
        }
        if (const char* env_nc = std::getenv("SOFTALIGN_GEMM_NC")) {
            params.NC = std::atoi(env_nc);
        }
        if (const char* env_kc = std::getenv("SOFTALIGN_GEMM_KC")) {
            params.KC = std::atoi(env_kc);
        }
        if (const char* env_mr = std::getenv("SOFTALIGN_GEMM_MR")) {
            params.MR = std::atoi(env_mr);
        }
        if (const char* env_nr = std::getenv("SOFTALIGN_GEMM_NR")) {
            params.NR = std::atoi(env_nr);
        }

        return params;
    }

    /**
     * Get optimal parameters based on problem size (runtime heuristics).
     */
    static GEMMTuning get_for_size(int M, int N, [[maybe_unused]] int K) {
        GEMMTuning params = get();

        // Small matrices: reduce blocking overhead
        if (M < 32 && N < 32) {
            params.MC = 16;
            params.NC = 16;
            params.KC = 32;
        }
        // Large matrices: increase blocking for better cache reuse
        else if (M > 512 && N > 512) {
            params.MC = 128;
            params.NC = 128;
            params.KC = 128;
        }

        return params;
    }
};

/**
 * Tunable parameters for RBF kernel.
 */
struct RBFTuning {
    int num_bins;        // Number of Gaussian bins (default: 16)
    float min_distance;  // Minimum distance in Å (default: 2.0)
    float max_distance;  // Maximum distance in Å (default: 22.0)

    static RBFTuning get() {
        RBFTuning params;

#ifndef SOFTALIGN_RBF_BINS
        params.num_bins = 16;
#else
        params.num_bins = SOFTALIGN_RBF_BINS;
#endif

#ifndef SOFTALIGN_RBF_MIN_DIST
        params.min_distance = 2.0f;
#else
        params.min_distance = SOFTALIGN_RBF_MIN_DIST;
#endif

#ifndef SOFTALIGN_RBF_MAX_DIST
        params.max_distance = 22.0f;
#else
        params.max_distance = SOFTALIGN_RBF_MAX_DIST;
#endif

        // Environment variable overrides
        if (const char* env = std::getenv("SOFTALIGN_RBF_BINS")) {
            params.num_bins = std::atoi(env);
        }
        if (const char* env = std::getenv("SOFTALIGN_RBF_MIN_DIST")) {
            params.min_distance = static_cast<float>(std::atof(env));
        }
        if (const char* env = std::getenv("SOFTALIGN_RBF_MAX_DIST")) {
            params.max_distance = static_cast<float>(std::atof(env));
        }

        return params;
    }
};

/**
 * Tunable parameters for KNN search.
 */
struct KNNTuning {
    int k;                 // Number of neighbors (default: 30)
    bool use_brute_force;  // Force brute force even for large N

    static KNNTuning get() {
        KNNTuning params;

#ifndef SOFTALIGN_KNN_K
        params.k = 30;
#else
        params.k = SOFTALIGN_KNN_K;
#endif

        params.use_brute_force = false;

        if (const char* env = std::getenv("SOFTALIGN_KNN_K")) {
            params.k = std::atoi(env);
        }
        if (const char* env = std::getenv("SOFTALIGN_KNN_BRUTE_FORCE")) {
            params.use_brute_force = (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0);
        }

        return params;
    }
};

}  // namespace tuning
}  // namespace pfalign
