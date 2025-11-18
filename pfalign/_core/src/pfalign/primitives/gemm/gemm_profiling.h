/**
 * GEMM Profiling Infrastructure
 *
 * Conditional compilation for size distribution analysis.
 * Enable with: ENABLE_GEMM_PROFILING=ON cmake flag
 *
 * Usage:
 *   1. Build with profiling enabled
 *   2. Run benchmarks
 *   3. Call export_gemm_profile() to save data
 *   4. Analyze with Python script
 */

#pragma once

#include <string>

#ifdef ENABLE_GEMM_PROFILING

    #include <vector>
    #include <chrono>
    #include <fstream>
    #include <iostream>
    #include <algorithm>
    #include <numeric>
    #include <map>

namespace pfalign {
namespace gemm {

/**
 * Single GEMM call record.
 */
struct GEMMCallRecord {
    int M, N, K;
    double time_us;  // Microseconds

    GEMMCallRecord(int m, int n, int k, double t) : M(m), N(n), K(k), time_us(t) {
    }
};

/**
 * Thread-local profiling data accumulator.
 */
struct GEMMProfiler {
    std::vector<GEMMCallRecord> calls;

    void record(int M, int N, int K, double time_us) {
        calls.emplace_back(M, N, K, time_us);
    }

    void clear() {
        calls.clear();
    }
};

// Thread-local profiler instance (defined in gemm_profiling.cpp)
extern thread_local GEMMProfiler profiler;

/**
 * RAII timer for GEMM calls.
 */
class GEMMTimer {
    int M_, N_, K_;
    std::chrono::high_resolution_clock::time_point start_;

public:
    GEMMTimer(int M, int N, int K)
        : M_(M), N_(N), K_(K), start_(std::chrono::high_resolution_clock::now()) {
    }

    ~GEMMTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
        double time_us = duration.count() / 1000.0;
        profiler.record(M_, N_, K_, time_us);
    }
};

/**
 * Export profiling data to CSV file.
 */
inline void export_gemm_profile(const std::string& filename = "gemm_profile.csv") {
    if (profiler.calls.empty()) {
        std::cout << "No GEMM profiling data collected.\n";
        return;
    }

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // CSV header
    out << "M,N,K,time_us\n";

    // Write all records
    for (const auto& call : profiler.calls) {
        out << call.M << "," << call.N << "," << call.K << "," << call.time_us << "\n";
    }

    out.close();
    std::cout << "GEMM profiling data exported to " << filename << " (" << profiler.calls.size()
              << " calls)\n";
}

/**
 * Print summary statistics to stdout.
 */
inline void print_gemm_profile_summary() {
    if (profiler.calls.empty()) {
        std::cout << "No GEMM profiling data collected.\n";
        return;
    }

    // Group by size
    struct SizeKey {
        int M, N, K;
        bool operator<(const SizeKey& other) const {
            if (M != other.M)
                return M < other.M;
            if (N != other.N)
                return N < other.N;
            return K < other.K;
        }
    };

    std::map<SizeKey, std::pair<int, double>> size_stats;  // count, total_time

    for (const auto& call : profiler.calls) {
        SizeKey key{call.M, call.N, call.K};
        size_stats[key].first++;
        size_stats[key].second += call.time_us;
    }

    // Sort by total time
    std::vector<std::tuple<SizeKey, int, double>> sorted;
    for (const auto& [key, stats] : size_stats) {
        sorted.emplace_back(key, stats.first, stats.second);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

    // Calculate total time
    double total_time = 0.0;
    for (const auto& call : profiler.calls) {
        total_time += call.time_us;
    }

    std::cout << "\n==============================================\n";
    std::cout << "  GEMM Profiling Summary\n";
    std::cout << "==============================================\n";
    std::cout << "Total calls: " << profiler.calls.size() << "\n";
    std::cout << "Total time: " << (total_time / 1000.0) << " ms\n\n";

    std::cout << "Top 10 GEMM sizes by time:\n";
    std::cout << "  Size (M*N*K)          Calls    Time (ms)   % of Total\n";
    std::cout << "  ------------------------------------------------------\n";

    int count = 0;
    for (const auto& entry : sorted) {
        if (count >= 10)
            break;
        const SizeKey& key = std::get<0>(entry);
        int num_calls = std::get<1>(entry);
        double time = std::get<2>(entry);
        double percent = (time / total_time) * 100.0;
        printf("  %4d*%4d*%4d     %6d    %8.2f     %5.1f%%\n", key.M, key.N, key.K, num_calls,
               time / 1000.0, percent);
        count++;
    }

    std::cout << "==============================================\n\n";
}

}  // namespace gemm
}  // namespace pfalign

#else

    // No-op macros when profiling disabled
    #define GEMMTimer(M, N, K) \
        do {                   \
        } while (0)
namespace pfalign {
namespace gemm {
inline void export_gemm_profile(const std::string& = "") {
}
inline void print_gemm_profile_summary() {
}
}  // namespace gemm
}  // namespace pfalign

#endif  // ENABLE_GEMM_PROFILING
