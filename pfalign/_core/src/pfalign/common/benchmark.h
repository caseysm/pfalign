#pragma once

#include <chrono>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>

namespace pfalign {
namespace benchmark {

/**
 * Statistics for benchmark results.
 */
struct BenchmarkStats {
    double mean_ms;
    double median_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
    size_t num_iterations;

    /**
     * Calculate speedup relative to baseline.
     */
    double speedup_vs(const BenchmarkStats& baseline) const {
        return baseline.mean_ms / mean_ms;
    }

    /**
     * Print formatted stats.
     */
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << "[" << name << "] ";
        }
        std::cout << std::fixed << std::setprecision(3) << "Mean: " << mean_ms << "ms +/- "
                  << stddev_ms << "ms"
                  << " | Median: " << median_ms << "ms"
                  << " | Range: [" << min_ms << ", " << max_ms << "]ms"
                  << " | N=" << num_iterations << "\n";
    }
};

/**
 * Micro-benchmark runner for establishing NEON baselines.
 *
 * Features:
 * - Warmup iterations to stabilize CPU frequency/cache
 * - Multiple benchmark iterations for statistical significance
 * - Outlier detection and removal
 * - Comparison between scalar and NEON implementations
 *
 * Usage:
 *   Benchmark bench;
 *   auto scalar_stats = bench.run("GEMM Scalar", []() { gemm_scalar(...); });
 *   auto neon_stats = bench.run("GEMM NEON", []() { gemm_neon(...); });
 *   bench.print_comparison(scalar_stats, neon_stats, "GEMM");
 */
class Benchmark {
public:
    /**
     * Create benchmark runner with default parameters.
     * @param warmup_iters Number of warmup iterations (default: 3)
     * @param bench_iters Number of benchmark iterations (default: 10)
     */
    explicit Benchmark(size_t warmup_iters = 3, size_t bench_iters = 10)
        : warmup_iters_(warmup_iters), bench_iters_(bench_iters), remove_outliers_(true) {
    }

    /**
     * Set number of warmup iterations.
     */
    void set_warmup(size_t iters) {
        warmup_iters_ = iters;
    }

    /**
     * Set number of benchmark iterations.
     */
    void set_iterations(size_t iters) {
        bench_iters_ = iters;
    }

    /**
     * Enable/disable outlier removal (default: enabled).
     */
    void set_outlier_removal(bool enable) {
        remove_outliers_ = enable;
    }

    /**
     * Run benchmark on a function.
     * @param name Name of benchmark (for reporting)
     * @param fn Function to benchmark (should be idempotent)
     * @return Statistics of benchmark run
     */
    template <typename Fn>
    BenchmarkStats run(const std::string& name, Fn fn) {
        // Warmup phase
        for (size_t i = 0; i < warmup_iters_; i++) {
            fn();
        }

        // Benchmark phase
        std::vector<double> timings;
        timings.reserve(bench_iters_);

        for (size_t i = 0; i < bench_iters_; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            fn();
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            timings.push_back(elapsed_ms);
        }

        // Remove outliers if enabled
        if (remove_outliers_ && timings.size() >= 5) {
            timings = remove_outliers_iqr(timings);
        }

        // Compute statistics
        return compute_stats(timings);
    }

    /**
     * Run and compare two implementations.
     * @param name_baseline Name of baseline implementation
     * @param fn_baseline Baseline function
     * @param name_optimized Name of optimized implementation
     * @param fn_optimized Optimized function
     */
    template <typename Fn1, typename Fn2>
    void compare(const std::string& name_baseline, Fn1 fn_baseline,
                 const std::string& name_optimized, Fn2 fn_optimized) {
        auto stats_baseline = run(name_baseline, fn_baseline);
        auto stats_optimized = run(name_optimized, fn_optimized);

        print_comparison(stats_baseline, stats_optimized, name_baseline, name_optimized);
    }

    /**
     * Print comparison between two benchmark results.
     */
    static void print_comparison(const BenchmarkStats& baseline, const BenchmarkStats& optimized,
                                 const std::string& baseline_name = "Baseline",
                                 const std::string& optimized_name = "Optimized") {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Benchmark Comparison\n";
        std::cout << "========================================\n";

        baseline.print(baseline_name);
        optimized.print(optimized_name);

        double speedup = optimized.speedup_vs(baseline);
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) << speedup << "* ";

        if (speedup > 1.1) {
            std::cout << "[OK] (faster)";
        } else if (speedup < 0.9) {
            std::cout << "⚠ (slower)";
        } else {
            std::cout << "~ (similar)";
        }
        std::cout << "\n";
        std::cout << "========================================\n";
    }

private:
    size_t warmup_iters_;
    size_t bench_iters_;
    bool remove_outliers_;

    /**
     * Compute statistics from timing samples.
     */
    static BenchmarkStats compute_stats(const std::vector<double>& timings) {
        BenchmarkStats stats;
        stats.num_iterations = timings.size();

        if (timings.empty()) {
            stats.mean_ms = 0.0;
            stats.median_ms = 0.0;
            stats.stddev_ms = 0.0;
            stats.min_ms = 0.0;
            stats.max_ms = 0.0;
            return stats;
        }

        // Min and max
        stats.min_ms = *std::min_element(timings.begin(), timings.end());
        stats.max_ms = *std::max_element(timings.begin(), timings.end());

        // Mean
        double sum = 0.0;
        for (double t : timings) {
            sum += t;
        }
        stats.mean_ms = sum / timings.size();

        // Median
        std::vector<double> sorted_timings = timings;
        std::sort(sorted_timings.begin(), sorted_timings.end());
        size_t mid = sorted_timings.size() / 2;
        if (sorted_timings.size() % 2 == 0) {
            stats.median_ms = (sorted_timings[mid - 1] + sorted_timings[mid]) / 2.0;
        } else {
            stats.median_ms = sorted_timings[mid];
        }

        // Standard deviation
        double variance = 0.0;
        for (double t : timings) {
            double diff = t - stats.mean_ms;
            variance += diff * diff;
        }
        stats.stddev_ms = std::sqrt(variance / timings.size());

        return stats;
    }

    /**
     * Remove outliers using IQR (Interquartile Range) method.
     * Keeps values within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
     */
    static std::vector<double> remove_outliers_iqr(const std::vector<double>& data) {
        if (data.size() < 5) {
            return data;  // Need at least 5 points for meaningful outlier removal
        }

        std::vector<double> sorted = data;
        std::sort(sorted.begin(), sorted.end());

        // Compute Q1, Q3
        size_t n = sorted.size();
        double q1 = sorted[n / 4];
        double q3 = sorted[(3 * n) / 4];
        double iqr = q3 - q1;

        // Define bounds
        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;

        // Filter outliers
        std::vector<double> filtered;
        for (double val : data) {
            if (val >= lower && val <= upper) {
                filtered.push_back(val);
            }
        }

        // If we filtered out too many points, return original
        if (filtered.size() < data.size() / 2) {
            return data;
        }

        return filtered;
    }
};

/**
 * Simple timer for quick one-off measurements.
 */
class SimpleTimer {
public:
    SimpleTimer() : start_(std::chrono::high_resolution_clock::now()) {
    }

    /**
     * Get elapsed time in milliseconds.
     */
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    /**
     * Reset timer.
     */
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace benchmark
}  // namespace pfalign
