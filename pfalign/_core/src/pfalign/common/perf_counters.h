#pragma once

#include <cstdint>
#include <cstdio>

namespace pfalign {
namespace profiling {

/**
 * ARM Performance Counter Utilities
 *
 * Provides direct access to ARM PMU (Performance Monitoring Unit) for
 * ultra-low-overhead cycle counting.
 *
 * Note: Requires kernel.perf_event_paranoid = -1 for user-space access.
 */

/**
 * Read ARM cycle counter (PMCCNTR_EL0).
 * This is much faster than std::chrono (1 cycle vs ~30 cycles).
 *
 * @return Current cycle count
 */
static inline uint64_t read_cycles() {
    uint64_t val;
    asm volatile("mrs %0, pmccntr_el0" : "=r"(val));
    return val;
}

/**
 * Enable user-space access to ARM cycle counter.
 * Call this once at program startup.
 *
 * Note: This only works if kernel.perf_event_paranoid = -1
 */
static inline void enable_cycle_counter() {
    uint64_t val;

    // Enable user-space access to PMU
    asm volatile("mrs %0, pmuserenr_el0" : "=r"(val));
    val |= 0x1;  // EN bit
    asm volatile("msr pmuserenr_el0, %0" ::"r"(val));

    // Enable and reset cycle counter
    asm volatile("mrs %0, pmcr_el0" : "=r"(val));
    val |= 0x1 | 0x4;  // E (enable) + C (cycle counter reset)
    asm volatile("msr pmcr_el0, %0" ::"r"(val));

    // Enable PMCCNTR_EL0 specifically
    val = (1u << 31);  // Bit 31 = cycle counter enable
    asm volatile("msr pmcntenset_el0, %0" ::"r"(val));
}

/**
 * Memory barrier (for accurate timing).
 * Ensures all memory operations complete before/after timing.
 */
static inline void memory_barrier() {
    asm volatile("dmb sy" ::: "memory");
}

/**
 * Full instruction barrier.
 * Ensures all instructions complete before reading cycle counter.
 */
static inline void instruction_barrier() {
    asm volatile("isb" ::: "memory");
}

/**
 * Precise cycle measurement with barriers.
 * Use this when you need highly accurate cycle counts.
 *
 * @return Current cycle count after all pending operations
 */
static inline uint64_t read_cycles_precise() {
    memory_barrier();
    instruction_barrier();
    uint64_t cycles = read_cycles();
    instruction_barrier();
    return cycles;
}

/**
 * RAII-style cycle counter for micro-benchmarking.
 *
 * Usage:
 *   {
 *       CycleTimer timer("my_kernel");
 *       // ... code to measure ...
 *   } // Automatically prints elapsed cycles
 */
struct CycleTimer {
    const char* name;
    uint64_t start;

    explicit CycleTimer(const char* n) : name(n) {
        memory_barrier();
        start = read_cycles();
    }

    ~CycleTimer() {
        memory_barrier();
        uint64_t elapsed = read_cycles() - start;
        printf("[CYCLES] %s: %lu cycles\n", name, elapsed);
    }
};

/**
 * Estimate CPU frequency by measuring cycles over a known time period.
 * Returns frequency in GHz.
 *
 * Note: This is approximate and assumes constant frequency (no turbo/throttling).
 */
inline double estimate_cpu_frequency_ghz() {
#include <chrono>

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_cycles = read_cycles();

    // Wait approximately 100ms
    while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time)
               .count() < 0.1) {
        // Busy wait
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    uint64_t end_cycles = read_cycles();

    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    return (elapsed_cycles / elapsed_seconds) / 1e9;
}

}  // namespace profiling
}  // namespace pfalign
