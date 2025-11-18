/**
 * Unit tests for MemoryBudget global tracker.
 */

#include "pfalign/common/growable_arena.h"
#include "pfalign/common/memory_budget.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <vector>

using pfalign::memory::MemoryBudget;
using pfalign::memory::GrowableArena;

// Helper to check approximate equality
bool approx_equal(size_t a, size_t b, size_t tolerance) {
    return (a > b) ? (a - b <= tolerance) : (b - a <= tolerance);
}

int main() {
    int test_num = 0;

    // Get global budget instance
    MemoryBudget& budget = MemoryBudget::global();

    // Test 1: System RAM detection
    {
        test_num++;
        std::cout << "Test " << test_num << ": System RAM detection" << std::endl;

        size_t system_ram_mb = budget.system_ram_mb();
        assert(system_ram_mb > 0);
        assert(system_ram_mb >= 1024);  // At least 1 GB (reasonable minimum)

        std::cout << "  ✓ Detected system RAM: " << system_ram_mb << " MB" << std::endl;
    }

    // Test 2: Initial state
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Initial budget state" << std::endl;

        // Note: There might be some allocations from other tests or runtime
        // So we just verify the budget is initialized
        size_t initial_allocated = budget.allocated_mb();
        size_t initial_peak = budget.peak_mb();

        std::cout << "  Initial allocated: " << initial_allocated << " MB" << std::endl;
        std::cout << "  Initial peak: " << initial_peak << " MB" << std::endl;
        std::cout << "  ✓ Budget initialized" << std::endl;
    }

    // Test 3: Allocation tracking with GrowableArena
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Allocation tracking" << std::endl;

        size_t before_allocated = budget.allocated_mb();

        // Create arena (allocates 10 MB initially)
        {
            GrowableArena arena(10, "test_tracking");
            size_t after_create = budget.allocated_mb();

            // Should have increased by ~10 MB (allow 1 MB tolerance for alignment)
            assert(after_create >= before_allocated + 9);
            assert(after_create <= before_allocated + 11);

            std::cout << "  Before arena: " << before_allocated << " MB" << std::endl;
            std::cout << "  After arena: " << after_create << " MB" << std::endl;
        }

        // After arena destruction, allocation should decrease
        size_t after_destroy = budget.allocated_mb();
        assert(approx_equal(after_destroy, before_allocated, 1));

        std::cout << "  After destruction: " << after_destroy << " MB" << std::endl;
        std::cout << "  ✓ Allocation tracking works correctly" << std::endl;
    }

    // Test 4: Peak tracking
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Peak tracking" << std::endl;

        budget.reset_peak();  // Reset peak to current allocation
        size_t initial_peak = budget.peak_mb();

        // Create large arena
        {
            GrowableArena arena(50, "test_peak");
            size_t peak_during = budget.peak_mb();

            // Peak should have increased
            assert(peak_during >= initial_peak + 49);

            std::cout << "  Initial peak: " << initial_peak << " MB" << std::endl;
            std::cout << "  Peak during allocation: " << peak_during << " MB" << std::endl;
        }

        // After destruction, peak should NOT decrease (it's a high-water mark)
        size_t peak_after = budget.peak_mb();
        assert(peak_after >= initial_peak + 49);

        std::cout << "  Peak after destruction: " << peak_after << " MB (unchanged)" << std::endl;
        std::cout << "  ✓ Peak tracking persists across deallocations" << std::endl;
    }

    // Test 5: Multiple arenas tracking
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Multiple arenas tracking" << std::endl;

        budget.reset_peak();
        size_t before = budget.allocated_mb();

        // Create 3 arenas simultaneously
        {
            GrowableArena arena1(5, "test_multi_1");
            GrowableArena arena2(5, "test_multi_2");
            GrowableArena arena3(5, "test_multi_3");

            size_t during = budget.allocated_mb();

            // Should have increased by ~15 MB total
            assert(during >= before + 14);
            assert(during <= before + 16);

            std::cout << "  Before: " << before << " MB" << std::endl;
            std::cout << "  With 3 arenas (5 MB each): " << during << " MB" << std::endl;
        }

        // After all destroyed, should be back to before
        size_t after = budget.allocated_mb();
        assert(approx_equal(after, before, 1));

        std::cout << "  After destruction: " << after << " MB" << std::endl;
        std::cout << "  ✓ Multiple arenas tracked correctly" << std::endl;
    }

    // Test 6: suggest_thread_count() logic
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Thread count suggestions" << std::endl;

        budget.reset_peak();

        // Test various arena sizes
        size_t threads_10mb = budget.suggest_thread_count(10);
        size_t threads_100mb = budget.suggest_thread_count(100);
        size_t threads_1gb = budget.suggest_thread_count(1024);

        // Larger arena sizes should suggest fewer threads
        assert(threads_100mb <= threads_10mb);
        assert(threads_1gb <= threads_100mb);

        // Should never suggest 0 threads
        assert(threads_10mb >= 1);
        assert(threads_100mb >= 1);
        assert(threads_1gb >= 1);

        // Should respect hardware_concurrency
        size_t hw_threads = std::max(1u, std::thread::hardware_concurrency());
        assert(threads_10mb <= hw_threads);

        std::cout << "  HW threads: " << hw_threads << std::endl;
        std::cout << "  Suggested for 10 MB/thread: " << threads_10mb << std::endl;
        std::cout << "  Suggested for 100 MB/thread: " << threads_100mb << std::endl;
        std::cout << "  Suggested for 1 GB/thread: " << threads_1gb << std::endl;
        std::cout << "  ✓ Thread suggestions scale appropriately" << std::endl;
    }

    // Test 7: Available memory calculation
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Available memory calculation" << std::endl;

        size_t system_ram = budget.system_ram_mb();
        size_t allocated = budget.allocated_mb();
        size_t available = budget.available_mb();

        // Available should be system - allocated (approximately)
        assert(approx_equal(available, system_ram - allocated, 10));

        std::cout << "  System RAM: " << system_ram << " MB" << std::endl;
        std::cout << "  Allocated: " << allocated << " MB" << std::endl;
        std::cout << "  Available: " << available << " MB" << std::endl;
        std::cout << "  ✓ Available calculation correct" << std::endl;
    }

    // Test 8: Arena growth tracking
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Arena growth tracking" << std::endl;

        budget.reset_peak();
        size_t before = budget.allocated_mb();

        {
            GrowableArena arena(1, "test_growth_tracking");
            size_t after_initial = budget.allocated_mb();

            // Initial: ~1 MB
            assert(after_initial >= before);

            // Trigger growth by allocating 2 MB (exceeds initial 1 MB block)
            float* large_buf = arena.allocate<float>(500000);  // 2 MB
            (void)large_buf;  // Suppress unused warning

            size_t after_growth = budget.allocated_mb();

            // Should now have ~2.5 MB (1 MB initial + 1.5 MB growth)
            assert(after_growth >= after_initial + 1);

            std::cout << "  Before arena: " << before << " MB" << std::endl;
            std::cout << "  After initial block: " << after_initial << " MB" << std::endl;
            std::cout << "  After growth: " << after_growth << " MB" << std::endl;
        }

        size_t after_destroy = budget.allocated_mb();
        assert(approx_equal(after_destroy, before, 1));

        std::cout << "  After destruction: " << after_destroy << " MB" << std::endl;
        std::cout << "  ✓ Arena growth properly tracked" << std::endl;
    }

    // Test 9: Thread safety (basic check)
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Thread safety (concurrent allocations)" << std::endl;

        budget.reset_peak();
        size_t before = budget.allocated_mb();

        // Create multiple arenas in parallel
        std::vector<std::thread> threads;
        const int num_threads = 4;
        const int arena_size_mb = 5;

        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([arena_size_mb, i]() {
                GrowableArena arena(arena_size_mb, ("test_thread_" + std::to_string(i)).c_str());
                // Allocate some data
                float* buf = arena.allocate<float>(100000);
                buf[0] = static_cast<float>(i);
                // Arena destructs when thread exits
            });
        }

        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }

        // After all threads done, allocation should be back to before
        size_t after = budget.allocated_mb();
        assert(approx_equal(after, before, 1));

        std::cout << "  ✓ Concurrent allocations handled correctly" << std::endl;
        std::cout << "  Peak during concurrent test: " << budget.peak_mb() << " MB" << std::endl;
    }

    // Test 10: Safety factor in suggest_thread_count
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Safety factor parameter" << std::endl;

        // With different safety factors, should get different suggestions
        size_t threads_80pct = budget.suggest_thread_count(100, 0.8f);   // 80% of available
        size_t threads_50pct = budget.suggest_thread_count(100, 0.5f);   // 50% of available
        size_t threads_100pct = budget.suggest_thread_count(100, 1.0f);  // 100% of available

        // More conservative safety factor should suggest fewer or equal threads
        assert(threads_50pct <= threads_80pct);
        assert(threads_80pct <= threads_100pct);

        std::cout << "  Threads with 50% safety: " << threads_50pct << std::endl;
        std::cout << "  Threads with 80% safety: " << threads_80pct << std::endl;
        std::cout << "  Threads with 100% safety: " << threads_100pct << std::endl;
        std::cout << "  ✓ Safety factor respected" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All " << test_num << " tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;

    // Print final budget stats
    std::cout << "\nFinal budget state:" << std::endl;
    budget.print_stats(stdout);

    return 0;
}
