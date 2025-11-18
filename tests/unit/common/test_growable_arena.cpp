/**
 * Comprehensive tests for GrowableArena allocator.
 */

#include "pfalign/common/growable_arena.h"
#include <iostream>
#include <cassert>
#include <cmath>

using pfalign::memory::GrowableArena;

// Helper to check approximate equality for floating point
bool approx_equal(float a, float b, float eps = 0.01f) {
    return std::abs(a - b) < eps;
}

int main() {
    int test_num = 0;

    // Test 1: Basic allocation (single block)
    {
        test_num++;
        std::cout << "Test " << test_num << ": Basic allocation (single block)" << std::endl;

        GrowableArena arena(1, "test_basic");  // 1 MB

        // Allocate some data
        float* buf1 = arena.allocate<float>(1000);
        assert(buf1 != nullptr);

        // Write and read back
        for (int i = 0; i < 1000; i++) {
            buf1[i] = i * 1.5f;
        }
        assert(approx_equal(buf1[500], 750.0f));

        // Check stats
        assert(arena.num_blocks() == 1);
        assert(arena.used() > 0);
        assert(arena.capacity() == 1024 * 1024);

        std::cout << "  ✓ Single block allocation works" << std::endl;
        std::cout << "  Used: " << arena.used() << " / " << arena.capacity() << " bytes" << std::endl;
        std::cout << "  Peak: " << arena.peak() << " bytes" << std::endl;
    }

    // Test 2: Growth behavior (multiple blocks with 1.5* growth)
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Growth behavior (1.5* strategy)" << std::endl;

        GrowableArena arena(1, "test_growth");  // Start with 1 MB

        // Fill first block (allocate ~900 KB to leave room for alignment)
        float* buf1 = arena.allocate<float>(225000);  // 900 KB
        assert(arena.num_blocks() == 1);
        size_t used_block1 = arena.used();

        // Trigger growth by allocating more
        float* buf2 = arena.allocate<float>(100000);  // 400 KB
        assert(arena.num_blocks() == 2);
        assert(buf2 != nullptr);

        // Second block should be ~1.5 MB (1.5* of first 1 MB block)
        size_t total_capacity = arena.capacity();
        size_t expected_capacity = 1024 * 1024 + static_cast<size_t>(1024 * 1024 * 1.5f);
        assert(std::abs(static_cast<int>(total_capacity) - static_cast<int>(expected_capacity)) < 100000);

        std::cout << "  ✓ Arena grew from 1 block to 2 blocks" << std::endl;
        std::cout << "  Total capacity: " << total_capacity / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Blocks: " << arena.num_blocks() << std::endl;
    }

    // Test 3: Reset functionality
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Reset functionality" << std::endl;

        GrowableArena arena(1, "test_reset");

        // Allocate across multiple blocks
        float* buf1 = arena.allocate<float>(250000);  // ~1 MB
        float* buf2 = arena.allocate<float>(250000);  // Triggers growth

        size_t used_before = arena.used();
        size_t blocks_before = arena.num_blocks();
        assert(used_before > 0);
        assert(blocks_before >= 2);

        // Reset
        arena.reset();

        // After reset: usage should be zero, but blocks still allocated
        assert(arena.used() == 0);
        assert(arena.num_blocks() == blocks_before);
        assert(arena.capacity() > 0);

        // Can allocate again
        float* buf3 = arena.allocate<float>(100);
        assert(buf3 != nullptr);
        buf3[0] = 42.0f;
        assert(arena.used() > 0);

        std::cout << "  ✓ Reset cleared usage while keeping blocks" << std::endl;
        std::cout << "  Before reset: " << used_before << " bytes used, " << blocks_before << " blocks" << std::endl;
        std::cout << "  After reset: " << arena.used() << " bytes used, " << arena.num_blocks() << " blocks" << std::endl;
    }

    // Test 4: Peak tracking
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Peak tracking" << std::endl;

        GrowableArena arena(1, "test_peak");

        // Allocate progressively
        float* buf1 = arena.allocate<float>(50000);   // ~200 KB
        size_t peak1 = arena.peak();

        float* buf2 = arena.allocate<float>(100000);  // ~400 KB more
        size_t peak2 = arena.peak();

        assert(peak2 >= peak1);
        assert(peak2 >= arena.used());

        // Reset and allocate less - peak should NOT decrease
        arena.reset();
        float* buf3 = arena.allocate<float>(10000);  // ~40 KB
        size_t peak3 = arena.peak();

        assert(peak3 == peak2);  // Peak persists across reset

        std::cout << "  ✓ Peak tracking works correctly" << std::endl;
        std::cout << "  Peak after buf1: " << peak1 / 1024.0 << " KB" << std::endl;
        std::cout << "  Peak after buf2: " << peak2 / 1024.0 << " KB" << std::endl;
        std::cout << "  Peak after reset+buf3: " << peak3 / 1024.0 << " KB (unchanged)" << std::endl;
    }

    // Test 5: Shrink to fit
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Shrink to fit" << std::endl;

        GrowableArena arena(1, "test_shrink");

        // Grow to multiple blocks
        float* buf1 = arena.allocate<float>(250000);  // ~1 MB
        float* buf2 = arena.allocate<float>(250000);  // ~1 MB, triggers growth
        float* buf3 = arena.allocate<float>(250000);  // Triggers another growth

        size_t blocks_before = arena.num_blocks();
        size_t capacity_before = arena.capacity();
        assert(blocks_before >= 3);

        // Shrink
        arena.shrink_to_fit();

        // After shrink: only first block remains
        assert(arena.num_blocks() == 1);
        assert(arena.capacity() < capacity_before);
        assert(arena.used() == 0);  // shrink_to_fit also resets

        std::cout << "  ✓ Shrink to fit deallocated extra blocks" << std::endl;
        std::cout << "  Before: " << blocks_before << " blocks, " << capacity_before / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  After: " << arena.num_blocks() << " blocks, " << arena.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    // Test 6: Large allocation (exceeds MAX_BLOCK_SIZE)
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Large allocation (exact-fit block)" << std::endl;

        GrowableArena arena(1, "test_large");

        // Allocate 300 MB (exceeds MAX_BLOCK_SIZE of 256 MB)
        size_t huge_count = 75 * 1024 * 1024;  // 300 MB of floats
        float* huge_buf = arena.allocate<float>(huge_count);
        assert(huge_buf != nullptr);

        // Should have created an exact-fit block
        assert(arena.num_blocks() == 2);  // Initial 1 MB + 300 MB exact-fit

        // Verify we can write to it
        huge_buf[0] = 1.0f;
        huge_buf[huge_count - 1] = 2.0f;
        assert(huge_buf[0] == 1.0f);
        assert(huge_buf[huge_count - 1] == 2.0f);

        std::cout << "  ✓ Large allocation created exact-fit block" << std::endl;
        std::cout << "  Allocated: " << (huge_count * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Blocks: " << arena.num_blocks() << std::endl;
    }

    // Test 7: Zero allocation
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Zero allocation (edge case)" << std::endl;

        GrowableArena arena(1, "test_zero");

        float* buf = arena.allocate<float>(0);
        assert(buf == nullptr);
        assert(arena.used() == 0);

        std::cout << "  ✓ Zero allocation returns nullptr" << std::endl;
    }

    // Test 8: Move semantics
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Move semantics" << std::endl;

        GrowableArena arena1(2, "test_move_src");
        float* buf1 = arena1.allocate<float>(100000);
        buf1[0] = 123.45f;

        size_t capacity1 = arena1.capacity();
        size_t used1 = arena1.used();

        // Move construct
        GrowableArena arena2(std::move(arena1));

        // arena2 should have arena1's resources
        assert(arena2.capacity() == capacity1);
        assert(arena2.used() == used1);
        assert(arena2.num_blocks() >= 1);

        // arena1 should be empty
        assert(arena1.capacity() == 0);
        assert(arena1.used() == 0);

        // Move assign
        GrowableArena arena3(1, "test_move_dst");
        arena3 = std::move(arena2);

        // arena3 should have arena2's resources
        assert(arena3.capacity() == capacity1);
        assert(arena3.used() == used1);

        // arena2 should be empty
        assert(arena2.capacity() == 0);
        assert(arena2.used() == 0);

        std::cout << "  ✓ Move semantics work correctly" << std::endl;
    }

    // Test 9: Alignment verification
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Cache-line alignment (64-byte)" << std::endl;

        GrowableArena arena(1, "test_align");

        // Allocate various types
        float* f_ptr = arena.allocate<float>(1);
        double* d_ptr = arena.allocate<double>(1);
        int* i_ptr = arena.allocate<int>(1);

        // All should be 64-byte aligned
        assert(reinterpret_cast<uintptr_t>(f_ptr) % 64 == 0);
        assert(reinterpret_cast<uintptr_t>(d_ptr) % 64 == 0);
        assert(reinterpret_cast<uintptr_t>(i_ptr) % 64 == 0);

        std::cout << "  ✓ All allocations are 64-byte aligned" << std::endl;
    }

    // Test 10: Multiple resets and reuse
    {
        test_num++;
        std::cout << "\nTest " << test_num << ": Multiple resets and reuse" << std::endl;

        GrowableArena arena(1, "test_reuse");

        for (int iteration = 0; iteration < 5; iteration++) {
            // Allocate
            float* buf = arena.allocate<float>(100000);
            buf[0] = iteration * 10.0f;
            assert(arena.used() > 0);

            // Reset for next iteration
            arena.reset();
            assert(arena.used() == 0);
        }

        // Should still have same capacity (no extra blocks)
        assert(arena.num_blocks() == 1);

        std::cout << "  ✓ Multiple reset/reuse cycles work correctly" << std::endl;
        std::cout << "  Blocks after 5 cycles: " << arena.num_blocks() << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ All " << test_num << " tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
