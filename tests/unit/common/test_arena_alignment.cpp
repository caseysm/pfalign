#include "pfalign/common/arena_allocator.h"
#include <cassert>
#include <iostream>
#include <cstdint>

using namespace pfalign::memory;

void test_simd_alignment() {
    std::cout << "Testing SIMD alignment (16-byte for NEON)...\n";

    Arena arena(64 * 1024);  // 64 KB buffer (Arena expects bytes)

    // Test 1: Single allocation should be aligned
    float* ptr1 = arena.allocate<float>(1);
    uintptr_t addr1 = reinterpret_cast<uintptr_t>(ptr1);
    assert(addr1 % 16 == 0);
    std::cout << "  ✓ Single float allocation: 16-byte aligned\n";

    // Test 2: Odd-sized allocation should still result in next allocation being aligned
    arena.reset();
    float* ptr2 = arena.allocate<float>(13);  // Odd size
    float* ptr3 = arena.allocate<float>(4);   // Next allocation
    uintptr_t addr3 = reinterpret_cast<uintptr_t>(ptr3);
    assert(addr3 % 16 == 0);
    std::cout << "  ✓ After odd-sized allocation: next allocation 16-byte aligned\n";

    // Test 3: Vector-sized allocations (4 floats = 16 bytes)
    arena.reset();
    float* ptr4 = arena.allocate<float>(4);
    uintptr_t addr4 = reinterpret_cast<uintptr_t>(ptr4);
    assert(addr4 % 16 == 0);
    std::cout << "  ✓ Vector-sized allocation (4 floats): 16-byte aligned\n";

    // Test 4: Large allocation
    arena.reset();
    float* ptr5 = arena.allocate<float>(1024);
    uintptr_t addr5 = reinterpret_cast<uintptr_t>(ptr5);
    assert(addr5 % 16 == 0);
    assert(addr5 % 64 == 0);  // Should also be cache-line aligned
    std::cout << "  ✓ Large allocation (1024 floats): 64-byte cache-line aligned\n";

    // Test 5: Multiple consecutive allocations
    arena.reset();
    for (int i = 0; i < 10; i++) {
        float* ptr = arena.allocate<float>(7);  // Prime number to test alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        assert(addr % 16 == 0);
    }
    std::cout << "  ✓ Multiple consecutive odd-sized allocations: all 16-byte aligned\n";

    std::cout << "\n✅ All SIMD alignment tests passed!\n";
}

void test_cache_line_alignment() {
    std::cout << "\nTesting cache-line alignment (64-byte)...\n";

    Arena arena(64 * 1024);  // bytes

    // Arena buffer itself should be 64-byte aligned
    float* ptr1 = arena.allocate<float>(1);
    uintptr_t addr1 = reinterpret_cast<uintptr_t>(ptr1);
    assert(addr1 % 64 == 0);
    std::cout << "  ✓ Arena buffer: 64-byte cache-line aligned\n";

    // All allocations should maintain 64-byte alignment
    arena.reset();
    for (int size : {1, 4, 16, 64, 256}) {
        float* ptr = arena.allocate<float>(size);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        assert(addr % 64 == 0);
    }
    std::cout << "  ✓ All allocations maintain 64-byte alignment\n";

    std::cout << "\n✅ All cache-line alignment tests passed!\n";
}

void test_alignment_constants() {
    std::cout << "\nVerifying alignment constants...\n";

    // Check static constants
    assert(Arena::SIMD_ALIGNMENT == 16);
    assert(Arena::CACHE_LINE_ALIGNMENT == 64);
    assert(Arena::ALIGNMENT == 64);

    // Verify constraint
    assert(Arena::CACHE_LINE_ALIGNMENT >= Arena::SIMD_ALIGNMENT);

    std::cout << "  ✓ SIMD_ALIGNMENT = " << Arena::SIMD_ALIGNMENT << " (NEON 128-bit)\n";
    std::cout << "  ✓ CACHE_LINE_ALIGNMENT = " << Arena::CACHE_LINE_ALIGNMENT << "\n";
    std::cout << "  ✓ ALIGNMENT = " << Arena::ALIGNMENT << " (used for allocations)\n";
    std::cout << "  ✓ Cache line alignment >= SIMD alignment: "
              << (Arena::CACHE_LINE_ALIGNMENT >= Arena::SIMD_ALIGNMENT ? "YES" : "NO") << "\n";

    std::cout << "\n✅ Alignment constants verified!\n";
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "Arena Allocator SIMD Alignment Test Suite\n";
    std::cout << "==============================================\n\n";

    try {
        test_alignment_constants();
        test_simd_alignment();
        test_cache_line_alignment();

        std::cout << "\n==============================================\n";
        std::cout << "✅ ALL TESTS PASSED\n";
        std::cout << "==============================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
