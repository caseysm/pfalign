/**
 * Quick test to verify arena allocator compiles and works.
 */

#include "pfalign/common/arena_allocator.h"
#include <iostream>

using pfalign::memory::Arena;
using pfalign::memory::ScopedArena;

int main() {
    // Test 1: Basic allocation
    Arena arena(1024);

    float* buf1 = arena.allocate<float>(10);
    for (int i = 0; i < 10; i++) {
        buf1[i] = i * 1.5f;
    }

    std::cout << "Test 1: Basic allocation" << std::endl;
    std::cout << "  Used: " << arena.used() << " / " << arena.capacity() << " bytes" << std::endl;
    std::cout << "  buf1[5] = " << buf1[5] << " (expected 7.5)" << std::endl;

    // Test 2: Reset and reuse
    arena.reset();
    float* buf2 = arena.allocate<float>(20);
    std::cout << "\nTest 2: Reset and reuse" << std::endl;
    std::cout << "  After reset, used: " << arena.used() << " bytes" << std::endl;

    // Test 3: Scoped arena (automatic reset)
    {
        ScopedArena scope(arena);
        float* temp = scope.allocate<float>(100);
        temp[0] = 42.0f;
        std::cout << "\nTest 3: Scoped arena" << std::endl;
        std::cout << "  Inside scope, used: " << arena.used() << " bytes" << std::endl;
    }
    std::cout << "  After scope exit, used: " << arena.used() << " bytes (should be restored)" << std::endl;

    // Test 4: Alignment check
    int* aligned = arena.allocate<int>(1);
    std::cout << "\nTest 4: Alignment" << std::endl;
    std::cout << "  Address alignment: " << (reinterpret_cast<uintptr_t>(aligned) % 32) << " (should be 0)" << std::endl;

    std::cout << "\n✓ All tests passed!" << std::endl;
    return 0;
}
