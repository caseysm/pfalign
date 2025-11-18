/**
 * Test float16 support on this system.
 */

#include <iostream>
#include <cmath>

// Test different float16 types
int main() {
    std::cout << "Testing float16 support on Apple M3 Pro...\n\n";

#if defined(__ARM_NEON)
    std::cout << "✓ ARM NEON is available\n";
#else
    std::cout << "✗ ARM NEON is NOT available\n";
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    std::cout << "✓ FP16 vector arithmetic is available\n";
#else
    std::cout << "✗ FP16 vector arithmetic is NOT available\n";
#endif

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    std::cout << "✓ FP16 scalar arithmetic is available\n";
#else
    std::cout << "✗ FP16 scalar arithmetic is NOT available\n";
#endif

    // Test __fp16 type (ARM extension)
#ifdef __ARM_FP16_FORMAT_IEEE
    std::cout << "\n✓ __fp16 type is available (IEEE format)\n";
    __fp16 half_val = 1.5;
    float float_val = half_val;
    std::cout << "  Test: __fp16(1.5) converts to float(" << float_val << ")\n";
#else
    std::cout << "\n✗ __fp16 type is NOT available\n";
#endif

    // Test _Float16 type (C++ standard)
#if defined(__STDCPP_FLOAT16_T__)
    std::cout << "\n✓ _Float16 standard type is available\n";
    _Float16 f16_val = 2.5;
    float f32_val = f16_val;
    std::cout << "  Test: _Float16(2.5) converts to float(" << f32_val << ")\n";
#else
    std::cout << "\n✗ _Float16 standard type is NOT available\n";
#endif

    std::cout << "\nConclusion:\n";
    std::cout << "We should be able to use __fp16 on Apple Silicon for half-precision testing.\n";

    return 0;
}
