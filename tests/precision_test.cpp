/**
 * Test temperature invariance with different floating point precisions.
 *
 * Tests smith_waterman_direct_regular with:
 * - float32 (standard)
 * - float64 (double precision)
 * - float16 (half precision) - if available
 *
 * Goal: Determine if 0.61% temperature variation is due to precision or fundamental.
 */

#include "../pfalign/_core/src/pfalign/primitives/smith_waterman/smith_waterman_templated.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

using namespace pfalign::smith_waterman;

/**
 * Test a specific precision.
 */
template <typename T>
void test_precision(const std::string& precision_name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Testing with " << precision_name << "\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Small test matrix (3x3) - same as C++ unit test
    const int L1 = 3, L2 = 3;
    T scores[9] = {
        static_cast<T>(1.0), static_cast<T>(0.5), static_cast<T>(0.3),
        static_cast<T>(0.6), static_cast<T>(1.2), static_cast<T>(0.4),
        static_cast<T>(0.3), static_cast<T>(0.5), static_cast<T>(0.9)
    };

    // Temperature range
    std::vector<T> temperatures = {
        static_cast<T>(0.1),
        static_cast<T>(0.5),
        static_cast<T>(1.0),
        static_cast<T>(2.0),
        static_cast<T>(10.0)
    };

    std::cout << "Matrix: " << L1 << "*" << L2 << "\n";
    std::cout << "Gap penalty: -0.2\n";
    std::cout << "Temperatures: [0.1, 0.5, 1.0, 2.0, 10.0]\n\n";

    std::cout << std::left << std::setw(12) << "Temperature"
              << std::setw(18) << "Partition"
              << std::setw(18) << "Post Sum"
              << std::setw(15) << "% vs Mean" << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::vector<T> post_sums;
    std::vector<T> partitions;

    for (T temp : temperatures) {
        // Configure
        SWConfigT<T> config;
        config.gap = static_cast<T>(-0.2);
        config.temperature = temp;

        // Allocate
        T alpha[(L1 + 1) * (L2 + 1)];
        T partition;
        T posteriors[L1 * L2];

        // Forward pass
        smith_waterman_direct_regular_templated(scores, L1, L2, config, alpha, &partition);

        // Backward pass
        smith_waterman_direct_regular_backward_templated(alpha, scores, L1, L2, config,
                                                         partition, posteriors);

        // Sum posteriors
        T post_sum = static_cast<T>(0);
        for (int i = 0; i < L1 * L2; i++) {
            post_sum += posteriors[i];
        }

        post_sums.push_back(post_sum);
        partitions.push_back(partition);
    }

    // Calculate mean
    T mean_sum = static_cast<T>(0);
    for (T ps : post_sums) {
        mean_sum += ps;
    }
    mean_sum /= static_cast<T>(post_sums.size());

    // Print with deviations
    for (size_t i = 0; i < temperatures.size(); i++) {
        T deviation = (post_sums[i] / mean_sum - static_cast<T>(1)) * static_cast<T>(100);
        std::cout << std::setw(12) << static_cast<double>(temperatures[i])
                  << std::setw(18) << static_cast<double>(partitions[i])
                  << std::setw(18) << static_cast<double>(post_sums[i])
                  << std::setw(15) << static_cast<double>(deviation) << "%\n";
    }

    std::cout << std::string(80, '-') << "\n";

    // Statistics
    T max_dev = static_cast<T>(0);
    for (T ps : post_sums) {
        T dev = (ps > mean_sum) ? (ps - mean_sum) : (mean_sum - ps);
        max_dev = (dev > max_dev) ? dev : max_dev;
    }
    T rel_dev = (max_dev / mean_sum) * static_cast<T>(100);

    std::cout << "\n📊 Statistics:\n";
    std::cout << "  Mean posterior sum:    " << static_cast<double>(mean_sum) << "\n";
    std::cout << "  Max deviation:         " << static_cast<double>(max_dev) << "\n";
    std::cout << "  Relative deviation:    " << static_cast<double>(rel_dev) << "%\n";
    std::cout << "  Target:                <5%\n";
    std::cout << "  Status:                " << (rel_dev < 5.0 ? "✓ PASS" : "✗ FAIL") << "\n";

    // Partition scaling
    bool all_increase = true;
    for (size_t i = 1; i < partitions.size(); i++) {
        if (partitions[i] <= partitions[i-1]) {
            all_increase = false;
            break;
        }
    }

    std::cout << "\n📈 Partition Scaling:\n";
    std::cout << "  Monotonically increases: " << (all_increase ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  T=0.1:  " << static_cast<double>(partitions[0]) << "\n";
    std::cout << "  T=1.0:  " << static_cast<double>(partitions[2]) << "\n";
    std::cout << "  T=10.0: " << static_cast<double>(partitions[4]) << "\n";
}

/**
 * Test with realistic protein-sized matrix.
 */
template <typename T>
void test_realistic_size(const std::string& precision_name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Testing REALISTIC SIZE with " << precision_name << "\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Medium matrix (30x30)
    const int L1 = 30, L2 = 30;
    std::vector<T> scores(L1 * L2);

    // Generate random similarity matrix (reproducible)
    unsigned int seed = 42;
    for (int i = 0; i < L1 * L2; i++) {
        // Simple LCG random
        seed = seed * 1103515245 + 12345;
        float r = static_cast<float>((seed / 65536) % 32768) / 32768.0f;
        scores[i] = static_cast<T>(r * 4.0 - 2.0 + 5.0);  // Range ~[3, 7]
    }

    // Temperature range (realistic)
    std::vector<T> temperatures = {
        static_cast<T>(0.5),
        static_cast<T>(0.75),
        static_cast<T>(1.0),
        static_cast<T>(1.25),
        static_cast<T>(1.5),
        static_cast<T>(2.0)
    };

    std::cout << "Matrix: " << L1 << "*" << L2 << "\n";
    std::cout << "Gap penalties: -11.0 (realistic)\n";
    std::cout << "Temperatures: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]\n\n";

    std::vector<T> post_sums;
    std::vector<T> partitions;

    for (T temp : temperatures) {
        SWConfigT<T> config;
        config.gap = static_cast<T>(-11.0);
        config.temperature = temp;

        std::vector<T> alpha((L1 + 1) * (L2 + 1));
        T partition;
        std::vector<T> posteriors(L1 * L2);

        smith_waterman_direct_regular_templated(scores.data(), L1, L2, config,
                                                alpha.data(), &partition);
        smith_waterman_direct_regular_backward_templated(alpha.data(), scores.data(),
                                                         L1, L2, config, partition,
                                                         posteriors.data());

        T post_sum = static_cast<T>(0);
        for (T p : posteriors) {
            post_sum += p;
        }

        post_sums.push_back(post_sum);
        partitions.push_back(partition);
    }

    // Statistics
    T mean_sum = static_cast<T>(0);
    for (T ps : post_sums) mean_sum += ps;
    mean_sum /= static_cast<T>(post_sums.size());

    T max_dev = static_cast<T>(0);
    for (T ps : post_sums) {
        T dev = (ps > mean_sum) ? (ps - mean_sum) : (mean_sum - ps);
        max_dev = (dev > max_dev) ? dev : max_dev;
    }
    T rel_dev = (max_dev / mean_sum) * static_cast<T>(100);

    std::cout << "📊 Results:\n";
    std::cout << "  Mean posterior sum:    " << static_cast<double>(mean_sum) << "\n";
    std::cout << "  Max deviation:         " << static_cast<double>(max_dev) << "\n";
    std::cout << "  Relative deviation:    " << static_cast<double>(rel_dev) << "%\n";
    std::cout << "  Target:                <5%\n";
    std::cout << "  Status:                " << (rel_dev < 5.0 ? "✓ PASS" : "✗ FAIL") << "\n";
}

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "PRECISION ANALYSIS: Does floating point precision affect temperature invariance?\n";
    std::cout << std::string(80, '=') << "\n";

    // Test 1: Small matrix (edge case from C++ unit test)
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 1: Small Matrix (3*3) - Edge Case\n";
    std::cout << std::string(80, '=') << "\n";

#ifdef __ARM_FP16_FORMAT_IEEE
    test_precision<__fp16>("FLOAT16 (__fp16)");
#endif
    test_precision<float>("FLOAT32");
    test_precision<double>("FLOAT64");

    // Test 2: Realistic size
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST 2: Realistic Matrix (30*30)\n";
    std::cout << std::string(80, '=') << "\n";

#ifdef __ARM_FP16_FORMAT_IEEE
    test_realistic_size<__fp16>("FLOAT16 (__fp16)");
#endif
    test_realistic_size<float>("FLOAT32");
    test_realistic_size<double>("FLOAT64");

    // Conclusion
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "CONCLUSION:\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "If float16 shows higher variation:\n";
    std::cout << "  → Limited precision affects results\n";
    std::cout << "If float64 shows significantly lower variation than float32:\n";
    std::cout << "  → Variation is due to PRECISION (numerical error)\n";
    std::cout << "If float64 shows similar variation to float32:\n";
    std::cout << "  → Variation is FUNDAMENTAL (mathematical behavior)\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
