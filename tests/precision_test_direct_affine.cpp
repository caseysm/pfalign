/**
 * Test temperature invariance for DIRECT AFFINE (the production default).
 */

#include "../pfalign/_core/src/pfalign/primitives/smith_waterman/smith_waterman.h"
#include "../pfalign/_core/src/pfalign/dispatch/scalar_traits.h"
#include "../pfalign/_core/src/pfalign/common/growable_arena.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace pfalign::smith_waterman;
using pfalign::ScalarBackend;

void test_direct_affine(int L1, int L2, const std::string& description) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << description << " - Direct Affine (PRODUCTION DEFAULT)\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Generate similarity matrix
    std::vector<float> scores(L1 * L2);
    unsigned int seed = 42;
    for (int i = 0; i < L1 * L2; i++) {
        seed = seed * 1103515245 + 12345;
        float r = static_cast<float>((seed / 65536) % 32768) / 32768.0f;
        scores[i] = r * 4.0f - 2.0f + 5.0f;  // Range ~[3, 7]
    }

    // Temperature range (realistic)
    std::vector<float> temperatures = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f};

    std::cout << "Matrix: " << L1 << "*" << L2 << "\n";
    std::cout << "Gap: open=-11.0, extend=-1.0\n";
    std::cout << "Temperatures: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]\n\n";

    std::vector<float> post_sums;
    std::vector<float> partitions;

    for (float temp : temperatures) {
        // Configure (affine gaps)
        SWConfig config;
        config.affine = true;
        config.gap_open = -11.0f;
        config.gap_extend = -1.0f;
        config.temperature = temp;

        // Allocate (3 states: M, I, D for affine)
        std::vector<float> alpha((L1 + 1) * (L2 + 1) * 3);
        float partition;
        std::vector<float> posteriors(L1 * L2);

        // Forward pass
        smith_waterman_direct_affine<ScalarBackend>(
            scores.data(), L1, L2, config, alpha.data(), &partition);

        // Backward pass
        pfalign::memory::GrowableArena temp_arena(10);
        smith_waterman_direct_affine_backward<ScalarBackend>(
            alpha.data(), scores.data(), L1, L2, config, partition,
            posteriors.data(), &temp_arena);

        // Sum posteriors
        float post_sum = 0.0f;
        for (float p : posteriors) {
            post_sum += p;
        }

        post_sums.push_back(post_sum);
        partitions.push_back(partition);
    }

    // Statistics
    float mean_sum = 0.0f;
    for (float ps : post_sums) mean_sum += ps;
    mean_sum /= static_cast<float>(post_sums.size());

    float max_dev = 0.0f;
    for (float ps : post_sums) {
        float dev = (ps > mean_sum) ? (ps - mean_sum) : (mean_sum - ps);
        max_dev = (dev > max_dev) ? dev : max_dev;
    }
    float rel_dev = (max_dev / mean_sum) * 100.0f;

    // Print results table
    std::cout << std::left << std::setw(12) << "Temp"
              << std::setw(18) << "Partition"
              << std::setw(18) << "Post Sum"
              << std::setw(15) << "% vs Mean" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < temperatures.size(); i++) {
        float deviation = (post_sums[i] / mean_sum - 1.0f) * 100.0f;
        std::cout << std::setw(12) << temperatures[i]
                  << std::setw(18) << partitions[i]
                  << std::setw(18) << post_sums[i]
                  << std::setw(15) << deviation << "%\n";
    }
    std::cout << std::string(80, '-') << "\n";

    std::cout << "\n📊 Results:\n";
    std::cout << "  Mean posterior sum:    " << mean_sum << "\n";
    std::cout << "  Max deviation:         " << max_dev << "\n";
    std::cout << "  Relative deviation:    " << rel_dev << "%\n";
    std::cout << "  Target:                <5%\n";
    std::cout << "  Status:                " << (rel_dev < 5.0f ? "✓ PASS" : "✗ FAIL") << "\n";
}

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "DIRECT AFFINE TEMPERATURE INVARIANCE TEST\n";
    std::cout << "(This is the PRODUCTION DEFAULT algorithm)\n";
    std::cout << std::string(80, '=') << "\n";

    // Test different sizes
    test_direct_affine(3, 3, "Small Matrix (3*3)");
    test_direct_affine(10, 10, "Small-Medium Matrix (10*10)");
    test_direct_affine(30, 30, "Medium Matrix (30*30)");
    test_direct_affine(50, 40, "Large Matrix (50*40)");

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SUMMARY:\n";
    std::cout << "Testing the production default (Direct Affine) to verify temperature invariance.\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
