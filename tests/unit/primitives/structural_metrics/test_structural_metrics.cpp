/**
 * Unit tests for structural alignment metrics.
 */

#include "pfalign/primitives/structural_metrics/structural_metrics_impl.h"
#include "pfalign/dispatch/scalar_traits.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace pfalign;

const float EPS = 1e-4f;

bool test_tm_score_perfect() {
    std::cout << "\n=== Test: TM-Score (Perfect Match) ===\n";

    int N = 10;
    float P[30];
    for (int i = 0; i < N; i++) {
        P[i*3 + 0] = static_cast<float>(i);
        P[i*3 + 1] = static_cast<float>(i * 2);
        P[i*3 + 2] = static_cast<float>(i * 3);
    }

    float tm = structural_metrics::compute_tm_score<ScalarBackend>(P, P, N, N);
    std::cout << "TM-score (identical): " << tm << "\n";

    if (std::abs(tm - 1.0f) < EPS) {
        std::cout << "  ✓ Perfect match (TM ~= 1.0)\n";
        return true;
    } else {
        std::cout << "  ✗ Expected 1.0\n";
        return false;
    }
}

bool test_gdt_perfect() {
    std::cout << "\n=== Test: GDT (Perfect Match) ===\n";

    int N = 10;
    float P[30];
    for (int i = 0; i < N; i++) {
        P[i*3 + 0] = static_cast<float>(i);
        P[i*3 + 1] = static_cast<float>(i * 2);
        P[i*3 + 2] = static_cast<float>(i * 3);
    }

    float gdt_ts, gdt_ha;
    structural_metrics::compute_gdt<ScalarBackend>(P, P, N, &gdt_ts, &gdt_ha);

    std::cout << "GDT-TS: " << gdt_ts << "\n";
    std::cout << "GDT-HA: " << gdt_ha << "\n";

    if (std::abs(gdt_ts - 1.0f) < EPS && std::abs(gdt_ha - 1.0f) < EPS) {
        std::cout << "  ✓ Perfect match (GDT ~= 1.0)\n";
        return true;
    } else {
        std::cout << "  ✗ Expected 1.0\n";
        return false;
    }
}

bool test_gdt_with_noise() {
    std::cout << "\n=== Test: GDT (With Small Noise) ===\n";

    int N = 100;
    float P[300], Q[300];

    for (int i = 0; i < N; i++) {
        P[i*3 + 0] = static_cast<float>(i);
        P[i*3 + 1] = static_cast<float>(i * 2);
        P[i*3 + 2] = static_cast<float>(i * 3);

        // Add 0.3Å noise (should be within 0.5Å cutoff)
        Q[i*3 + 0] = P[i*3 + 0] + 0.3f;
        Q[i*3 + 1] = P[i*3 + 1] + 0.2f;
        Q[i*3 + 2] = P[i*3 + 2] + 0.1f;
    }

    float gdt_ts, gdt_ha, p0_5, p1, p2, p4, p8;
    structural_metrics::compute_gdt<ScalarBackend>(
        P, Q, N, &gdt_ts, &gdt_ha, &p1, &p2, &p4, &p8, &p0_5
    );

    std::cout << "GDT-TS: " << gdt_ts << "\n";
    std::cout << "GDT-HA: " << gdt_ha << "\n";
    std::cout << "  P(< 0.5Å): " << p0_5 * 100 << "%\n";
    std::cout << "  P(< 1.0Å): " << p1 * 100 << "%\n";

    // With ~0.3Å noise, all should be < 1Å
    if (p1 > 0.99f) {
        std::cout << "  ✓ All residues < 1Å (as expected)\n";
        return true;
    } else {
        std::cout << "  ✗ Expected all < 1Å\n";
        return false;
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Structural Metrics Tests\n";
    std::cout << "========================================\n";

    int passed = 0;
    int total = 0;

    #define RUN_TEST(test_func) \
        total++; \
        if (test_func()) { \
            passed++; \
            std::cout << "  PASS\n"; \
        } else { \
            std::cout << "  FAIL\n"; \
        }

    RUN_TEST(test_tm_score_perfect);
    RUN_TEST(test_gdt_perfect);
    RUN_TEST(test_gdt_with_noise);

    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " / " << total << " tests passed\n";
    std::cout << "========================================\n";

    return (passed == total) ? 0 : 1;
}
