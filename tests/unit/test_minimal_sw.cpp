#include <cstdio>
#include <cstring>
#include "pfalign/primitives/smith_waterman/smith_waterman.h"
#include "pfalign/dispatch/scalar_traits.h"

using namespace pfalign;
using namespace pfalign::smith_waterman;

int main() {
    printf("=== Minimal Smith-Waterman Test ===\n");

    int L1 = 10;
    int L2 = 10;

    // Create simple similarity matrix (all 1.0)
    float* similarity = new float[L1 * L2];
    for (int i = 0; i < L1 * L2; i++) {
        similarity[i] = 1.0f;
    }

    // Allocate DP matrix (3 states for affine)
    float* dp_matrix = new float[L1 * L2 * 3];
    std::memset(dp_matrix, 0, L1 * L2 * 3 * sizeof(float));

    // Configure SW
    SWConfig config;
    config.gap = -1.0f;
    config.gap_open = -11.0f;
    config.gap_extend = -1.0f;
    config.temperature = 1.0f;
    config.affine = true;

    float partition;

    printf("Calling smith_waterman_jax_affine_flexible...\n");
    fflush(stdout);

    smith_waterman_jax_affine_flexible<pfalign::ScalarBackend>(
        similarity,
        L1, L2,
        config,
        dp_matrix,
        &partition
    );

    printf("✓ SW completed, partition=%f\n", partition);

    delete[] similarity;
    delete[] dp_matrix;

    return 0;
}
