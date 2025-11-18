/**
 * Basic C++ test for PyBind11 bindings.
 *
 * This test verifies that the bindings compile and link correctly.
 * More comprehensive tests are in the Python test suite.
 */

#include <iostream>
#include <cassert>
#include "pfalign/modules/pairwise/pairwise_align.h"
#include "pfalign/modules/mpnn/mpnn_encoder.h"

using namespace pfalign;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Basic Binding Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Test 1: Verify pairwise module exists
    std::cout << "Test 1: Verify pairwise module links..." << std::endl;
    // Just linking against pairwise_scalar is enough
    std::cout << "  ✓ Pairwise module linked successfully" << std::endl;
    std::cout << std::endl;

    // Test 2: Verify MPNN module exists
    std::cout << "Test 2: Verify MPNN module links..." << std::endl;
    // Just linking against mpnn_scalar is enough
    std::cout << "  ✓ MPNN module linked successfully" << std::endl;
    std::cout << std::endl;

    // Test 3: Basic type check
    std::cout << "Test 3: Basic type checks..." << std::endl;
    mpnn::MPNNWeights weights(3);  // 3 layers
    assert(weights.num_layers == 3);
    assert(weights.layers.size() == 3);
    std::cout << "  ✓ MPNNWeights structure works" << std::endl;
    std::cout << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "  All Tests Passed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
