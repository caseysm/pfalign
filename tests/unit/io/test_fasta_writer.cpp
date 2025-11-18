/**
 * Test FASTA alignment writer with gaps.
 *
 * Verifies that:
 * - Alignment paths are correctly converted to gapped sequences
 * - Gaps are inserted at correct positions
 * - FASTA output format is valid
 * - Line wrapping works correctly
 * - Metadata is included in headers
 */

#include "pfalign/io/fasta_writer.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>  // For remove()

using pfalign::io::alignment_path_to_gapped_sequences;
using pfalign::io::write_alignment_fasta;
using pfalign::AlignmentPair;

void test_perfect_alignment() {
    std::cout << "=== Test 1: Perfect Alignment (No Gaps) ===" << std::endl;

    // Perfect diagonal alignment: 0->0, 1->1, 2->2, 3->3, 4->4
    AlignmentPair path[] = {
        {0, 0, 0.2f},
        {1, 1, 0.2f},
        {2, 2, 0.2f},
        {3, 3, 0.2f},
        {4, 4, 0.2f}
    };

    std::string seq1 = "ACDEF";
    std::string seq2 = "ACDEF";
    std::string gapped1, gapped2;

    alignment_path_to_gapped_sequences(path, 5, seq1, seq2, gapped1, gapped2);

    std::cout << "  Original seq1: " << seq1 << std::endl;
    std::cout << "  Original seq2: " << seq2 << std::endl;
    std::cout << "  Gapped seq1:   " << gapped1 << std::endl;
    std::cout << "  Gapped seq2:   " << gapped2 << std::endl;

    bool test_passed = (gapped1 == "ACDEF") && (gapped2 == "ACDEF");
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;

    assert(test_passed);
}

void test_gap_in_seq2() {
    std::cout << "=== Test 2: Gap in Sequence 2 ===" << std::endl;

    // Path with gap in seq2: 0->0, 1->1, 2->2, 3->3, 4->-1 (gap), 5->4
    AlignmentPair path[] = {
        {0, 0, 0.15f},
        {1, 1, 0.15f},
        {2, 2, 0.15f},
        {3, 3, 0.15f},
        {4, -1, 0.0f},  // Gap in seq2 (F has no match)
        {5, 4, 0.15f}
    };

    std::string seq1 = "ACDEFG";
    std::string seq2 = "ACDEG";  // Missing F
    std::string gapped1, gapped2;

    alignment_path_to_gapped_sequences(path, 6, seq1, seq2, gapped1, gapped2);

    std::cout << "  Original seq1: " << seq1 << std::endl;
    std::cout << "  Original seq2: " << seq2 << std::endl;
    std::cout << "  Gapped seq1:   " << gapped1 << std::endl;
    std::cout << "  Gapped seq2:   " << gapped2 << std::endl;

    bool test_passed = (gapped1 == "ACDEFG") && (gapped2 == "ACDE-G");
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;

    assert(test_passed);
}

void test_gap_in_seq1() {
    std::cout << "=== Test 3: Gap in Sequence 1 ===" << std::endl;

    // Path with gap in seq1: 0->0, 1->1, 2->2, -1->3 (gap), 3->4
    AlignmentPair path[] = {
        {0, 0, 0.2f},
        {1, 1, 0.2f},
        {2, 2, 0.2f},
        {-1, 3, 0.0f},  // Gap in seq1 (E has no match)
        {3, 4, 0.2f}
    };

    std::string seq1 = "ACDG";  // Missing E
    std::string seq2 = "ACDEG";
    std::string gapped1, gapped2;

    alignment_path_to_gapped_sequences(path, 5, seq1, seq2, gapped1, gapped2);

    std::cout << "  Original seq1: " << seq1 << std::endl;
    std::cout << "  Original seq2: " << seq2 << std::endl;
    std::cout << "  Gapped seq1:   " << gapped1 << std::endl;
    std::cout << "  Gapped seq2:   " << gapped2 << std::endl;

    bool test_passed = (gapped1 == "ACD-G") && (gapped2 == "ACDEG");
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;

    assert(test_passed);
}

void test_multiple_gaps() {
    std::cout << "=== Test 4: Multiple Gaps ===" << std::endl;

    // Complex alignment with multiple gaps
    AlignmentPair path[] = {
        {0, 0, 0.1f},    // A-A
        {-1, 1, 0.0f},   // -B (gap in seq1)
        {1, 2, 0.1f},    // C-C
        {2, -1, 0.0f},   // D- (gap in seq2)
        {3, 3, 0.1f},    // E-E
        {4, -1, 0.0f},   // F- (gap in seq2)
        {-1, 4, 0.0f},   // -G (gap in seq1)
        {5, 5, 0.1f}     // H-H
    };

    std::string seq1 = "ACDEFH";
    std::string seq2 = "ABCEGH";
    std::string gapped1, gapped2;

    alignment_path_to_gapped_sequences(path, 8, seq1, seq2, gapped1, gapped2);

    std::cout << "  Original seq1: " << seq1 << std::endl;
    std::cout << "  Original seq2: " << seq2 << std::endl;
    std::cout << "  Gapped seq1:   " << gapped1 << std::endl;
    std::cout << "  Gapped seq2:   " << gapped2 << std::endl;

    bool test_passed = (gapped1 == "A-CDEF-H") && (gapped2 == "ABC-E-GH");
    std::cout << (test_passed ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
    std::cout << std::endl;

    assert(test_passed);
}

void test_fasta_output() {
    std::cout << "=== Test 5: FASTA File Output ===" << std::endl;

    // Create a simple alignment
    AlignmentPair path[] = {
        {0, 0, 0.2f},
        {1, 1, 0.2f},
        {2, -1, 0.0f},  // Gap in seq2
        {3, 2, 0.2f},
        {4, 3, 0.2f}
    };

    std::string seq1 = "ACDEF";
    std::string seq2 = "ACEF";

    const char* output_file = "test_alignment.fasta";

    // Write alignment
    bool success = write_alignment_fasta(
        output_file,
        "protein1", "protein2",
        seq1, seq2,
        path, 5,
        0.845f,  // score
        123.45f  // partition
    );

    std::cout << "  Write success: " << (success ? "yes" : "no") << std::endl;

    if (success) {
        // Read back and verify
        std::ifstream in(output_file);
        std::string line;
        int line_num = 0;

        std::cout << "  Output file contents:" << std::endl;
        while (std::getline(in, line)) {
            std::cout << "    " << line << std::endl;
            line_num++;

            // Verify key elements
            if (line_num == 1) {
                // Header for protein1
                assert(line.find("protein1") != std::string::npos);
                assert(line.find("Score:") != std::string::npos);
                assert(line.find("0.845") != std::string::npos);
            } else if (line_num == 2) {
                // Gapped sequence 1
                assert(line == "ACDEF");
            } else if (line_num == 3) {
                // Header for protein2
                assert(line.find("protein2") != std::string::npos);
            } else if (line_num == 4) {
                // Gapped sequence 2
                assert(line == "AC-EF");
            }
        }
        in.close();

        // Clean up
        std::remove(output_file);

        std::cout << "  ✓ Test passed" << std::endl;
    } else {
        std::cout << "  ✗ Test failed: could not write file" << std::endl;
        assert(false);
    }

    std::cout << std::endl;
}

void test_line_wrapping() {
    std::cout << "=== Test 6: Line Wrapping ===" << std::endl;

    // Create a long sequence that will require line wrapping
    std::string long_seq1(100, 'A');
    std::string long_seq2(100, 'A');

    // Perfect alignment (no gaps)
    std::vector<AlignmentPair> path;
    for (int i = 0; i < 100; i++) {
        path.push_back({i, i, 0.01f});
    }

    const char* output_file = "test_long_alignment.fasta";

    bool success = write_alignment_fasta(
        output_file,
        "long_protein1", "long_protein2",
        long_seq1, long_seq2,
        path.data(), path.size(),
        0.99f,
        500.0f
    );

    std::cout << "  Write success: " << (success ? "yes" : "no") << std::endl;

    if (success) {
        // Read back and check line lengths
        std::ifstream in(output_file);
        std::string line;
        bool wrapping_correct = true;

        while (std::getline(in, line)) {
            if (line[0] != '>') {  // Sequence line, not header
                if (line.length() > 80) {
                    wrapping_correct = false;
                    std::cout << "  Line too long: " << line.length() << " characters" << std::endl;
                }
            }
        }
        in.close();

        // Clean up
        std::remove(output_file);

        std::cout << "  Line wrapping: " << (wrapping_correct ? "correct" : "incorrect") << std::endl;
        std::cout << (wrapping_correct ? "  ✓ Test passed" : "  ✗ Test failed") << std::endl;
        assert(wrapping_correct);
    } else {
        std::cout << "  ✗ Test failed: could not write file" << std::endl;
        assert(false);
    }

    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  FASTA Alignment Writer Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_perfect_alignment();
    test_gap_in_seq2();
    test_gap_in_seq1();
    test_multiple_gaps();
    test_fasta_output();
    test_line_wrapping();

    std::cout << "========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "- Gap insertion works correctly for seq1 and seq2" << std::endl;
    std::cout << "- Multiple gaps handled properly" << std::endl;
    std::cout << "- FASTA output format is valid" << std::endl;
    std::cout << "- Line wrapping at 80 characters" << std::endl;
    std::cout << "- Metadata included in headers" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
