/**
 * Unit tests for sequence utilities.
 */

#include "pfalign/io/sequence_utils.h"
#include <iostream>
#include <cassert>

using pfalign::io::three_to_one;
using pfalign::io::extract_sequence;
using pfalign::io::insert_gaps;
using pfalign::io::Residue;
using pfalign::AlignmentPair;

//==============================================================================
// Test 1: Three-to-One Conversion (Standard AAs)
//==============================================================================

bool test_three_to_one_standard() {
    std::cout << "=== Test 1: Three-to-One (Standard AAs) ===" << std::endl;

    // Test all 20 standard amino acids (capitalized format)
    assert(three_to_one("Ala") == 'A');
    assert(three_to_one("Cys") == 'C');
    assert(three_to_one("Asp") == 'D');
    assert(three_to_one("Glu") == 'E');
    assert(three_to_one("Phe") == 'F');
    assert(three_to_one("Gly") == 'G');
    assert(three_to_one("His") == 'H');
    assert(three_to_one("Ile") == 'I');
    assert(three_to_one("Lys") == 'K');
    assert(three_to_one("Leu") == 'L');
    assert(three_to_one("Met") == 'M');
    assert(three_to_one("Asn") == 'N');
    assert(three_to_one("Pro") == 'P');
    assert(three_to_one("Gln") == 'Q');
    assert(three_to_one("Arg") == 'R');
    assert(three_to_one("Ser") == 'S');
    assert(three_to_one("Thr") == 'T');
    assert(three_to_one("Val") == 'V');
    assert(three_to_one("Trp") == 'W');
    assert(three_to_one("Tyr") == 'Y');

    // Test uppercase format
    assert(three_to_one("ALA") == 'A');
    assert(three_to_one("CYS") == 'C');
    assert(three_to_one("TRP") == 'W');

    std::cout << "  ✓ PASS (all 20 standard AAs)" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 2: Three-to-One Conversion (Extended AAs)
//==============================================================================

bool test_three_to_one_extended() {
    std::cout << "=== Test 2: Three-to-One (Extended AAs) ===" << std::endl;

    assert(three_to_one("Asx") == 'B');  // Asp or Asn
    assert(three_to_one("Glx") == 'Z');  // Glu or Gln
    assert(three_to_one("Xaa") == 'X');  // Unknown
    assert(three_to_one("Pyl") == 'O');  // Pyrrolysine
    assert(three_to_one("Sec") == 'U');  // Selenocysteine
    assert(three_to_one("Xle") == 'J');  // Leu or Ile

    // Uppercase
    assert(three_to_one("ASX") == 'B');
    assert(three_to_one("GLX") == 'Z');

    std::cout << "  ✓ PASS (extended AAs)" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 3: Three-to-One Conversion (Unknown)
//==============================================================================

bool test_three_to_one_unknown() {
    std::cout << "=== Test 3: Three-to-One (Unknown) ===" << std::endl;

    assert(three_to_one("XXX") == 'X');
    assert(three_to_one("UNK") == 'X');
    assert(three_to_one("FOO") == 'X');

    std::cout << "  ✓ PASS (unknown residues → X)" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 4: Extract Sequence from Residues
//==============================================================================

bool test_extract_sequence() {
    std::cout << "=== Test 4: Extract Sequence ===" << std::endl;

    std::vector<Residue> residues = {
        Residue(1, ' ', "Ala"),
        Residue(2, ' ', "Cys"),
        Residue(3, ' ', "Glu"),
        Residue(4, ' ', "Phe")
    };

    std::string seq = extract_sequence(residues);
    assert(seq == "ACEF");

    std::cout << "  Sequence: " << seq << " (expected: ACEF)" << std::endl;
    std::cout << "  ✓ PASS" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 5: Extract Sequence (Mixed Case)
//==============================================================================

bool test_extract_sequence_mixed_case() {
    std::cout << "=== Test 5: Extract Sequence (Mixed Case) ===" << std::endl;

    std::vector<Residue> residues = {
        Residue(1, ' ', "ALA"),  // Uppercase
        Residue(2, ' ', "Cys"),  // Capitalized
        Residue(3, ' ', "glu"),  // Lowercase (should handle)
        Residue(4, ' ', "PHE")   // Uppercase
    };

    std::string seq = extract_sequence(residues);
    assert(seq == "ACEF");

    std::cout << "  Sequence: " << seq << " (expected: ACEF)" << std::endl;
    std::cout << "  ✓ PASS" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 6: Insert Gaps (No Gaps)
//==============================================================================

bool test_insert_gaps_no_gaps() {
    std::cout << "=== Test 6: Insert Gaps (No Gaps) ===" << std::endl;

    std::string seq1 = "ACE";
    std::string seq2 = "ACE";

    AlignmentPair path[] = {
        {0, 0, 0.9},
        {1, 1, 0.9},
        {2, 2, 0.9}
    };

    std::string aligned1 = insert_gaps(seq1, path, 3, true);
    std::string aligned2 = insert_gaps(seq2, path, 3, false);

    assert(aligned1 == "ACE");
    assert(aligned2 == "ACE");

    std::cout << "  Seq1: " << aligned1 << " (expected: ACE)" << std::endl;
    std::cout << "  Seq2: " << aligned2 << " (expected: ACE)" << std::endl;
    std::cout << "  ✓ PASS" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 7: Insert Gaps (Gap in Seq2)
//==============================================================================

bool test_insert_gaps_seq2() {
    std::cout << "=== Test 7: Insert Gaps (Gap in Seq2) ===" << std::endl;

    std::string seq1 = "ACE";
    std::string seq2 = "AE";

    // Alignment: A-A, C-gap, E-E
    AlignmentPair path[] = {
        {0, 0, 0.9},   // A-A
        {1, -1, 0.0},  // C-gap (gap in seq2)
        {2, 1, 0.9}    // E-E
    };

    std::string aligned1 = insert_gaps(seq1, path, 3, true);
    std::string aligned2 = insert_gaps(seq2, path, 3, false);

    assert(aligned1 == "ACE");
    assert(aligned2 == "A-E");

    std::cout << "  Seq1: " << aligned1 << " (expected: ACE)" << std::endl;
    std::cout << "  Seq2: " << aligned2 << " (expected: A-E)" << std::endl;
    std::cout << "  ✓ PASS" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 8: Insert Gaps (Gap in Seq1)
//==============================================================================

bool test_insert_gaps_seq1() {
    std::cout << "=== Test 8: Insert Gaps (Gap in Seq1) ===" << std::endl;

    std::string seq1 = "AE";
    std::string seq2 = "ACE";

    // Alignment: A-A, gap-C, E-E
    AlignmentPair path[] = {
        {0, 0, 0.9},   // A-A
        {-1, 1, 0.0},  // gap-C (gap in seq1)
        {1, 2, 0.9}    // E-E
    };

    std::string aligned1 = insert_gaps(seq1, path, 3, true);
    std::string aligned2 = insert_gaps(seq2, path, 3, false);

    assert(aligned1 == "A-E");
    assert(aligned2 == "ACE");

    std::cout << "  Seq1: " << aligned1 << " (expected: A-E)" << std::endl;
    std::cout << "  Seq2: " << aligned2 << " (expected: ACE)" << std::endl;
    std::cout << "  ✓ PASS" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Test 9: Insert Gaps (Multiple Gaps)
//==============================================================================

bool test_insert_gaps_multiple() {
    std::cout << "=== Test 9: Insert Gaps (Multiple) ===" << std::endl;

    std::string seq1 = "ACEF";
    std::string seq2 = "ADF";

    // Alignment: A-A, C-gap, E-D, gap-gap?, F-F
    AlignmentPair path[] = {
        {0, 0, 0.9},   // A-A
        {1, -1, 0.0},  // C-gap
        {2, 1, 0.7},   // E-D (mismatch but aligned)
        {3, 2, 0.9}    // F-F
    };

    std::string aligned1 = insert_gaps(seq1, path, 4, true);
    std::string aligned2 = insert_gaps(seq2, path, 4, false);

    assert(aligned1 == "ACEF");
    assert(aligned2 == "A-DF");

    std::cout << "  Seq1: " << aligned1 << " (expected: ACEF)" << std::endl;
    std::cout << "  Seq2: " << aligned2 << " (expected: A-DF)" << std::endl;
    std::cout << "  ✓ PASS" << std::endl << std::endl;
    return true;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Sequence Utils Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 9;

    if (test_three_to_one_standard()) passed++;
    if (test_three_to_one_extended()) passed++;
    if (test_three_to_one_unknown()) passed++;
    if (test_extract_sequence()) passed++;
    if (test_extract_sequence_mixed_case()) passed++;
    if (test_insert_gaps_no_gaps()) passed++;
    if (test_insert_gaps_seq2()) passed++;
    if (test_insert_gaps_seq1()) passed++;
    if (test_insert_gaps_multiple()) passed++;

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
