#!/usr/bin/env python3
"""
Validate C++ PDB/mmCIF parsers against BioPython ground truth.

This test compares residue counts from our C++ parsers with BioPython's
reference implementation to identify parsing discrepancies, particularly
with insertion codes.
"""

import sys
from pathlib import Path
import pytest

# BioPython is optional for this test
try:
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    pytest.skip("BioPython not available", allow_module_level=True)

import pfalign


def _find_repo_root():
    """Find repository root or test data directory.

    Tries multiple strategies:
    1. Look for pyproject.toml (in-repo development)
    2. Look for data/ directory in current location (CI with copied data)
    3. Look for data/ directory relative to test file (copied tests)
    """
    # Strategy 1: Find pyproject.toml (development/in-tree)
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Strategy 2: Check if data/ exists alongside tests (CI pattern)
    test_dir = Path(__file__).resolve().parent
    if (test_dir / "data").exists():
        return test_dir.parent  # Return parent so tests/data works

    # Strategy 3: Assume we're in copied tests, data should be alongside
    # This handles the /tmp/pfalign_tests case in CI
    return test_dir.parent


# Test PDB files from the repository (relative to repo root)
_REPO_ROOT = _find_repo_root()
TEST_PDBS = [
    str(_REPO_ROOT / "data" / "structures" / "pdb" / "families" / "globin" / "1MBO.pdb"),
    str(_REPO_ROOT / "data" / "structures" / "pdb" / "families" / "globin" / "1MBA.pdb"),
    str(_REPO_ROOT / "data" / "structures" / "pdb" / "families" / "globin" / "1HBS.pdb"),
    str(_REPO_ROOT / "data" / "structures" / "pdb" / "families" / "globin" / "2NRL.pdb"),
]

# Test mmCIF files from the repository (relative to repo root)
TEST_MMCIFS = [
    str(_REPO_ROOT / "data" / "structures" / "pdb" / "small" / "1CRN.cif"),
    str(_REPO_ROOT / "data" / "structures" / "pdb" / "small" / "1UBQ.cif"),
]


def count_biopython_residues(pdb_path, chain_idx=0):
    """
    Count residues using BioPython (reference implementation).

    Args:
        pdb_path: Path to PDB or mmCIF file
        chain_idx: Chain index (0-based)

    Returns:
        dict with keys:
        - total_with_icodes: Total residue count (each insertion code is separate)
        - unique_resi: Unique residue numbers (biological count, collapsed insertion codes)
        - has_insertion_codes: Whether any insertion codes exist
        - chain_id: Chain identifier
        - sequence: One-letter sequence
    """
    # Suppress BioPython warnings about discontinuous chains, etc.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=PDBConstructionWarning)

        # Auto-detect file format
        if pdb_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure('protein', pdb_path)

    # Get first model
    model = structure[0]

    # Get chain by index
    chains = list(model.get_chains())
    if chain_idx >= len(chains):
        raise ValueError(f"Chain index {chain_idx} out of range (file has {len(chains)} chains)")

    chain = chains[chain_idx]
    chain_id = chain.id

    # Filter to standard amino acid residues only (skip HETATM)
    # In BioPython, residue.id is (hetfield, resseq, icode)
    # hetfield == ' ' means standard residue (not HETATM)
    standard_residues = [r for r in chain if r.id[0] == ' ']

    # Count total residues (with insertion codes as separate)
    total_with_icodes = len(standard_residues)

    # Count unique residue numbers (collapse insertion codes)
    unique_resi_numbers = set(r.id[1] for r in standard_residues)
    unique_resi = len(unique_resi_numbers)

    # Check if any insertion codes exist
    has_insertion_codes = any(r.id[2] != ' ' for r in standard_residues)

    # Extract sequence (three-letter codes)
    three_letter_seq = [r.get_resname() for r in standard_residues]

    # Convert to one-letter sequence (simplified, may not handle all cases)
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'MSE': 'M',  # Selenomethionine
    }
    sequence = ''.join(aa_map.get(resn, 'X') for resn in three_letter_seq)

    return {
        'total_with_icodes': total_with_icodes,
        'unique_resi': unique_resi,
        'has_insertion_codes': has_insertion_codes,
        'chain_id': chain_id,
        'sequence': sequence,
    }


def count_cpp_residues(pdb_path, chain_idx=0):
    """
    Count residues using C++ parser via pfalign.encode().

    Args:
        pdb_path: Path to PDB file
        chain_idx: Chain index (0-based)

    Returns:
        dict with keys:
        - residue_count: Number of residues
        - embedding_length: Length of embedding (should match residue_count)
    """
    result = pfalign.encode(pdb_path, chain=chain_idx, k_neighbors=30)

    # Get embedding shape
    embeddings = result.embeddings  # Should be [L, D]

    return {
        'residue_count': embeddings.shape[0],
        'embedding_length': embeddings.shape[0],
    }


@pytest.mark.skipif(not BIOPYTHON_AVAILABLE, reason="BioPython not installed")
@pytest.mark.parametrize("pdb_path", TEST_PDBS)
def test_parser_vs_biopython(pdb_path):
    """Compare C++ parser against BioPython for each test PDB."""
    if not Path(pdb_path).exists():
        pytest.skip(f"Test PDB not found: {pdb_path}")

    print(f"\n{'='*70}")
    print(f"Validating: {Path(pdb_path).name}")
    print(f"{'='*70}")

    # Get BioPython counts (ground truth)
    bio_counts = count_biopython_residues(pdb_path, chain_idx=0)

    # Get C++ parser counts
    cpp_counts = count_cpp_residues(pdb_path, chain_idx=0)

    # Print comparison
    print(f"Chain ID: {bio_counts['chain_id']}")
    print(f"\nBioPython (with insertion codes):  {bio_counts['total_with_icodes']}")
    print(f"BioPython (unique residue #s):     {bio_counts['unique_resi']}")
    print(f"C++ Parser:                        {cpp_counts['residue_count']}")
    print(f"\nHas insertion codes:               {bio_counts['has_insertion_codes']}")
    print(f"Sequence (first 50):               {bio_counts['sequence'][:50]}")

    # Determine expected behavior
    if cpp_counts['residue_count'] == bio_counts['total_with_icodes']:
        print("\n✓ C++ matches BioPython (treating insertion codes as separate residues)")
        status = "TREATS_ICODES_SEPARATE"
    elif cpp_counts['residue_count'] == bio_counts['unique_resi']:
        print("\n✓ C++ matches BioPython (collapsed insertion codes)")
        status = "COLLAPSED_ICODES"
    else:
        diff_from_total = cpp_counts['residue_count'] - bio_counts['total_with_icodes']
        diff_from_unique = cpp_counts['residue_count'] - bio_counts['unique_resi']
        print(f"\n✗ MISMATCH!")
        print(f"  Difference from total:  {diff_from_total:+d}")
        print(f"  Difference from unique: {diff_from_unique:+d}")
        status = "MISMATCH"

    print(f"{'='*70}\n")

    # For now, document the behavior (we know it treats icodes as separate)
    # After fix, change this assertion to expect COLLAPSED_ICODES
    assert status in ["TREATS_ICODES_SEPARATE", "COLLAPSED_ICODES"], \
        f"Parser behavior is inconsistent: {status}"


@pytest.mark.skipif(not BIOPYTHON_AVAILABLE, reason="BioPython not installed")
@pytest.mark.parametrize("cif_path", TEST_MMCIFS)
def test_mmcif_parser_vs_biopython(cif_path):
    """Compare C++ mmCIF parser against BioPython for each test mmCIF."""
    if not Path(cif_path).exists():
        pytest.skip(f"Test mmCIF not found: {cif_path}")

    print(f"\n{'='*70}")
    print(f"Validating mmCIF: {Path(cif_path).name}")
    print(f"{'='*70}")

    # Get BioPython counts (ground truth)
    bio_counts = count_biopython_residues(cif_path, chain_idx=0)

    # Get C++ parser counts
    cpp_counts = count_cpp_residues(cif_path, chain_idx=0)

    # Print comparison
    print(f"Chain ID: {bio_counts['chain_id']}")
    print(f"\nBioPython (with insertion codes):  {bio_counts['total_with_icodes']}")
    print(f"BioPython (unique residue #s):     {bio_counts['unique_resi']}")
    print(f"C++ Parser:                        {cpp_counts['residue_count']}")
    print(f"\nHas insertion codes:               {bio_counts['has_insertion_codes']}")
    print(f"Sequence (first 50):               {bio_counts['sequence'][:50]}")

    # Determine expected behavior
    if cpp_counts['residue_count'] == bio_counts['total_with_icodes']:
        print("\n✓ C++ matches BioPython (treating insertion codes as separate residues)")
        status = "TREATS_ICODES_SEPARATE"
    elif cpp_counts['residue_count'] == bio_counts['unique_resi']:
        print("\n✓ C++ matches BioPython (collapsed insertion codes)")
        status = "COLLAPSED_ICODES"
    else:
        diff_from_total = cpp_counts['residue_count'] - bio_counts['total_with_icodes']
        diff_from_unique = cpp_counts['residue_count'] - bio_counts['unique_resi']
        print(f"\n✗ MISMATCH!")
        print(f"  Difference from total:  {diff_from_total:+d}")
        print(f"  Difference from unique: {diff_from_unique:+d}")
        status = "MISMATCH"

    print(f"{'='*70}\n")

    # mmCIF parser should match BioPython exactly
    assert status in ["TREATS_ICODES_SEPARATE", "COLLAPSED_ICODES"], \
        f"mmCIF parser behavior is inconsistent: {status}"


@pytest.mark.skipif(not BIOPYTHON_AVAILABLE, reason="BioPython not installed")
def test_parser_validation_summary():
    """Print summary of all parser validations."""
    print("\n" + "="*70)
    print("PARSER VALIDATION SUMMARY")
    print("="*70)

    # PDB files
    pdb_results = []
    for pdb_path in TEST_PDBS:
        if not Path(pdb_path).exists():
            continue

        try:
            bio_counts = count_biopython_residues(pdb_path, chain_idx=0)
            cpp_counts = count_cpp_residues(pdb_path, chain_idx=0)

            pdb_results.append({
                'file': Path(pdb_path).name,
                'bio_total': bio_counts['total_with_icodes'],
                'bio_unique': bio_counts['unique_resi'],
                'cpp': cpp_counts['residue_count'],
                'has_icodes': bio_counts['has_insertion_codes'],
            })
        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")

    # mmCIF files
    cif_results = []
    for cif_path in TEST_MMCIFS:
        if not Path(cif_path).exists():
            continue

        try:
            bio_counts = count_biopython_residues(cif_path, chain_idx=0)
            cpp_counts = count_cpp_residues(cif_path, chain_idx=0)

            cif_results.append({
                'file': Path(cif_path).name,
                'bio_total': bio_counts['total_with_icodes'],
                'bio_unique': bio_counts['unique_resi'],
                'cpp': cpp_counts['residue_count'],
                'has_icodes': bio_counts['has_insertion_codes'],
            })
        except Exception as e:
            print(f"Error processing {cif_path}: {e}")

    # Print PDB table
    print(f"\nPDB Files:")
    print(f"{'File':<15} {'BioPython':<12} {'BioPython':<12} {'C++':<8} {'Insertion':<10}")
    print(f"{'':15} {'(w/ icodes)':<12} {'(unique)':<12} {'Parser':<8} {'Codes':<10}")
    print("-" * 70)

    for r in pdb_results:
        icode_marker = "YES" if r['has_icodes'] else "NO"
        print(f"{r['file']:<15} {r['bio_total']:<12} {r['bio_unique']:<12} "
              f"{r['cpp']:<8} {icode_marker:<10}")

    # Print mmCIF table
    print(f"\nmmCIF Files:")
    print(f"{'File':<15} {'BioPython':<12} {'BioPython':<12} {'C++':<8} {'Insertion':<10}")
    print(f"{'':15} {'(w/ icodes)':<12} {'(unique)':<12} {'Parser':<8} {'Codes':<10}")
    print("-" * 70)

    for r in cif_results:
        icode_marker = "YES" if r['has_icodes'] else "NO"
        print(f"{r['file']:<15} {r['bio_total']:<12} {r['bio_unique']:<12} "
              f"{r['cpp']:<8} {icode_marker:<10}")

    print("="*70 + "\n")


if __name__ == '__main__':
    # Run as standalone script for quick validation
    if not BIOPYTHON_AVAILABLE:
        print("ERROR: BioPython is required for this validation")
        print("Install with: pip install biopython")
        sys.exit(1)

    print("PDB/mmCIF Parser Validation Against BioPython Ground Truth")
    print("="*70 + "\n")

    print("PDB Files:")
    print("-" * 70)
    for pdb_path in TEST_PDBS:
        if Path(pdb_path).exists():
            try:
                bio_counts = count_biopython_residues(pdb_path)
                cpp_counts = count_cpp_residues(pdb_path)

                print(f"File: {Path(pdb_path).name}")
                print(f"  BioPython (with icodes): {bio_counts['total_with_icodes']}")
                print(f"  BioPython (unique resi): {bio_counts['unique_resi']}")
                print(f"  C++ Parser:              {cpp_counts['residue_count']}")
                print(f"  Has insertion codes:     {bio_counts['has_insertion_codes']}")
                print()
            except Exception as e:
                print(f"Error: {e}\n")

    print("\nmmCIF Files:")
    print("-" * 70)
    for cif_path in TEST_MMCIFS:
        if Path(cif_path).exists():
            try:
                bio_counts = count_biopython_residues(cif_path)
                cpp_counts = count_cpp_residues(cif_path)

                print(f"File: {Path(cif_path).name}")
                print(f"  BioPython (with icodes): {bio_counts['total_with_icodes']}")
                print(f"  BioPython (unique resi): {bio_counts['unique_resi']}")
                print(f"  C++ Parser:              {cpp_counts['residue_count']}")
                print(f"  Has insertion codes:     {bio_counts['has_insertion_codes']}")
                print()
            except Exception as e:
                print(f"Error: {e}\n")
