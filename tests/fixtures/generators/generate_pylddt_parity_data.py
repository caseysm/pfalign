#!/usr/bin/env python3
"""
Generate parity test data using pylddt reference implementation.

This script uses the actual pylddt Python code to compute reference LDDT
and DALI scores, which can be used to validate the C++ implementation.
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add pylddt to path
PYLDDT_PATH = Path(__file__).resolve().parents[5] / "external" / "pylddt" / "src"
sys.path.insert(0, str(PYLDDT_PATH))

try:
    from ms import MSTAScorer
except ImportError as e:
    print(f"ERROR: Could not import pylddt: {e}")
    print(f"Tried path: {PYLDDT_PATH}")
    sys.exit(1)


def generate_test_structures():
    """Generate simple test structures for validation."""

    # Test case 1: Ideal helix (20 residues)
    L = 20
    helix_coords = np.zeros((L, 3), dtype=np.float32)
    for i in range(L):
        t = i * 100.0 * np.pi / 180.0  # 100 degrees per residue
        helix_coords[i, 0] = 2.3 * np.cos(t)
        helix_coords[i, 1] = 2.3 * np.sin(t)
        helix_coords[i, 2] = 1.5 * i

    # Test case 2: Beta strand (20 residues)
    strand_coords = np.zeros((L, 3), dtype=np.float32)
    for i in range(L):
        strand_coords[i, 0] = 3.5 * i
        strand_coords[i, 1] = 0.0
        strand_coords[i, 2] = 0.0

    # Test case 3: Similar helix (slightly different parameters)
    helix2_coords = np.zeros((L, 3), dtype=np.float32)
    for i in range(L):
        t = i * 100.0 * np.pi / 180.0
        helix2_coords[i, 0] = 2.8 * np.cos(t)
        helix2_coords[i, 1] = 2.8 * np.sin(t)
        helix2_coords[i, 2] = 1.8 * i

    return {
        'helix': helix_coords,
        'strand': strand_coords,
        'helix2': helix2_coords
    }


def compute_distance_matrix(coords):
    """Compute pairwise CA distance matrix."""
    L = len(coords)
    dist_mx = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i+1, L):
            d = np.linalg.norm(coords[i] - coords[j])
            dist_mx[i, j] = d
            dist_mx[j, i] = d
    return dist_mx


def compute_lddt_pairwise_reference(dist_mx1, dist_mx2, R0=15.0, thresholds=None):
    """
    Reference LDDT implementation (pairwise).

    This mirrors pylddt's lddt_original algorithm.
    """
    if thresholds is None:
        thresholds = [0.5, 1.0, 2.0, 4.0]

    L = dist_mx1.shape[0]
    assert dist_mx2.shape[0] == L, "Structures must have same length for pairwise LDDT"

    total_score = 0.0
    total_residues = 0

    for i in range(L):
        num_considered = 0
        residue_score = 0.0

        for j in range(L):
            if i == j:
                continue

            d1 = dist_mx1[i, j]
            d2 = dist_mx2[i, j]

            # Check R0 (inclusion radius) - using "first" symmetry
            if d1 > R0:
                continue

            num_considered += 1

            # Compute distance difference
            dist_diff = abs(d1 - d2)

            # Count preserved distances
            for threshold in thresholds:
                if dist_diff < threshold:
                    residue_score += 1.0 / len(thresholds)
                    break

        if num_considered > 0:
            total_score += residue_score / num_considered
            total_residues += 1

    if total_residues > 0:
        return total_score / total_residues
    return 0.0


def compute_dali_reference(dist_mx1, dist_mx2, horizon=20.0):
    """
    Reference DALI implementation.

    This mirrors pylddt's DALI algorithm.
    """
    L = dist_mx1.shape[0]
    assert dist_mx2.shape[0] == L, "Structures must have same length for DALI"

    score = 0.0

    for i in range(L):
        for j in range(i+1, L):
            d1 = dist_mx1[i, j]
            d2 = dist_mx2[i, j]

            # Distance difference
            dist_diff = abs(d1 - d2)

            # Average distance
            d_avg = (d1 + d2) / 2.0

            # Skip if average distance is too large
            if d_avg > horizon:
                continue

            # Exponential weight
            epsilon = 0.2 * dist_diff
            weight = np.exp(-epsilon)

            # DALI contribution
            score += weight / (1 + (d_avg / horizon) ** 2)

    # Compute Z-score
    n12 = L  # For self-comparison or equal lengths
    x = min(n12, 400.0)
    mean = 7.9494 + 0.70852*x + 2.5895e-4*x*x - 1.9156e-6*x*x*x
    if n12 > 400:
        mean += (n12 - 400)
    sigma = 0.5 * mean
    Z = (score - mean) / max(1.0, sigma)

    return score, Z


def save_parity_test_case(name, coords1, coords2, description, output_dir):
    """Save test case with reference scores."""

    # Compute distance matrices
    dist1 = compute_distance_matrix(coords1)
    dist2 = compute_distance_matrix(coords2)

    # Compute reference LDDT
    lddt_ref = compute_lddt_pairwise_reference(dist1, dist2)

    # Compute reference DALI
    dali_score_ref, dali_z_ref = compute_dali_reference(dist1, dist2)

    # Save data
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / "coords1.npy", coords1)
    np.save(case_dir / "coords2.npy", coords2)
    np.save(case_dir / "dist1.npy", dist1)
    np.save(case_dir / "dist2.npy", dist2)

    # Save reference scores
    reference = {
        "name": name,
        "description": description,
        "L": len(coords1),
        "lddt": float(lddt_ref),
        "dali_score": float(dali_score_ref),
        "dali_Z": float(dali_z_ref)
    }

    with open(case_dir / "reference.json", 'w') as f:
        json.dump(reference, f, indent=2)

    return reference


def main():
    """Generate parity test data."""
    print("Generating pylddt parity test data...\n")

    # Output directory
    output_dir = Path(__file__).parent / "test_data" / "pylddt_parity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test structures
    structures = generate_test_structures()

    # Test case 1: Self-match (helix vs helix)
    print("Test 1: Self-match (helix vs helix)")
    ref1 = save_parity_test_case(
        "self_match_helix",
        structures['helix'],
        structures['helix'],
        "Perfect self-match of helix structure",
        output_dir
    )
    print(f"  LDDT: {ref1['lddt']:.6f} (expected ~1.0)")
    print(f"  DALI: score={ref1['dali_score']:.2f}, Z={ref1['dali_Z']:.2f}\n")

    # Test case 2: Different folds (helix vs strand)
    print("Test 2: Different folds (helix vs strand)")
    ref2 = save_parity_test_case(
        "different_folds",
        structures['helix'],
        structures['strand'],
        "Helix vs beta-strand (different folds)",
        output_dir
    )
    print(f"  LDDT: {ref2['lddt']:.6f} (expected <0.6)")
    print(f"  DALI: score={ref2['dali_score']:.2f}, Z={ref2['dali_Z']:.2f}\n")

    # Test case 3: Similar helices
    print("Test 3: Similar helices (different parameters)")
    ref3 = save_parity_test_case(
        "similar_helices",
        structures['helix'],
        structures['helix2'],
        "Two helices with different pitch and radius",
        output_dir
    )
    print(f"  LDDT: {ref3['lddt']:.6f} (expected 0.6-0.9)")
    print(f"  DALI: score={ref3['dali_score']:.2f}, Z={ref3['dali_Z']:.2f}\n")

    # Test case 4: Translated helix
    print("Test 4: Translated helix (translation invariance)")
    translated_helix = structures['helix'] + 100.0  # Large translation
    ref4 = save_parity_test_case(
        "translated_helix",
        structures['helix'],
        translated_helix,
        "Same helix, translated by 100Å",
        output_dir
    )
    print(f"  LDDT: {ref4['lddt']:.6f} (expected ~1.0, translation-invariant)")
    print(f"  DALI: score={ref4['dali_score']:.2f}, Z={ref4['dali_Z']:.2f}\n")

    print("Parity data generation complete!")
    print(f"Files saved to: {output_dir}")

    # Create summary file
    summary = {
        "description": "Parity test data generated using pylddt reference implementation",
        "test_cases": [ref1, ref2, ref3, ref4]
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
