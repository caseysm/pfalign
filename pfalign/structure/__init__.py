"""Structural superposition and coordinate manipulation.

This module provides tools for structural alignment, superposition, and
coordinate transformations using the Kabsch algorithm.

Functions:
    kabsch: Kabsch algorithm for optimal rotation/translation
    transform: Apply rotation and translation to coordinates
    get_coords: Extract coordinates from structure files
    superpose: High-level structural superposition
    superpose_from_alignment: Superpose using sequence alignment

Classes:
    SuperposeResult: Result container for superposition operations

Example:
    >>> import pfalign.structure
    >>>
    >>> # Extract coordinates
    >>> coords1 = pfalign.structure.get_coords("protein1.pdb")
    >>> coords2 = pfalign.structure.get_coords("protein2.pdb")
    >>>
    >>> # Kabsch alignment
    >>> R, t, rmsd = pfalign.structure.kabsch(coords1, coords2)
    >>> coords1_aligned = pfalign.structure.transform(coords1, R, t)
    >>>
    >>> # High-level superposition
    >>> result = pfalign.structure.superpose("protein1.pdb", "protein2.pdb")
    >>> print(f"RMSD: {result.rmsd:.3f} Å")
    >>> result.save_transformed("protein1_aligned.pdb")
"""

from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
from pfalign._align_cpp import structure as _structure_cpp


# Re-export C++ primitive functions
kabsch = _structure_cpp.kabsch
transform = _structure_cpp.transform
get_coords = _structure_cpp.get_coords


class SuperposeResult:
    """Result of structural superposition.

    Contains the transformation (rotation + translation), RMSD, and
    coordinates before/after alignment.

    Attributes:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
        rmsd: Root mean square deviation (Ångströms)
        coords1: Original coordinates of structure 1 (N, 3)
        coords2: Original coordinates of structure 2 (N, 3)
        coords1_transformed: Structure 1 after applying transformation (N, 3)

    Methods:
        save_transformed: Save transformed structure to PDB file
    """

    def __init__(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        rmsd: float,
        coords1: np.ndarray,
        coords2: np.ndarray,
        coords1_transformed: Optional[np.ndarray] = None,
        structure1_path: Optional[str] = None,
        structure2_path: Optional[str] = None,
        chain1: Union[int, str] = 0,
        chain2: Union[int, str] = 0,
    ):
        """Create SuperposeResult.

        Args:
            rotation: 3x3 rotation matrix
            translation: 3D translation vector
            rmsd: RMSD value
            coords1: Original coords of structure 1
            coords2: Original coords of structure 2
            coords1_transformed: Transformed coords (computed if None)
            structure1_path: Path to structure 1 file
            structure2_path: Path to structure 2 file
            chain1: Chain index/ID for structure 1
            chain2: Chain index/ID for structure 2
        """
        self.rotation = rotation
        self.translation = translation
        self.rmsd = rmsd
        self.coords1 = coords1
        self.coords2 = coords2

        # Compute transformed coords if not provided
        if coords1_transformed is None:
            self.coords1_transformed = transform(coords1, rotation, translation)
        else:
            self.coords1_transformed = coords1_transformed

        # Store paths for potential file operations
        self._structure1_path = structure1_path
        self._structure2_path = structure2_path
        self._chain1 = chain1
        self._chain2 = chain2

    def save_transformed(self, output_path: str) -> None:
        """Save transformed structure to PDB file.

        Note: This is a placeholder. Full PDB writing would require
        loading the original structure and updating coordinates.

        Args:
            output_path: Path for output PDB file

        Raises:
            NotImplementedError: PDB writing not yet implemented
        """
        raise NotImplementedError(
            "PDB writing not yet implemented. "
            "Use coords1_transformed attribute to access aligned coordinates."
        )

    def __repr__(self) -> str:
        return f"<SuperposeResult: {len(self.coords1)} atoms, RMSD={self.rmsd:.3f} Å>"


def superpose(
    structure1: Union[str, Path],
    structure2: Union[str, Path],
    chain1: Union[int, str] = 0,
    chain2: Union[int, str] = 0,
    atom_type: str = "CA",
) -> SuperposeResult:
    """Superpose two protein structures.

    Performs optimal structural alignment using the Kabsch algorithm on
    Calpha atoms (or backbone atoms). Returns transformation and RMSD.

    Args:
        structure1: Path to first structure (PDB/CIF)
        structure2: Path to second structure (PDB/CIF)
        chain1: Chain index (int) or ID (str) for structure1 (default: 0)
        chain2: Chain index (int) or ID (str) for structure2 (default: 0)
        atom_type: 'CA' for Calpha only or 'backbone' for N,CA,C,O (default: 'CA')

    Returns:
        SuperposeResult with transformation, RMSD, and coordinates

    Raises:
        ValueError: If structures have different number of atoms
        RuntimeError: If structure loading or alignment fails

    Example:
        >>> result = pfalign.structure.superpose("protein1.pdb", "protein2.pdb")
        >>> print(f"RMSD: {result.rmsd:.3f} Å")
        >>> print(f"Rotation matrix:\\n{result.rotation}")
        >>>
        >>> # Access aligned coordinates
        >>> aligned_coords = result.coords1_transformed
    """
    # Extract coordinates from both structures
    coords1 = get_coords(str(structure1), chain=chain1, atom_type=atom_type)
    coords2 = get_coords(str(structure2), chain=chain2, atom_type=atom_type)

    # Validate same number of atoms
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError(
            f"Structures have different number of {atom_type} atoms: "
            f"{coords1.shape[0]} vs {coords2.shape[0]}. "
            f"Use superpose_from_alignment() for structures with different lengths."
        )

    # Run Kabsch algorithm
    R, t, rmsd = kabsch(coords1, coords2)

    # Transform coords1 to align with coords2
    coords1_transformed = transform(coords1, R, t)

    return SuperposeResult(
        rotation=R,
        translation=t,
        rmsd=rmsd,
        coords1=coords1,
        coords2=coords2,
        coords1_transformed=coords1_transformed,
        structure1_path=str(structure1),
        structure2_path=str(structure2),
        chain1=chain1,
        chain2=chain2,
    )


def superpose_from_alignment(
    structure1: Union[str, Path],
    structure2: Union[str, Path],
    alignment,  # PairwiseResult or alignment path list
    chain1: Union[int, str] = 0,
    chain2: Union[int, str] = 0,
    atom_type: str = "CA",
) -> SuperposeResult:
    """Superpose structures using only aligned residues from sequence alignment.

    Uses sequence alignment to determine which residues correspond, then
    performs Kabsch alignment on only those positions. Gaps in the alignment
    are excluded from superposition.

    This is useful for:
    - Structures with different lengths
    - Structures with insertions/deletions
    - Improving superposition quality by excluding misaligned regions

    Args:
        structure1: Path to first structure (PDB/CIF)
        structure2: Path to second structure (PDB/CIF)
        alignment: PairwiseResult or list of (i, j) alignment pairs
        chain1: Chain index (int) or ID (str) for structure1 (default: 0)
        chain2: Chain index (int) or ID (str) for structure2 (default: 0)
        atom_type: 'CA' for Calpha only or 'backbone' for N,CA,C,O (default: 'CA')

    Returns:
        SuperposeResult with transformation based on aligned residues only

    Raises:
        ValueError: If no aligned pairs found
        RuntimeError: If structure loading or alignment fails

    Example:
        >>> # First, perform sequence alignment
        >>> aln = pfalign.pairwise("protein1.pdb", "protein2.pdb")
        >>>
        >>> # Then superpose using only aligned residues
        >>> result = pfalign.structure.superpose_from_alignment(
        ...     "protein1.pdb", "protein2.pdb", aln
        ... )
        >>> print(f"RMSD (aligned residues only): {result.rmsd:.3f} Å")
        >>>
        >>> # Or use alignment path directly
        >>> path = [(0, 0), (1, 1), (2, 3), (3, 4)]  # Skip position 2 in seq2
        >>> result = pfalign.structure.superpose_from_alignment(
        ...     "p1.pdb", "p2.pdb", path
        ... )
    """
    # Extract full coordinates
    coords1_full = get_coords(str(structure1), chain=chain1, atom_type=atom_type)
    coords2_full = get_coords(str(structure2), chain=chain2, atom_type=atom_type)

    # Extract alignment pairs
    if hasattr(alignment, 'alignment'):
        # PairwiseResult object
        alignment_pairs = alignment.alignment()
    else:
        # Assume it's a list of (i, j) tuples
        alignment_pairs = alignment

    # Filter aligned pairs (exclude gaps)
    aligned_pairs = [(i, j) for i, j in alignment_pairs if i >= 0 and j >= 0]

    if len(aligned_pairs) == 0:
        raise ValueError("No aligned residue pairs found (all gaps)")

    # Extract only aligned coordinates
    indices1 = [i for i, j in aligned_pairs]
    indices2 = [j for i, j in aligned_pairs]

    coords1_aligned = coords1_full[indices1]
    coords2_aligned = coords2_full[indices2]

    # Run Kabsch on aligned subset
    R, t, rmsd = kabsch(coords1_aligned, coords2_aligned)

    # Transform FULL coords1 (not just aligned subset)
    coords1_transformed = transform(coords1_full, R, t)

    return SuperposeResult(
        rotation=R,
        translation=t,
        rmsd=rmsd,
        coords1=coords1_full,
        coords2=coords2_full,
        coords1_transformed=coords1_transformed,
        structure1_path=str(structure1),
        structure2_path=str(structure2),
        chain1=chain1,
        chain2=chain2,
    )


__all__ = [
    # Primitives (C++ functions)
    'kabsch',
    'transform',
    'get_coords',
    # High-level functions
    'superpose',
    'superpose_from_alignment',
    # Result class
    'SuperposeResult',
]
