"""Quality metrics for structural alignment and MSA analysis.

This module provides metrics for evaluating alignment quality, structural
similarity, and MSA conservation.

Functions:
    # Structural Metrics
    rmsd: Root Mean Square Deviation between coordinate sets
    tm_score: TM-score for global fold similarity
    gdt: Global Distance Test (returns both GDT-TS and GDT-HA)
    gdt_ts: GDT Total Score (cutoffs: 1, 2, 4, 8 Å)
    gdt_ha: GDT High Accuracy (cutoffs: 0.5, 1, 2, 4 Å)
    lddt: Local Distance Difference Test (superposition-free)
    dali_score: DALI score and Z-score for structural alignment

    # Sequence Metrics
    identity: Sequence identity between aligned sequences
    coverage: Alignment coverage (fraction of non-gap positions)
    alignment_stats: Comprehensive alignment statistics

    # MSA Metrics
    ecs: Expected Column Score for MSA quality
    pairwise_identity: Pairwise identity matrix for MSA
    conservation: Per-column conservation scores

Example:
    >>> import pfalign.metrics
    >>>
    >>> # Structural similarity
    >>> rmsd_val = pfalign.metrics.rmsd(coords1, coords2)
    >>> print(f"RMSD: {rmsd_val:.3f} Å")
    >>>
    >>> # TM-score (fold similarity)
    >>> tm = pfalign.metrics.tm_score(coords1, coords2, len1=150, len2=145)
    >>> print(f"TM-score: {tm:.3f}")
    >>>
    >>> # GDT scores
    >>> gdt_ts, gdt_ha = pfalign.metrics.gdt(coords1, coords2)
    >>> print(f"GDT-TS: {gdt_ts:.3f}, GDT-HA: {gdt_ha:.3f}")
    >>>
    >>> # lDDT (local similarity)
    >>> lddt_val = pfalign.metrics.lddt(coords1, coords2, alignment)
    >>> print(f"lDDT: {lddt_val:.3f}")
    >>>
    >>> # Sequence identity
    >>> identity = pfalign.metrics.identity(seq1, seq2)
    >>> print(f"Identity: {identity:.2%}")
    >>>
    >>> # MSA quality
    >>> ecs_score = pfalign.metrics.ecs(msa)
    >>> conservation = pfalign.metrics.conservation(msa)
"""

from typing import Dict, Optional, Tuple
import numpy as np

from pfalign import _align_cpp

# Import structural metrics from C++ bindings (metrics submodule)
_tm_score = _align_cpp.metrics.tm_score
_gdt = _align_cpp.metrics.gdt
_gdt_ts = _align_cpp.metrics.gdt_ts
_gdt_ha = _align_cpp.metrics.gdt_ha
_lddt = _align_cpp.metrics.lddt
_dali_score = _align_cpp.metrics.dali_score


def rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    aligned: bool = True,
) -> float:
    """Compute Root Mean Square Deviation between two coordinate sets.

    RMSD measures the average distance between corresponding atoms in two
    structures. Lower values indicate better structural similarity.

    Args:
        coords1: First coordinate set (N, 3)
        coords2: Second coordinate set (N, 3)
        aligned: If False, performs Kabsch alignment first (default: True)

    Returns:
        RMSD value in Ångströms

    Raises:
        ValueError: If coordinate arrays have different shapes

    Example:
        >>> # Compute RMSD for aligned structures
        >>> rmsd_val = pfalign.metrics.rmsd(coords1, coords2, aligned=True)
        >>> print(f"RMSD: {rmsd_val:.3f} Å")
        >>>
        >>> # Compute RMSD with automatic alignment
        >>> rmsd_val = pfalign.metrics.rmsd(coords1, coords2, aligned=False)
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}"
        )

    if coords1.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {coords1.shape}")

    if not aligned:
        # Perform Kabsch alignment first
        from pfalign.structure import kabsch
        _, _, rmsd_val = kabsch(coords1, coords2)
        return rmsd_val

    # Direct RMSD calculation for aligned structures
    diff = coords1 - coords2
    rmsd_val = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return float(rmsd_val)


def identity(
    seq1: str,
    seq2: str,
    ignore_gaps: bool = True,
) -> float:
    """Compute sequence identity between two aligned sequences.

    Sequence identity is the fraction of positions with identical residues.

    Args:
        seq1: First aligned sequence (with gaps as '-')
        seq2: Second aligned sequence (with gaps as '-')
        ignore_gaps: If True, only count non-gap positions (default: True)

    Returns:
        Identity fraction (0.0 to 1.0)

    Raises:
        ValueError: If sequences have different lengths

    Example:
        >>> seq1 = "AC-DEFGHIKLM"
        >>> seq2 = "ACGDEFGHIKLM"
        >>> id_frac = pfalign.metrics.identity(seq1, seq2)
        >>> print(f"Identity: {id_frac:.2%}")  # 91.67%
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Aligned sequences must have same length: {len(seq1)} vs {len(seq2)}"
        )

    if len(seq1) == 0:
        return 0.0

    matches = 0
    total = 0

    for c1, c2 in zip(seq1, seq2):
        if ignore_gaps and (c1 == '-' or c2 == '-'):
            continue

        total += 1
        if c1 == c2:
            matches += 1

    if total == 0:
        return 0.0

    return matches / total


def coverage(
    seq1: str,
    seq2: str,
) -> float:
    """Compute alignment coverage (fraction of positions without gaps).

    Coverage measures how much of the alignment is non-gap positions.
    Higher coverage indicates fewer gaps.

    Args:
        seq1: First aligned sequence
        seq2: Second aligned sequence

    Returns:
        Coverage fraction (0.0 to 1.0)

    Raises:
        ValueError: If sequences have different lengths

    Example:
        >>> seq1 = "AC-DEFGHIKLM"
        >>> seq2 = "ACGDEFGHIKLM"
        >>> cov = pfalign.metrics.coverage(seq1, seq2)
        >>> print(f"Coverage: {cov:.2%}")  # 91.67%
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Aligned sequences must have same length: {len(seq1)} vs {len(seq2)}"
        )

    if len(seq1) == 0:
        return 0.0

    non_gap = sum(1 for c1, c2 in zip(seq1, seq2) if c1 != '-' and c2 != '-')
    return non_gap / len(seq1)


def alignment_stats(
    seq1: str,
    seq2: str,
) -> Dict[str, float]:
    """Compute comprehensive alignment statistics.

    Computes multiple alignment quality metrics in one pass.

    Args:
        seq1: First aligned sequence
        seq2: Second aligned sequence

    Returns:
        Dictionary with statistics:
            - identity: Sequence identity (0-1)
            - coverage: Alignment coverage (0-1)
            - length: Alignment length
            - gaps: Total number of gap characters
            - gap_percentage: Percentage of positions with gaps (0-100)
            - matches: Number of matching positions
            - mismatches: Number of mismatched positions

    Example:
        >>> stats = pfalign.metrics.alignment_stats(seq1, seq2)
        >>> print(f"Identity: {stats['identity']:.2%}")
        >>> print(f"Coverage: {stats['coverage']:.2%}")
        >>> print(f"Gaps: {stats['gap_percentage']:.1f}%")
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Aligned sequences must have same length: {len(seq1)} vs {len(seq2)}"
        )

    length = len(seq1)

    if length == 0:
        return {
            'identity': 0.0,
            'coverage': 0.0,
            'length': 0,
            'gaps': 0,
            'gap_percentage': 0.0,
            'matches': 0,
            'mismatches': 0,
        }

    gaps = sum(1 for c1, c2 in zip(seq1, seq2) if c1 == '-' or c2 == '-')
    matches = sum(1 for c1, c2 in zip(seq1, seq2)
                  if c1 != '-' and c2 != '-' and c1 == c2)
    non_gap_positions = length - gaps
    mismatches = non_gap_positions - matches

    return {
        'identity': matches / non_gap_positions if non_gap_positions > 0 else 0.0,
        'coverage': non_gap_positions / length,
        'length': length,
        'gaps': gaps,
        'gap_percentage': (gaps / length) * 100,
        'matches': matches,
        'mismatches': mismatches,
    }


def ecs(
    msa: 'MSAResult',  # type: ignore
    temperature: float = 5.0,
) -> float:
    """Compute Expected Column Score for MSA quality.

    ECS measures how well-conserved the columns in an MSA are. Higher
    scores indicate better-quality alignments with more conservation.

    Args:
        msa: MSA result object
        temperature: Temperature parameter for weighting (default: 5.0)

    Returns:
        ECS score (typically 0-1, higher is better)

    Example:
        >>> msa = pfalign.msa(structures)
        >>> ecs_score = pfalign.metrics.ecs(msa)
        >>> print(f"ECS: {ecs_score:.5f}")
    """
    # If the MSA already has ECS computed, return it
    if hasattr(msa, 'ecs_score'):
        return msa.ecs_score

    # Otherwise, we'd need to recompute it (would require C++ binding)
    # For now, raise an error
    raise NotImplementedError(
        "ECS computation from MSA not yet implemented. "
        "Use the ecs_score attribute of MSAResult instead."
    )


def pairwise_identity(
    msa: 'MSAResult',  # type: ignore
) -> np.ndarray:
    """Compute pairwise identity matrix for all sequences in MSA.

    Computes sequence identity between every pair of sequences in the MSA.

    Args:
        msa: MSA result object

    Returns:
        Symmetric identity matrix (N, N) where N is number of sequences

    Example:
        >>> msa = pfalign.msa(structures)
        >>> id_matrix = pfalign.metrics.pairwise_identity(msa)
        >>> print(f"Shape: {id_matrix.shape}")
        >>> print(f"Mean identity: {id_matrix.mean():.2%}")
    """
    # Get aligned sequences from MSA
    sequences = msa.sequences()
    n = len(sequences)

    # Compute pairwise identities
    matrix = np.zeros((n, n), dtype=np.float32)

    # Create progress bar for O(N^2) computation
    total_pairs = n * (n + 1) // 2  # Include diagonal
    progress_bar = _align_cpp.ProgressBar(total_pairs, "Computing pairwise identities", width=20)

    pair_count = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                id_val = identity(sequences[i], sequences[j], ignore_gaps=True)
                matrix[i, j] = id_val
                matrix[j, i] = id_val

            pair_count += 1
            progress_bar.update(pair_count)

    progress_bar.finish()
    return matrix


def conservation(
    msa: 'MSAResult',  # type: ignore
    method: str = 'shannon',
) -> np.ndarray:
    """Compute per-column conservation scores for MSA.

    Measures how conserved each column in the MSA is. Higher scores
    indicate more conserved (less variable) positions.

    Args:
        msa: MSA result object
        method: Conservation method - 'shannon' (entropy-based) or
                'gap_penalty' (fraction non-gap) (default: 'shannon')

    Returns:
        Conservation score per column (L,) where L is alignment length.
        Values are normalized to [0, 1] where 1 = most conserved.

    Raises:
        ValueError: If method is not recognized

    Example:
        >>> msa = pfalign.msa(structures)
        >>> cons = pfalign.metrics.conservation(msa, method='shannon')
        >>> print(f"Mean conservation: {cons.mean():.3f}")
        >>> print(f"Most conserved position: {cons.argmax()}")
    """
    sequences = msa.sequences()

    if len(sequences) == 0:
        return np.array([])

    aln_len = len(sequences[0])
    conservation_scores = np.zeros(aln_len, dtype=np.float32)

    # Create progress bar for column-by-column computation
    progress_bar = _align_cpp.ProgressBar(aln_len, "Computing conservation scores", width=20)

    if method == 'shannon':
        # Shannon entropy-based conservation
        for col_idx in range(aln_len):
            # Get residues at this column
            column = [seq[col_idx] for seq in sequences]

            # Count residue frequencies
            counts = {}
            for residue in column:
                if residue != '-':  # Ignore gaps
                    counts[residue] = counts.get(residue, 0) + 1

            if len(counts) == 0:
                # All gaps
                conservation_scores[col_idx] = 0.0
            else:
                # Compute Shannon entropy
                total = sum(counts.values())
                entropy = 0.0
                for count in counts.values():
                    p = count / total
                    entropy -= p * np.log2(p)

                # Normalize by max entropy (log2(20) for amino acids)
                max_entropy = np.log2(min(20, len(counts)))
                if max_entropy > 0:
                    conservation_scores[col_idx] = 1.0 - (entropy / max_entropy)
                else:
                    conservation_scores[col_idx] = 1.0

            progress_bar.update(col_idx + 1)

    elif method == 'gap_penalty':
        # Simple conservation: fraction of non-gap residues
        for col_idx in range(aln_len):
            column = [seq[col_idx] for seq in sequences]
            non_gap = sum(1 for res in column if res != '-')
            conservation_scores[col_idx] = non_gap / len(sequences)
            progress_bar.update(col_idx + 1)

    else:
        raise ValueError(
            f"Unknown conservation method: {method}. "
            f"Choose 'shannon' or 'gap_penalty'"
        )

    progress_bar.finish()
    return conservation_scores


def tm_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    len1: int,
    len2: int,
    aligned: bool = True,
) -> float:
    """Compute TM-score (Template Modeling score) between two structures.

    TM-score measures global fold similarity. Unlike RMSD, it's length-normalized
    and less sensitive to local structural deviations.

    Args:
        coords1: First structure coordinates (N, 3)
        coords2: Second structure coordinates (N, 3)
        len1: Full sequence length of structure 1 (for normalization)
        len2: Full sequence length of structure 2 (for normalization)
        aligned: If False, performs Kabsch alignment first (default: True)

    Returns:
        TM-score value in [0, 1] where 1 is perfect
        - > 0.5: Same fold (high confidence)
        - 0.3-0.5: Possible homology
        - < 0.3: Different fold

    Raises:
        ValueError: If coordinate arrays have incompatible shapes

    Example:
        >>> tm = pfalign.metrics.tm_score(coords1, coords2, len1=150, len2=145)
        >>> print(f"TM-score: {tm:.3f}")
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}"
        )
    if coords1.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {coords1.shape}")

    return float(_tm_score(coords1, coords2, len1, len2, aligned))


def gdt(
    coords1: np.ndarray,
    coords2: np.ndarray,
    aligned: bool = True,
) -> Tuple[float, float]:
    """Compute GDT-TS and GDT-HA scores.

    GDT (Global Distance Test) measures the percentage of residues aligned
    within distance cutoffs.

    Args:
        coords1: First structure coordinates (N, 3)
        coords2: Second structure coordinates (N, 3)
        aligned: If False, performs Kabsch alignment first (default: True)

    Returns:
        Tuple of (gdt_ts, gdt_ha) scores in [0, 1]
        - GDT-TS: Cutoffs [1, 2, 4, 8] Å (standard)
        - GDT-HA: Cutoffs [0.5, 1, 2, 4] Å (high accuracy)
        - > 0.7: High quality
        - 0.5-0.7: Good quality
        - < 0.5: Poor quality

    Raises:
        ValueError: If coordinate arrays have incompatible shapes

    Example:
        >>> gdt_ts, gdt_ha = pfalign.metrics.gdt(coords1, coords2)
        >>> print(f"GDT-TS: {gdt_ts:.3f}, GDT-HA: {gdt_ha:.3f}")
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}"
        )
    if coords1.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {coords1.shape}")

    return _gdt(coords1, coords2, aligned)


def gdt_ts(
    coords1: np.ndarray,
    coords2: np.ndarray,
    aligned: bool = True,
) -> float:
    """Compute GDT-TS (Total Score) only.

    See gdt() for details. GDT-TS uses distance cutoffs [1, 2, 4, 8] Å.

    Args:
        coords1: First structure coordinates (N, 3)
        coords2: Second structure coordinates (N, 3)
        aligned: If False, performs Kabsch alignment first (default: True)

    Returns:
        GDT-TS score in [0, 1]

    Example:
        >>> gdt_ts_val = pfalign.metrics.gdt_ts(coords1, coords2)
        >>> print(f"GDT-TS: {gdt_ts_val:.3f}")
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}"
        )
    if coords1.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {coords1.shape}")

    return float(_gdt_ts(coords1, coords2, aligned))


def gdt_ha(
    coords1: np.ndarray,
    coords2: np.ndarray,
    aligned: bool = True,
) -> float:
    """Compute GDT-HA (High Accuracy) only.

    See gdt() for details. GDT-HA uses distance cutoffs [0.5, 1, 2, 4] Å.

    Args:
        coords1: First structure coordinates (N, 3)
        coords2: Second structure coordinates (N, 3)
        aligned: If False, performs Kabsch alignment first (default: True)

    Returns:
        GDT-HA score in [0, 1]

    Example:
        >>> gdt_ha_val = pfalign.metrics.gdt_ha(coords1, coords2)
        >>> print(f"GDT-HA: {gdt_ha_val:.3f}")
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shapes must match: {coords1.shape} vs {coords2.shape}"
        )
    if coords1.shape[1] != 3:
        raise ValueError(f"Coordinates must be (N, 3), got {coords1.shape}")

    return float(_gdt_ha(coords1, coords2, aligned))


def lddt(
    coords1: np.ndarray,
    coords2: np.ndarray,
    alignment: np.ndarray,
    radius: float = 15.0,
) -> float:
    """Compute lDDT (Local Distance Difference Test) score.

    lDDT is a superposition-free metric that measures local structural similarity
    by comparing distances between nearby residues. More sensitive to local geometry
    than global metrics like RMSD.

    Args:
        coords1: Coordinates for structure 1 (L1, 3)
        coords2: Coordinates for structure 2 (L2, 3)
        alignment: Aligned position pairs (M, 2) where alignment[i] = [pos1, pos2]
        radius: Inclusion radius in Ångströms (default: 15.0)

    Returns:
        lDDT score in [0, 1] where 1 is perfect
        - > 0.8: High quality model
        - 0.6-0.8: Good model
        - < 0.6: Poor model

    Raises:
        ValueError: If arrays have incorrect shapes

    Example:
        >>> alignment = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        >>> lddt_val = pfalign.metrics.lddt(coords1, coords2, alignment, radius=15.0)
        >>> print(f"lDDT: {lddt_val:.3f}")
    """
    if coords1.ndim != 2 or coords1.shape[1] != 3:
        raise ValueError(f"coords1 must be (N, 3), got {coords1.shape}")
    if coords2.ndim != 2 or coords2.shape[1] != 3:
        raise ValueError(f"coords2 must be (N, 3), got {coords2.shape}")
    if alignment.ndim != 2 or alignment.shape[1] != 2:
        raise ValueError(f"alignment must be (M, 2), got {alignment.shape}")

    # Ensure alignment is int32
    alignment = alignment.astype(np.int32, copy=False)

    return float(_lddt(coords1, coords2, alignment, radius))


def dali_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    alignment: np.ndarray,
    len1: int,
    len2: int,
    horizon: float = 20.0,
) -> Tuple[float, float]:
    """Compute DALI (Distance Alignment) score and Z-score.

    DALI measures structural similarity by comparing internal distance matrices.
    The Z-score is length-normalized for statistical significance.

    Args:
        coords1: Coordinates for structure 1 (L1, 3)
        coords2: Coordinates for structure 2 (L2, 3)
        alignment: Aligned position pairs (M, 2) where alignment[i] = [pos1, pos2]
        len1: Full length of structure 1 (for Z-score normalization)
        len2: Full length of structure 2 (for Z-score normalization)
        horizon: Distance weighting decay parameter in Ångströms (default: 20.0)

    Returns:
        Tuple of (dali_score, z_score)
        - Z > 2: Likely structural similarity
        - Z > 5: High confidence homology
        - Z > 20: Highly significant similarity

    Raises:
        ValueError: If arrays have incorrect shapes

    Example:
        >>> alignment = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
        >>> score, z = pfalign.metrics.dali_score(coords1, coords2, alignment, 150, 145)
        >>> print(f"DALI: {score:.2f} (Z={z:.2f})")
    """
    if coords1.ndim != 2 or coords1.shape[1] != 3:
        raise ValueError(f"coords1 must be (N, 3), got {coords1.shape}")
    if coords2.ndim != 2 or coords2.shape[1] != 3:
        raise ValueError(f"coords2 must be (N, 3), got {coords2.shape}")
    if alignment.ndim != 2 or alignment.shape[1] != 2:
        raise ValueError(f"alignment must be (M, 2), got {alignment.shape}")

    # Ensure alignment is int32
    alignment = alignment.astype(np.int32, copy=False)

    return _dali_score(coords1, coords2, alignment, len1, len2, horizon)


__all__ = [
    'rmsd',
    'identity',
    'coverage',
    'alignment_stats',
    'ecs',
    'pairwise_identity',
    'conservation',
    'tm_score',
    'gdt',
    'gdt_ts',
    'gdt_ha',
    'lddt',
    'dali_score',
]
