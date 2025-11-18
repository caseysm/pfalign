"""Shared utilities for pfalign Python tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def validate_result(
    result: Dict,
    *,
    expected_shape: Tuple[int, int] | None = None,
    require_posteriors: bool = True,
) -> None:
    """Validate the structure of a pairwise alignment result dictionary."""
    assert isinstance(result, dict), "Result must be a dictionary"

    for key in ("partition", "score", "posteriors", "alignment_path"):
        assert key in result, f"Missing key '{key}' in result"

    assert np.isfinite(result["partition"]), "Partition must be finite"
    assert np.isfinite(result["score"]), "Score must be finite"

    if require_posteriors:
        posteriors = result["posteriors"]
        assert isinstance(posteriors, np.ndarray), "Posteriors must be numpy array"
        assert posteriors.dtype == np.float32, "Posteriors must be float32"
        assert np.all(posteriors >= 0.0), "Posteriors contain negative probabilities"
        if expected_shape is not None:
            assert (
                posteriors.shape == expected_shape
            ), f"Expected posteriors shape {expected_shape}, got {posteriors.shape}"

    path = result["alignment_path"]
    assert isinstance(path, list), "alignment_path must be a list"
    for entry in path:
        assert (
            isinstance(entry, tuple) and len(entry) == 2
        ), "alignment_path entries must be (i, j) tuples"


def validate_fasta_output(
    fasta_path: Path,
    *,
    expected_ids: Iterable[str],
) -> None:
    """Validate that a FASTA file contains the expected headers and content."""
    assert fasta_path.exists(), f"FASTA file not found: {fasta_path}"

    data = fasta_path.read_text().strip().splitlines()
    assert len(data) >= 4, "FASTA output should contain two sequences"

    headers = [line for line in data if line.startswith(">")]
    expected_list = list(expected_ids)
    assert len(headers) == len(expected_list), "Unexpected number of FASTA headers"

    for header, expected_id in zip(headers, expected_list):
        assert (
            expected_id in header
        ), f"Expected identifier '{expected_id}' in FASTA header '{header}'"

    # Validate sequence content (letters and gaps only)
    sequences = [line for line in data if not line.startswith(">")]
    for seq in sequences:
        assert seq, "Empty FASTA sequence line"
        allowed = set("ACDEFGHIKLMNPQRSTVWYXBZ-")
        assert set(seq.upper()) <= allowed, f"Invalid characters found in FASTA: {seq}"


# NOTE: Removed _load_structure_cached(), _detect_parser_kind(), and load_pdb_with_sequence()
# These functions attempted to access internal PyBind11 parser classes (MMCIFParser, PDBParser)
# that are not exported from the C++ bindings. Tests should use the public pfalign.encode() API
# instead, which internally handles all parsing and returns embeddings.
