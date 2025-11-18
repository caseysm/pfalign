"""Pytest configuration and shared fixtures for protein_forge_align tests."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


def _find_repo_root():
    """Find repository root or test data directory.

    Tries multiple strategies:
    1. Look for pyproject.toml (in-repo development)
    2. Look for data/ directory in current location (CI with copied data)
    3. Look for data/ directory relative to conftest.py (copied tests)
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


@pytest.fixture
def test_coords_small():
    """Small synthetic coordinate arrays for testing.

    Returns two coordinate arrays of shape (5, 14, 3) and (6, 14, 3).
    """
    np.random.seed(42)

    coords1 = np.random.randn(5, 14, 3).astype(np.float32)
    coords2 = np.random.randn(6, 14, 3).astype(np.float32)

    return coords1, coords2


@pytest.fixture
def test_coords_medium():
    """Medium synthetic coordinate arrays for testing.

    Returns two coordinate arrays of shape (20, 14, 3) and (25, 14, 3).
    """
    np.random.seed(123)

    coords1 = np.random.randn(20, 14, 3).astype(np.float32)
    coords2 = np.random.randn(25, 14, 3).astype(np.float32)

    return coords1, coords2


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs.

    Yields a temporary directory path that is automatically cleaned up.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdbs():
    """Return two small PDB files for integration tests."""
    # Try multiple locations for test data
    test_dir = Path(__file__).resolve().parent
    possible_locations = [
        # In-tree development
        _find_repo_root() / "tests" / "data" / "integration" / "msa_families" / "lysozyme",
        # CI with copied data
        test_dir / "data" / "integration" / "msa_families" / "lysozyme",
        # Alternative CI structure
        test_dir.parent / "tests" / "data" / "integration" / "msa_families" / "lysozyme",
    ]

    for pdbs_dir in possible_locations:
        if pdbs_dir.exists():
            pdbs = sorted(pdbs_dir.glob("*.pdb"))
            if len(pdbs) >= 2:
                return [str(pdbs[0]), str(pdbs[1])]

    pytest.skip(f"Sample PDB files not available (tried {len(possible_locations)} locations)")


@pytest.fixture
def msa_structure_paths():
    """Return a list of at least three structure paths for MSA tests."""
    # Try multiple locations for test data
    test_dir = Path(__file__).resolve().parent
    possible_locations = [
        # In-tree development
        _find_repo_root() / "tests" / "data" / "integration" / "msa_families" / "lysozyme",
        # CI with copied data
        test_dir / "data" / "integration" / "msa_families" / "lysozyme",
        # Alternative CI structure
        test_dir.parent / "tests" / "data" / "integration" / "msa_families" / "lysozyme",
    ]

    for pdbs_dir in possible_locations:
        if pdbs_dir.exists():
            pdbs = sorted(pdbs_dir.glob("*.pdb"))
            if len(pdbs) >= 3:
                return [str(p) for p in pdbs[:3]]

    pytest.skip(f"Not enough structures for MSA tests (tried {len(possible_locations)} locations)")


@pytest.fixture
def random_embeddings():
    """Generate two compatible embedding arrays."""
    rng = np.random.default_rng(0)
    emb1 = rng.standard_normal((16, 64), dtype=np.float32)
    emb2 = rng.standard_normal((18, 64), dtype=np.float32)
    return emb1, emb2


@pytest.fixture
def golden_data_dir():
    """Path to golden data directory for validation tests.

    Returns path to tests/data/validation directory relative to project root.
    """
    repo_root = _find_repo_root()
    golden_dir = repo_root / "tests" / "data" / "validation"

    if not golden_dir.exists():
        pytest.skip(f"Golden data directory not found at {golden_dir}")

    return golden_dir
