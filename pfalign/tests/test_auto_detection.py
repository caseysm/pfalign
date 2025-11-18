from pathlib import Path

import numpy as np
import pytest

import pfalign


def test_detect_input_type(sample_pdbs, tmp_path):
    pdb = sample_pdbs[0]
    assert pfalign._detect_input_type(pdb) == "structure"
    arr = np.random.randn(4, 64).astype(np.float32)
    assert pfalign._detect_input_type(arr) == "embeddings"

    npy_path = tmp_path / "emb.npy"
    np.save(npy_path, arr)
    assert pfalign._detect_input_type(str(npy_path)) == "embeddings"


def test_extract_array(tmp_path):
    data = np.random.randn(5, 32).astype(np.float32)
    npy_path = tmp_path / "emb.npy"
    np.save(npy_path, data)
    loaded = pfalign._extract_array(str(npy_path))
    np.testing.assert_allclose(loaded, data)


def test_parse_msa_inputs_with_directory(msa_structure_paths, tmp_path):
    dir_path = tmp_path / "inputs"
    dir_path.mkdir()
    for idx, src in enumerate(msa_structure_paths):
        target = dir_path / f"seq_{idx}.pdb"
        target.write_text(Path(src).read_text())

    # Current API: pass directory path directly to inputs parameter
    paths = pfalign._parse_msa_inputs(inputs=str(dir_path))
    assert len(paths) == len(msa_structure_paths)


def test_parse_msa_inputs_with_list(msa_structure_paths):
    # Current API: pass list of paths
    paths = pfalign._parse_msa_inputs(inputs=msa_structure_paths)
    assert len(paths) == len(msa_structure_paths)
