import numpy as np
import pytest

import pfalign


def test_msa_embeddings(random_embeddings):
    """Test MSA uses embedded SW parameters from trained model."""
    emb1, emb2 = random_embeddings
    emb3 = np.copy(emb1)
    result = pfalign.msa(
        [emb1, emb2, emb3],
        method="nj",
        # Gap parameters loaded from embedded weights
        ecs_temperature=3.0,
        arena_size_mb=150,
    )
    assert isinstance(result, pfalign.MSAResult)
    assert result.num_sequences == 3
    assert result.alignment_length > 0
    arr = result.to_array()
    assert arr.shape == (result.alignment_length, result.num_sequences)


@pytest.mark.slow
def test_msa_structures(msa_structure_paths):
    """Test MSA from structures uses embedded SW parameters."""
    result = pfalign.msa(
        msa_structure_paths,
        method="upgma",
        k_neighbors=6,
        # Gap parameters loaded from embedded weights
        ecs_temperature=4.5,
        arena_size_mb=256,
    )
    assert isinstance(result, pfalign.MSAResult)
    assert result.num_sequences == len(msa_structure_paths)
    assert len(result.sequences()) == len(msa_structure_paths)
    consensus = result.get_consensus(0.4)
    assert len(consensus) == result.alignment_length
