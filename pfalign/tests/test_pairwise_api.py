import numpy as np
import pytest

import pfalign


def test_pairwise_embeddings(random_embeddings):
    """Test pairwise alignment uses embedded SW parameters from trained model."""
    emb1, emb2 = random_embeddings
    result = pfalign.pairwise(emb1, emb2)  # Uses embedded gap params
    assert isinstance(result, pfalign.PairwiseResult)
    assert result.L1 == emb1.shape[0]
    assert result.L2 == emb2.shape[0]
    posteriors = result.posteriors
    assert posteriors.shape == (emb1.shape[0], emb2.shape[0])
    assert np.isfinite(posteriors).all()


@pytest.mark.slow
def test_pairwise_structures(sample_pdbs):
    """Test pairwise alignment from structures uses embedded SW parameters."""
    pdb1, pdb2 = sample_pdbs
    result = pfalign.pairwise(
        pdb1,
        pdb2,
        chain1=0,
        chain2=0,
        k_neighbors=6,
        # Gap parameters loaded from embedded weights
    )
    assert isinstance(result, pfalign.PairwiseResult)
    assert result.L1 > 0 and result.L2 > 0
    assert result.posteriors.shape == (result.L1, result.L2)
    assert result.compute_coverage() > 0.0


def test_pairwise_custom_gap_params(random_embeddings):
    """Test that custom gap parameters can be overridden."""
    emb1, emb2 = random_embeddings

    # Use default embedded parameters
    result_default = pfalign.pairwise(emb1, emb2)

    # Use custom parameters (different from embedded -2.544, 0.194, 1.0)
    result_custom = pfalign.pairwise(
        emb1, emb2,
        gap_open=-1.0,
        gap_extend=-0.1,
        temperature=0.8
    )

    # Both should produce valid results
    assert isinstance(result_default, pfalign.PairwiseResult)
    assert isinstance(result_custom, pfalign.PairwiseResult)

    # Results should differ since parameters differ
    assert result_default.score != result_custom.score
    assert not np.allclose(result_default.posteriors, result_custom.posteriors)
