import numpy as np

import pfalign


def test_encode_and_similarity(sample_pdbs):
    pdb = sample_pdbs[0]
    embedding = pfalign.encode(pdb, k_neighbors=8, chain=0)
    assert isinstance(embedding, pfalign.EmbeddingResult)
    arr = np.asarray(embedding)
    assert arr.ndim == 2
    assert arr.shape[0] == embedding.sequence_length

    sim = pfalign.similarity(embedding, embedding)
    assert isinstance(sim, pfalign.SimilarityResult)
    assert sim.shape == (arr.shape[0], arr.shape[0])

    top = sim.get_top_k(5)
    assert len(top) == 5
