import numpy as np

import pfalign


def test_pairwise_result_threshold(random_embeddings):
    emb1, emb2 = random_embeddings
    result = pfalign.pairwise(emb1, emb2)
    thresh = result.threshold(0.5)
    assert isinstance(thresh, pfalign.PairwiseResult)
    assert thresh.posteriors.shape == result.posteriors.shape


def test_embedding_result_helpers(sample_pdbs):
    emb = pfalign.encode(sample_pdbs[0], k_neighbors=8)
    norm = emb.normalize()
    assert np.allclose(np.linalg.norm(np.asarray(norm), axis=1), 1.0, atol=1e-4)
    subset = emb.get_subset([0, 2, 4])
    assert subset.sequence_length() == 3


def test_similarity_result_utilities(random_embeddings):
    emb1, emb2 = random_embeddings
    sim = pfalign.similarity(emb1, emb2)
    top = sim.get_top_k(4)
    assert len(top) == 4
    filt = sim.threshold(0.0)
    assert isinstance(filt, pfalign.SimilarityResult)
    norm = sim.normalize()
    array = np.asarray(norm)
    assert array.min() >= 0.0 and array.max() <= 1.0


def test_msa_result_access(msa_structure_paths):
    result = pfalign.msa(msa_structure_paths[:3], method="upgma", k_neighbors=6)
    seq = result.get_sequence(0)
    assert len(seq) == result.alignment_length()
    column = result.get_column(0)
    assert len(column) == result.num_sequences()
