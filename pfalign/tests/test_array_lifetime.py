"""
Test NumPy array lifetime fix (P1 critical issue).

Verifies that NumPy arrays created from result objects remain valid
even after the C++ result object is deleted. This tests the fix where
py::object self_py is passed as the base parameter to py::array_t.

Without this fix, accessing the array after deleting the result would
cause use-after-free crashes.
"""
import numpy as np
import pytest

import pfalign


def test_pairwise_posteriors_lifetime(random_embeddings):
    """Verify posteriors array keeps PairwiseResult alive."""
    emb1, emb2 = random_embeddings

    result = pfalign.pairwise(emb1, emb2)
    arr = result.posteriors  # Get numpy view
    L1, L2 = result.L1(), result.L2()

    # Delete the C++ object - array should keep it alive via py::object base
    del result

    # This should NOT crash - arr keeps the C++ object alive
    assert arr.shape == (L1, L2)
    assert arr.dtype == np.float32

    # Access elements - should work without segfault
    assert arr[0, 0] >= 0.0
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0

    # Array should still be usable
    mean_val = np.mean(arr)
    assert 0.0 <= mean_val <= 1.0


def test_embedding_result_lifetime(sample_pdbs):
    """Verify embeddings array keeps EmbeddingResult alive."""
    result = pfalign.encode(sample_pdbs[0], k_neighbors=8)
    arr = result.embeddings  # Get numpy view
    L = result.sequence_length()
    D = result.hidden_dim()

    # Delete the C++ object
    del result

    # This should NOT crash
    assert arr.shape == (L, D)
    assert arr.dtype == np.float32

    # Access elements - should work
    assert arr[0, 0] != 0.0  # Embeddings are non-zero

    # Array operations should work
    norm = np.linalg.norm(arr, axis=1)
    assert len(norm) == L
    assert np.all(norm > 0.0)


def test_embedding_result_array_protocol_lifetime(sample_pdbs):
    """Verify __array__ protocol keeps EmbeddingResult alive."""
    result = pfalign.encode(sample_pdbs[0], k_neighbors=8)

    # Convert to numpy array via __array__ protocol
    arr = np.asarray(result)
    L = result.sequence_length()
    D = result.hidden_dim()

    # Delete the C++ object
    del result

    # This should NOT crash
    assert arr.shape == (L, D)
    assert arr.dtype == np.float32

    # Array operations should work
    assert arr[0, 0] != 0.0
    mean_val = np.mean(arr)
    assert mean_val != 0.0


def test_similarity_result_lifetime(random_embeddings):
    """Verify similarity array keeps SimilarityResult alive."""
    emb1, emb2 = random_embeddings

    result = pfalign.similarity(emb1, emb2)
    arr = result.similarity  # Get numpy view
    L1, L2 = result.L1(), result.L2()

    # Delete the C++ object
    del result

    # This should NOT crash
    assert arr.shape == (L1, L2)
    assert arr.dtype == np.float32

    # Access elements - should work
    max_sim = arr.max()
    assert max_sim > 0.0  # Similarity should be positive


def test_similarity_result_array_protocol_lifetime(random_embeddings):
    """Verify __array__ protocol keeps SimilarityResult alive."""
    emb1, emb2 = random_embeddings

    result = pfalign.similarity(emb1, emb2)
    arr = np.asarray(result)  # Convert via __array__ (use asarray, not array)
    L1, L2 = result.L1(), result.L2()

    # Delete the C++ object
    del result

    # This should NOT crash
    assert arr.shape == (L1, L2)
    assert arr.dtype == np.float32

    # Array operations should work (similarity can be negative)
    mean_val = np.mean(arr)
    assert np.isfinite(mean_val)


def test_multiple_views_same_result(random_embeddings):
    """Verify multiple numpy views to same result object work correctly."""
    emb1, emb2 = random_embeddings

    result = pfalign.pairwise(emb1, emb2)

    # Get multiple views
    arr1 = result.posteriors
    arr2 = result.posteriors
    arr3 = result.posteriors

    # All should share the same underlying data
    assert arr1.data == arr2.data == arr3.data

    # Delete result - all views should keep it alive
    del result

    # All views should still be valid
    assert arr1[0, 0] == arr2[0, 0] == arr3[0, 0]

    # Delete views one by one - last one should keep object alive
    del arr1
    assert arr2[0, 0] == arr3[0, 0]

    del arr2
    assert arr3.shape[0] > 0  # arr3 still valid


def test_array_lifetime_with_operations(sample_pdbs):
    """Verify array remains valid through numpy operations."""
    result = pfalign.encode(sample_pdbs[0], k_neighbors=8)
    arr = result.embeddings

    # Delete result before operations
    del result

    # Complex numpy operations - should not crash
    normalized = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    assert normalized.shape == arr.shape

    distances = np.linalg.norm(arr[0] - arr[1:], axis=1)
    assert len(distances) == arr.shape[0] - 1

    # Slicing operations
    subset = arr[:10]
    assert subset.shape[0] == 10

    # All operations should produce valid results
    assert np.all(np.isfinite(normalized))
    assert np.all(np.isfinite(distances))


def test_threshold_creates_new_independent_result(random_embeddings):
    """Verify threshold() creates independent result with its own lifetime."""
    emb1, emb2 = random_embeddings

    original = pfalign.pairwise(emb1, emb2)
    arr_original = original.posteriors

    # Create filtered result
    filtered = original.threshold(0.5)
    arr_filtered = filtered.posteriors

    # Both arrays should be valid
    assert arr_original.shape == arr_filtered.shape

    # Delete original - filtered should still work
    del original
    assert arr_filtered[0, 0] >= 0.0

    # Delete filtered - its array should still work
    del filtered
    assert arr_filtered.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
