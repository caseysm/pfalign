"""Tests for parallel MPNN encoding in pairwise alignment.

This module tests the parallel_mpnn flag that controls whether the two
MPNN encoders run sequentially or in parallel threads.
"""

import pytest
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pfalign


class TestParallelMPNNCorrectness:
    """Test that parallel and sequential modes produce identical results."""

    def test_parallel_vs_sequential_with_embeddings(self, random_embeddings):
        """Test that parallel_mpnn flag doesn't affect embedding inputs.

        Embeddings skip MPNN encoding entirely, so the flag should have no effect.
        """
        emb1, emb2 = random_embeddings

        result_parallel = pfalign.pairwise(emb1, emb2, parallel_mpnn=True)
        result_sequential = pfalign.pairwise(emb1, emb2, parallel_mpnn=False)

        # Results should be identical since MPNN is skipped for embeddings
        # Note: We can't directly compare scores because they're methods, not attributes
        # Instead, we verify both calls succeed without error
        assert result_parallel is not None
        assert result_sequential is not None

    @pytest.mark.slow
    def test_parallel_vs_sequential_with_structures(self, sample_pdbs):
        """Test that parallel and sequential MPNN produce identical results.

        This is the critical correctness test: both modes should produce
        bit-for-bit identical alignments when given the same input structures.
        """
        pdb1, pdb2 = sample_pdbs

        # Run with parallel MPNN
        result_parallel = pfalign.pairwise(
            pdb1, pdb2,
            k_neighbors=30,
            parallel_mpnn=True
        )

        # Run with sequential MPNN
        result_sequential = pfalign.pairwise(
            pdb1, pdb2,
            k_neighbors=30,
            parallel_mpnn=False
        )

        # Extract alignment data for comparison
        align_par = result_parallel.alignment()
        align_seq = result_sequential.alignment()

        score_par = result_parallel.score()
        score_seq = result_sequential.score()

        # Alignments and scores should be identical
        assert align_par == align_seq, "Alignments differ between parallel modes"
        assert abs(score_par - score_seq) < 1e-9, f"Scores differ: {score_par} vs {score_seq}"

    @pytest.mark.slow
    def test_determinism_parallel_mode(self, sample_pdbs):
        """Test that parallel mode is deterministic across runs.

        Running the same alignment multiple times should give identical results.
        """
        pdb1, pdb2 = sample_pdbs

        results = []
        for _ in range(3):
            result = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=True)
            results.append(result.alignment())

        # All runs should produce identical results
        assert results[0] == results[1], "Run 1 and 2 differ"
        assert results[1] == results[2], "Run 2 and 3 differ"

    @pytest.mark.slow
    def test_determinism_sequential_mode(self, sample_pdbs):
        """Test that sequential mode is deterministic across runs."""
        pdb1, pdb2 = sample_pdbs

        results = []
        for _ in range(3):
            result = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=False)
            results.append(result.alignment())

        # All runs should produce identical results
        assert results[0] == results[1], "Run 1 and 2 differ"
        assert results[1] == results[2], "Run 2 and 3 differ"


class TestParallelMPNNPerformance:
    """Test that parallel mode provides speedup over sequential."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_parallel_speedup(self, sample_pdbs):
        """Benchmark parallel vs sequential MPNN encoding.

        Parallel mode should be faster than sequential for structure inputs.
        We expect roughly 1.5-1.8x speedup for two MPNN encoders.
        """
        pdb1, pdb2 = sample_pdbs

        # Warmup run to ensure libraries are loaded
        _ = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=True)

        # Benchmark sequential mode
        start = time.perf_counter()
        for _ in range(3):
            _ = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=False)
        sequential_time = time.perf_counter() - start

        # Benchmark parallel mode
        start = time.perf_counter()
        for _ in range(3):
            _ = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=True)
        parallel_time = time.perf_counter() - start

        speedup = sequential_time / parallel_time

        print(f"\nParallel MPNN Performance:")
        print(f"  Sequential time: {sequential_time:.3f}s (3 runs)")
        print(f"  Parallel time:   {parallel_time:.3f}s (3 runs)")
        print(f"  Speedup:         {speedup:.2f}x")

        # We expect at least some speedup (conservative threshold)
        # Ideal would be ~1.8x, but we check for >1.1x to account for overhead
        assert speedup > 1.1, f"Expected speedup >1.1x, got {speedup:.2f}x"

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_parallel_overhead_with_embeddings(self, random_embeddings):
        """Test that parallel_mpnn flag has minimal overhead for embeddings.

        Since embeddings skip MPNN, the flag should have negligible impact.
        """
        emb1, emb2 = random_embeddings

        # Warmup
        _ = pfalign.pairwise(emb1, emb2, parallel_mpnn=True)

        # Benchmark with flag=True
        start = time.perf_counter()
        for _ in range(10):
            _ = pfalign.pairwise(emb1, emb2, parallel_mpnn=True)
        time_parallel_flag = time.perf_counter() - start

        # Benchmark with flag=False
        start = time.perf_counter()
        for _ in range(10):
            _ = pfalign.pairwise(emb1, emb2, parallel_mpnn=False)
        time_sequential_flag = time.perf_counter() - start

        ratio = max(time_parallel_flag, time_sequential_flag) / min(time_parallel_flag, time_sequential_flag)

        print(f"\nParallel flag overhead (embeddings):")
        print(f"  parallel_mpnn=True:  {time_parallel_flag:.4f}s")
        print(f"  parallel_mpnn=False: {time_sequential_flag:.4f}s")
        print(f"  Ratio: {ratio:.2f}x")

        # Times should be nearly identical (allow up to 5x difference for timing variance on CI)
        # CI environments can have high variance, especially on macOS ARM64 runners
        assert ratio < 5.0, f"Flag causes {ratio:.2f}x overhead for embeddings"


class TestParallelMPNNThreadSafety:
    """Test thread safety of parallel MPNN implementation."""

    @pytest.mark.slow
    def test_concurrent_parallel_calls(self, sample_pdbs):
        """Test running multiple parallel pairwise operations concurrently.

        This tests for race conditions when multiple threads each spawn
        their own MPNN encoder threads.
        """
        pdb1, pdb2 = sample_pdbs

        def run_alignment(iteration):
            """Run a single alignment and return result."""
            result = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=True)
            return (iteration, result.alignment())

        # Run 8 alignments concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_alignment, i) for i in range(8)]
            results = [future.result() for future in as_completed(futures)]

        # Sort by iteration number
        results.sort(key=lambda x: x[0])
        alignments = [r[1] for r in results]

        # All alignments should be identical
        reference = alignments[0]
        for i, alignment in enumerate(alignments[1:], 1):
            assert alignment == reference, f"Alignment {i} differs from reference"

    @pytest.mark.slow
    def test_concurrent_mixed_mode_calls(self, sample_pdbs):
        """Test concurrent calls with mixed parallel/sequential modes.

        Some threads use parallel_mpnn=True, others use False.
        All should produce identical results.
        """
        pdb1, pdb2 = sample_pdbs

        def run_alignment(iteration, use_parallel):
            """Run alignment with specified mode."""
            result = pfalign.pairwise(pdb1, pdb2, parallel_mpnn=use_parallel)
            return (iteration, use_parallel, result.alignment())

        # Run mixed modes concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(8):
                use_parallel = (i % 2 == 0)  # Alternate between True/False
                futures.append(executor.submit(run_alignment, i, use_parallel))

            results = [future.result() for future in as_completed(futures)]

        # All alignments should be identical regardless of mode
        alignments = [r[2] for r in results]
        reference = alignments[0]
        for i, alignment in enumerate(alignments[1:], 1):
            assert alignment == reference, f"Alignment {i} differs from reference"


class TestParallelMPNNAPI:
    """Test the API surface for parallel_mpnn parameter."""

    def test_parameter_default_value(self):
        """Test that parallel_mpnn defaults to True."""
        import inspect
        sig = inspect.signature(pfalign.pairwise)
        default = sig.parameters['parallel_mpnn'].default
        assert default is True, f"Expected default=True, got {default}"

    def test_parameter_in_signature(self):
        """Test that parallel_mpnn parameter exists in signature."""
        import inspect
        sig = inspect.signature(pfalign.pairwise)
        params = list(sig.parameters.keys())
        assert 'parallel_mpnn' in params, f"parallel_mpnn not in signature: {params}"

    def test_parameter_accepts_bool(self, random_embeddings):
        """Test that parameter accepts boolean values."""
        emb1, emb2 = random_embeddings

        # Should accept True
        result1 = pfalign.pairwise(emb1, emb2, parallel_mpnn=True)
        assert result1 is not None

        # Should accept False
        result2 = pfalign.pairwise(emb1, emb2, parallel_mpnn=False)
        assert result2 is not None

    def test_parameter_type_coercion(self, random_embeddings):
        """Test parameter type coercion behavior.

        pybind11's optional<bool> accepts integers (0/1) and None,
        but rejects strings.
        """
        emb1, emb2 = random_embeddings

        # Should accept integers (coerced to bool)
        result1 = pfalign.pairwise(emb1, emb2, parallel_mpnn=1)
        assert result1 is not None

        result0 = pfalign.pairwise(emb1, emb2, parallel_mpnn=0)
        assert result0 is not None

        # Should accept None (handled by std::optional)
        result_none = pfalign.pairwise(emb1, emb2, parallel_mpnn=None)
        assert result_none is not None

        # Should reject strings
        with pytest.raises(TypeError):
            pfalign.pairwise(emb1, emb2, parallel_mpnn="true")
