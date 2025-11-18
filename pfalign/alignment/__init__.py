"""Low-level alignment primitives.

This module exposes the internal Smith-Waterman and Viterbi decoding algorithms
as composable primitives for research and custom alignment pipelines.

Functions:
    forward: Smith-Waterman forward pass only
    backward: Smith-Waterman backward pass only
    forward_backward: Compute soft alignment (posterior matrix + score)
    score: Get alignment score only (faster than forward_backward)
    viterbi_decode: Decode hard alignment with sequences from posteriors
    viterbi_path: Decode hard alignment path (coordinates only) from posteriors

Example:
    >>> import pfalign
    >>> import numpy as np
    >>>
    >>> # Encode structures
    >>> emb1 = pfalign.encode("protein1.pdb")
    >>> emb2 = pfalign.encode("protein2.pdb")
    >>>
    >>> # Compute similarity
    >>> sim = emb1.embeddings @ emb2.embeddings.T
    >>>
    >>> # Soft alignment (full forward-backward)
    >>> posterior, log_Z = pfalign.alignment.forward_backward(sim)
    >>>
    >>> # Or just get the score
    >>> score = pfalign.alignment.score(sim)
    >>>
    >>> # Hard alignment decoding
    >>> seq1, seq2, path = pfalign.alignment.viterbi_decode(posterior)
    >>> print(f"Alignment: {seq1}")
    >>> print(f"           {seq2}")
    >>>
    >>> # Or just get the path coordinates
    >>> path = pfalign.alignment.viterbi_path(posterior)
    >>>
    >>> # Low-level: forward pass only
    >>> fwd = pfalign.alignment.forward(sim, gap_open=-2.544, gap_extend=0.194)
    >>>
    >>> # Experiment with different gap penalties
    >>> path_strict = pfalign.alignment.viterbi_path(posterior, gap_penalty=-10.0)
    >>> path_relaxed = pfalign.alignment.viterbi_path(posterior, gap_penalty=-1.0)
"""

from typing import Tuple, List
import numpy as np
from pfalign._align_cpp import alignment as _alignment_cpp

# Re-export all functions from C++ module
forward = _alignment_cpp.forward
backward = _alignment_cpp.backward
forward_backward = _alignment_cpp.forward_backward
score = _alignment_cpp.score
viterbi_decode = _alignment_cpp.viterbi_decode
viterbi_path = _alignment_cpp.viterbi_path

__all__ = [
    "forward",
    "backward",
    "forward_backward",
    "score",
    "viterbi_decode",
    "viterbi_path",
]
