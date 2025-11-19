"""Phylogenetic tree building for multiple sequence alignment.

This module provides tree construction algorithms for building guide trees
from distance matrices. Guide trees determine the order in which sequences
are progressively aligned in MSA.

Functions:
    build: Generic tree builder with method selection
    upgma: UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    nj: Neighbor-Joining
    bionj: BioNJ (improved Neighbor-Joining)
    mst: Minimum Spanning Tree

Classes:
    Tree: User-friendly tree wrapper with Newick export and I/O

Example:
    >>> import pfalign
    >>> import numpy as np
    >>>
    >>> # Build distance matrix
    >>> embeddings = [pfalign.encode(f) for f in files]
    >>> distances = pfalign.compute_distances(embeddings)
    >>>
    >>> # Build tree
    >>> tree = pfalign.tree.build(distances, method='upgma', labels=labels)
    >>> print(tree.newick)
    '((seq1:0.5,seq2:0.5):1.0,seq3:2.0);'
    >>>
    >>> # Save tree
    >>> tree.save("guide_tree.newick")
    >>>
    >>> # Or use specific methods
    >>> tree_nj = pfalign.tree.nj(distances, labels=labels)
    >>> tree_bionj = pfalign.tree.bionj(distances, labels=labels)
"""

from typing import List, Optional
import numpy as np
from pathlib import Path
from pfalign._align_cpp import tree as _tree_cpp


class Tree:
    """Phylogenetic tree with Newick format support.

    Wrapper around C++ GuideTree that provides user-friendly access to
    tree structure via Newick format and file I/O operations.

    Attributes:
        newick: Newick format string representation

    Methods:
        save(path): Save tree to Newick file
        load(path): Load tree from Newick file (class method)
    """

    def __init__(self, guide_tree, labels: Optional[List[str]] = None):
        """Create Tree from GuideTree object.

        Args:
            guide_tree: C++ GuideTree object from tree builders
            labels: Sequence labels (default: "seq0", "seq1", ...)
        """
        self._guide_tree = guide_tree

        # Generate default labels if not provided
        if labels is None:
            n = guide_tree.num_sequences
            labels = [f"seq{i}" for i in range(n)]

        # Validate labels count
        if len(labels) != guide_tree.num_sequences:
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of sequences ({guide_tree.num_sequences})"
            )

        self._labels = labels
        self._newick = None  # Cached Newick string

    @property
    def newick(self) -> str:
        """Get Newick format string representation.

        Newick format uses nested parentheses to represent tree topology:
        - Leaves: sequence names
        - Internal nodes: (left,right)
        - Branch lengths: :length after each node
        - Root: semicolon at end

        Example: ((A:0.5,B:0.5):1.0,C:2.0);

        Returns:
            Newick format string
        """
        if self._newick is None:
            self._newick = self._guide_tree.to_newick(self._labels)
        return self._newick

    def save(self, path: str) -> None:
        """Save tree to Newick format file.

        Args:
            path: Output file path (.newick or .nwk extension recommended)

        Example:
            >>> tree.save("guide_tree.newick")
        """
        with open(path, 'w') as f:
            f.write(self.newick)
            f.write('\n')  # Add newline for better file formatting

    @classmethod
    def load(cls, path: str) -> 'Tree':
        """Load tree from Newick format file.

        Note: This creates a Tree object that only stores the Newick string.
        It cannot be used for MSA guide tree construction, only for
        visualization or format conversion.

        Args:
            path: Input Newick file path

        Returns:
            Tree object

        Example:
            >>> tree = pfalign.tree.Tree.load("tree.newick")
            >>> print(tree.newick)
        """
        with open(path) as f:
            newick = f.read().strip()

        # Create a minimal Tree object that only holds the Newick string
        # This is a bit of a hack - we create a dummy GuideTree with 0 sequences
        # and override the newick property
        class NewickOnlyTree(Tree):
            def __init__(self, newick_str):
                self._newick = newick_str
                self._guide_tree = None
                self._labels = []

            @property
            def newick(self):
                return self._newick

        return NewickOnlyTree(newick)

    @property
    def num_sequences(self) -> int:
        """Get number of leaf sequences in tree."""
        if self._guide_tree is None:
            return 0
        return self._guide_tree.num_sequences

    @property
    def num_nodes(self) -> int:
        """Get total number of nodes (leaves + internal)."""
        if self._guide_tree is None:
            return 0
        return self._guide_tree.num_nodes()

    def __str__(self) -> str:
        """String representation returns Newick format."""
        return self.newick

    def __repr__(self) -> str:
        """Object representation."""
        if self._guide_tree is None:
            return f"<Tree (Newick-only)>"
        return f"<Tree with {self.num_sequences} sequences>"


# Re-export C++ tree builders with Tree wrapper
def upgma(distances: np.ndarray, labels: Optional[List[str]] = None) -> Tree:
    """Build UPGMA (Unweighted Pair Group Method with Arithmetic Mean) tree.

    UPGMA is a simple hierarchical clustering method that assumes a constant
    molecular clock (ultrametric tree). Fastest method, good for closely
    related sequences.

    Args:
        distances: Square symmetric distance matrix (N, N)
        labels: Sequence names (default: "seq0", "seq1", ...)

    Returns:
        Tree object with Newick representation

    Example:
        >>> distances = pfalign.compute_distances(embeddings)
        >>> tree = pfalign.tree.upgma(distances, labels=["proteinA", "proteinB", "proteinC"])
        >>> tree.save("upgma_tree.newick")
    """
    guide_tree = _tree_cpp.upgma(distances.astype(np.float32))
    return Tree(guide_tree, labels)


def nj(distances: np.ndarray, labels: Optional[List[str]] = None) -> Tree:
    """Build Neighbor-Joining tree.

    NJ is a bottom-up clustering method that doesn't assume a molecular clock.
    More accurate than UPGMA for divergent sequences. Standard method for
    phylogenetic tree construction.

    Args:
        distances: Square symmetric distance matrix (N, N)
        labels: Sequence names (default: "seq0", "seq1", ...")

    Returns:
        Tree object with Newick representation

    Example:
        >>> tree = pfalign.tree.nj(distances)
        >>> print(tree.newick)
    """
    guide_tree = _tree_cpp.nj(distances.astype(np.float32))
    return Tree(guide_tree, labels)


def bionj(distances: np.ndarray, labels: Optional[List[str]] = None) -> Tree:
    """Build BioNJ tree (improved Neighbor-Joining).

    BioNJ improves upon NJ by using variance-based distance corrections.
    More accurate than standard NJ, especially for sequences with varying
    evolutionary rates. Recommended for most use cases.

    Args:
        distances: Square symmetric distance matrix (N, N)
        labels: Sequence names (default: "seq0", "seq1", ...")

    Returns:
        Tree object with Newick representation

    Example:
        >>> tree = pfalign.tree.bionj(distances, labels=labels)
    """
    guide_tree = _tree_cpp.bionj(distances.astype(np.float32))
    return Tree(guide_tree, labels)


def mst(distances: np.ndarray, labels: Optional[List[str]] = None) -> Tree:
    """Build Minimum Spanning Tree.

    MST connects all sequences with minimum total edge weight. Results in
    a star-like topology. Useful for very divergent sequences or when
    phylogenetic relationships are unclear.

    Args:
        distances: Square symmetric distance matrix (N, N)
        labels: Sequence names (default: "seq0", "seq1", ...")

    Returns:
        Tree object with Newick representation

    Example:
        >>> tree = pfalign.tree.mst(distances)
    """
    guide_tree = _tree_cpp.mst(distances.astype(np.float32))
    return Tree(guide_tree, labels)


def build(
    distances: np.ndarray,
    method: str = 'upgma',
    labels: Optional[List[str]] = None
) -> Tree:
    """Build phylogenetic tree using specified method.

    Generic tree builder that dispatches to specific algorithm based on
    method parameter. Convenient for parameterized tree construction.

    Args:
        distances: Square symmetric distance matrix (N, N)
        method: Tree building method: 'upgma', 'nj', 'bionj', or 'mst'
               (default: 'upgma')
        labels: Sequence names (default: "seq0", "seq1", ...")

    Returns:
        Tree object with Newick representation

    Raises:
        ValueError: If method is not recognized

    Example:
        >>> # Try different methods
        >>> for method in ['upgma', 'nj', 'bionj']:
        ...     tree = pfalign.tree.build(distances, method=method)
        ...     tree.save(f"tree_{method}.newick")
    """
    method = method.lower()

    if method == 'upgma':
        return upgma(distances, labels)
    elif method == 'nj':
        return nj(distances, labels)
    elif method == 'bionj':
        return bionj(distances, labels)
    elif method == 'mst':
        return mst(distances, labels)
    else:
        raise ValueError(
            f"Unknown tree method: '{method}'. "
            f"Valid options: 'upgma', 'nj', 'bionj', 'mst'"
        )


__all__ = [
    'Tree',
    'build',
    'upgma',
    'nj',
    'bionj',
    'mst',
]
