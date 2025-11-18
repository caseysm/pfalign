"""Python API surface for PFalign v2.0."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import warnings


def _load_extension():
    pkg_dir = Path(__file__).resolve().parent
    candidates = sorted(pkg_dir.glob("_align_cpp*.so")) + sorted(pkg_dir.glob("_align_cpp*.pyd"))
    if candidates:
        module_name = "pfalign._align_cpp"
        spec = importlib.util.spec_from_file_location(module_name, candidates[0])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    return importlib.import_module("pfalign._align_cpp")


_align_cpp = _load_extension()

PairwiseResult = _align_cpp.PairwiseResult
MSAResult = _align_cpp.MSAResult
EmbeddingResult = _align_cpp.EmbeddingResult
SimilarityResult = _align_cpp.SimilarityResult

# Expose submodules
metrics = _align_cpp.metrics
structure = _align_cpp.structure
tree = _align_cpp.tree

# Import alignment module (has its own __init__.py)
from . import alignment

# Version is managed by setuptools-scm via git tags
try:
    from pfalign._version import version as __version__
except ImportError:
    # Fallback for development installations without setuptools-scm
    __version__ = "0.0.1a0"

__all__ = [
    "pairwise",
    "msa",
    "encode",
    "encode_batch",
    "similarity",
    "compute_distances",
    "progressive_align",
    "reformat",
    "info",
    "version_info",
    "batch_pairwise",
    "PairwiseResult",
    "MSAResult",
    "EmbeddingResult",
    "SimilarityResult",
    "metrics",
    "structure",
    "tree",
    "alignment",
    "__version__",
]

_STRUCTURE_EXTENSIONS = (".pdb", ".cif", ".mmcif")
_EMBEDDING_EXTENSIONS = (".npy",)
_PDB_HEADERS = ("HEADER", "ATOM", "HETATM", "MODEL", "REMARK", "COMPND")


def _detect_input_type(input_data):
    """Auto-detect structure vs embedding inputs."""
    if hasattr(input_data, "__array__") and hasattr(input_data, "embeddings"):
        return "embeddings"
    if hasattr(input_data, "__array__") and hasattr(input_data, "similarity"):
        return "embeddings"
    if isinstance(input_data, np.ndarray):
        return "embeddings"

    if isinstance(input_data, (str, os.PathLike)):
        path = str(input_data)
        lower = path.lower()
        if lower.endswith(_EMBEDDING_EXTENSIONS):
            return "embeddings"
        if lower.endswith(_STRUCTURE_EXTENSIONS):
            return "structure"
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                first_line = handle.readline().strip()
            if first_line.startswith(_PDB_HEADERS) or first_line.startswith("data_"):
                return "structure"
        except OSError:
            pass
        raise ValueError(
            f"Cannot detect file type for {path}. "
            "Supported: .pdb, .cif, .mmcif (structures) or .npy (embeddings)."
        )

    raise TypeError(f"Unsupported input type: {type(input_data)}")


def _extract_array(input_data):
    """Extract a float32 NumPy array from result objects or .npy paths."""
    if isinstance(input_data, (str, os.PathLike)) and str(input_data).lower().endswith(".npy"):
        return np.load(str(input_data), allow_pickle=False).astype(np.float32, copy=False)
    if hasattr(input_data, "__array__"):
        array = np.asarray(input_data)
        return array.astype(np.float32, copy=False)
    if isinstance(input_data, np.ndarray):
        return np.asarray(input_data, dtype=np.float32)
    raise TypeError(f"Cannot convert {type(input_data)} to numpy array")


def _resolve_chain(structure_path: str | os.PathLike, chain: int | str) -> int:
    """Resolve chain name or index to chain index.

    Args:
        structure_path: Path to structure file
        chain: Chain specifier - either:
            - int: Chain index (0, 1, 2, ...)
            - str: Chain ID ("A", "B", "C", ...)

    Returns:
        Chain index (int)

    Raises:
        RuntimeError: If chain ID not found in structure

    Examples:
        >>> _resolve_chain("protein.pdb", 0)  # Returns 0
        >>> _resolve_chain("protein.pdb", "A")  # Returns 0 if chain A is first
        >>> _resolve_chain("protein.pdb", "B")  # Returns 1 if chain B is second
    """
    if isinstance(chain, int):
        return chain
    if isinstance(chain, str):
        return _align_cpp._resolve_chain_index(str(structure_path), chain)
    raise TypeError(f"chain must be int or str, got {type(chain)}")


def _parse_msa_inputs(inputs):
    """Parse MSA inputs with smart auto-detection of input type.

    Supports three input modes:
    - List of paths: ["protein1.pdb", "protein2.pdb", ...]
    - Text file: "proteins.txt" (one path per line, # for comments)
    - Directory: "proteins/" (all .pdb, .cif, .mmcif, .npy files)
    """
    paths: List[object] = []

    if isinstance(inputs, (str, os.PathLike)):
        candidate = Path(inputs)

        # Auto-detect: directory, text file, or single file
        if candidate.is_dir():
            # Directory mode
            collected = []
            for item in candidate.iterdir():
                if item.is_file() and item.suffix.lower() in _STRUCTURE_EXTENSIONS + _EMBEDDING_EXTENSIONS:
                    collected.append(str(item.resolve()))
            if not collected:
                raise ValueError(f"No supported files found in directory: {candidate}")
            collected.sort()
            paths.extend(collected)

        elif candidate.suffix.lower() == ".txt":
            # Text file list mode
            list_path = candidate
            base = list_path.parent.resolve()
            with list_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    entry = Path(line)
                    if not entry.is_absolute():
                        entry = base / entry
                    paths.append(str(entry))
        else:
            # Single file mode
            paths = [str(candidate)]
    else:
        # List mode
        paths = list(inputs)

    if not paths:
        raise ValueError("No inputs provided for MSA")

    return paths


def pairwise(
    input1,
    input2,
    *,
    k_neighbors: int = 30,
    chain1: int | str = 0,
    chain2: int | str = 0,
    gap_open: float | None = None,
    gap_extend: float | None = None,
    temperature: float | None = None,
    parallel_mpnn: bool = True,
):
    """Pairwise alignment with auto-detection of input type.

    Gap parameters (gap_open, gap_extend, temperature) default to the
    embedded trained model weights (gap_open=-2.544, gap_extend=0.194,
    temperature=1.0) but can be overridden if needed.

    Args:
        input1: First input (structure path or embeddings array)
        input2: Second input (structure path or embeddings array)
        k_neighbors: Number of nearest neighbors for graph construction
        chain1: Chain specifier for first structure - index (0, 1) or ID ("A", "B")
        chain2: Chain specifier for second structure - index (0, 1) or ID ("A", "B")
        gap_open: Gap opening penalty (default: -2.544 from trained model)
        gap_extend: Gap extension penalty (default: 0.194 from trained model)
        temperature: Softmax temperature (default: 1.0 from trained model)
        parallel_mpnn: Run MPNN encoders in parallel (default: True)

    Examples:
        >>> # By chain index
        >>> result = pairwise("p1.pdb", "p2.pdb", chain1=0, chain2=0)

        >>> # By chain ID
        >>> result = pairwise("p1.pdb", "p2.pdb", chain1="A", chain2="B")
    """
    type1 = _detect_input_type(input1)
    type2 = _detect_input_type(input2)
    if type1 != type2:
        raise ValueError("Mixed input types are not supported")

    if type1 == "structure":
        # Resolve chain names to indices
        chain1_idx = _resolve_chain(input1, chain1)
        chain2_idx = _resolve_chain(input2, chain2)

        return _align_cpp._pairwise_from_structures(
            str(input1),
            str(input2),
            chain1=chain1_idx,
            chain2=chain2_idx,
            k_neighbors=k_neighbors,
            gap_open=gap_open,
            gap_extend=gap_extend,
            temperature=temperature,
            parallel_mpnn=parallel_mpnn,
        )

    emb1 = _extract_array(input1)
    emb2 = _extract_array(input2)
    return _align_cpp._pairwise_from_embeddings(
        emb1, emb2,
        gap_open=gap_open,
        gap_extend=gap_extend,
        temperature=temperature,
        parallel_mpnn=parallel_mpnn,
    )


def msa(
    inputs: Sequence[object] | str | os.PathLike,
    *,
    method: str = "upgma",
    ecs_temperature: float = 5.0,
    k_neighbors: int = 30,
    arena_size_mb: int = 200,
    gap_open: float | None = None,
    gap_extend: float | None = None,
    temperature: float | None = None,
    threads: int | None = None,
):
    """Multiple sequence alignment with smart input detection.

    The inputs parameter accepts three formats:
    - List of paths: ["protein1.pdb", "protein2.pdb", ...]
    - Path to text file: "proteins.txt" (one path per line)
    - Path to directory: "proteins/" (all structures in directory)

    Gap parameters (gap_open, gap_extend, temperature) default to the
    embedded trained model weights (gap_open=-2.544, gap_extend=0.194,
    temperature=1.0) but can be overridden if needed.

    Args:
        inputs: Input source - can be:
            - List: ["protein1.pdb", "protein2.pdb", ...]
            - Text file path: "proteins.txt" (one path per line)
            - Directory path: "proteins/" (all structures/embeddings)
        method: Guide tree method ('upgma', 'nj', 'bionj', 'mst')
        ecs_temperature: Temperature for ECS computation
        k_neighbors: Number of nearest neighbors for graph construction
        arena_size_mb: Memory arena size in MB
        gap_open: Gap opening penalty (default: -2.544 from trained model)
        gap_extend: Gap extension penalty (default: 0.194 from trained model)
        temperature: Softmax temperature (default: 1.0 from trained model)
        threads: Override worker threads (default: auto-detect, clamped to available cores)

    Examples:
        >>> # From list of files
        >>> result = msa(["protein1.pdb", "protein2.pdb", "protein3.pdb"])

        >>> # From text file
        >>> result = msa("proteins.txt")

        >>> # From directory
        >>> result = msa("structures/")
    """
    paths = _parse_msa_inputs(inputs)
    input_type = _detect_input_type(paths[0])
    thread_count: int | None = None
    if threads is not None:
        try:
            thread_value = int(threads)
        except (TypeError, ValueError) as exc:
            raise ValueError("threads must be an integer >= 0") from exc
        if thread_value < 0:
            raise ValueError("threads must be >= 0")
        hardware_threads = os.cpu_count() or 1
        if thread_value > hardware_threads:
            warnings.warn(
                f"Requested {thread_value} threads but only {hardware_threads} CPU cores detected; "
                f"using {hardware_threads} threads instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            thread_value = hardware_threads
        thread_count = thread_value

    if input_type == "structure":
        string_paths = [str(p) for p in paths]

        # Single progress bar for all phases
        progress_bar = _align_cpp.ProgressBar(1, "Initializing...", width=20)
        current_phase = {"description": "", "total": 0}

        def progress_callback(current, total, description):
            # Check if we've moved to a new phase
            if description != current_phase["description"] or total != current_phase["total"]:
                # Reset the bar for the new phase (stays on same line)
                progress_bar.reset(total, description)
                current_phase["description"] = description
                current_phase["total"] = total

            # Update progress within current phase
            progress_bar.update(current)

        result = _align_cpp._msa_from_structures(
            string_paths,
            method=method,
            ecs_temperature=ecs_temperature,
            k_neighbors=k_neighbors,
            arena_size_mb=arena_size_mb,
            gap_open=gap_open,
            gap_extend=gap_extend,
            temperature=temperature,
            thread_count=thread_count,
            progress_callback=progress_callback,
        )

        # Finish the progress bar
        progress_bar.finish()

        return result

    embeddings = [_extract_array(entry) for entry in paths]

    # Single progress bar for all phases
    progress_bar = _align_cpp.ProgressBar(1, "Initializing...", width=20)
    current_phase = {"description": "", "total": 0}

    def progress_callback(current, total, description):
        # Check if we've moved to a new phase
        if description != current_phase["description"] or total != current_phase["total"]:
            # Reset the bar for the new phase (stays on same line)
            progress_bar.reset(total, description)
            current_phase["description"] = description
            current_phase["total"] = total

        # Update progress within current phase
        progress_bar.update(current)

    result = _align_cpp._msa_from_embeddings(
        embeddings,
        method=method,
        ecs_temperature=ecs_temperature,
        arena_size_mb=arena_size_mb,
        gap_open=gap_open,
        gap_extend=gap_extend,
        temperature=temperature,
        thread_count=thread_count,
        progress_callback=progress_callback,
    )

    # Finish the progress bar
    progress_bar.finish()

    return result


def encode(
    input_path: str | os.PathLike,
    *,
    k_neighbors: int = 30,
    chain: int | str = 0,
):
    """Encode a structure (PDB/mmCIF) to PFalign embeddings.

    Args:
        input_path: Path to structure file
        k_neighbors: Number of nearest neighbors for MPNN graph
        chain: Chain specifier - either index (0, 1, ...) or ID ("A", "B", ...)

    Returns:
        EmbeddingResult with embeddings array

    Examples:
        >>> # By chain index
        >>> emb = encode("protein.pdb", chain=0)

        >>> # By chain ID
        >>> emb = encode("protein.pdb", chain="A")
    """
    chain_idx = _resolve_chain(input_path, chain)
    return _align_cpp._encode_structure(
        str(input_path),
        chain=chain_idx,
        k_neighbors=k_neighbors,
    )


def encode_batch(
    inputs: Sequence[object] | str | os.PathLike,
    *,
    output_dir: str | os.PathLike,
    k_neighbors: int = 30,
    chain: int | str = 0,
):
    """Batch encode multiple structures to PFalign embeddings with smart input detection.

    The inputs parameter accepts three formats:
    - List of paths: ["protein1.pdb", "protein2.pdb", ...]
    - Path to text file: "proteins.txt" (one path per line)
    - Path to directory: "proteins/" (all structures in directory)

    Args:
        inputs: Input source (list, text file, or directory)
        output_dir: Directory to save output embeddings (.npy files)
        k_neighbors: Number of nearest neighbors for graph construction
        chain: Chain specifier - either index (0, 1, ...) or ID ("A", "B", ...)

    Returns:
        List of output paths for the encoded embeddings

    Examples:
        >>> # From list with chain index
        >>> paths = encode_batch(["p1.pdb", "p2.pdb"], output_dir="embeddings/")

        >>> # From directory with chain ID
        >>> paths = encode_batch("structures/", output_dir="embeddings/", chain="A")
    """
    # Parse inputs with smart detection
    paths = _parse_msa_inputs(inputs)

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create progress bar for batch encoding
    total_structures = len(paths)
    progress_bar = _align_cpp.ProgressBar(total_structures, "Encoding structures", width=20)

    # Encode each structure
    output_paths = []
    for i, structure_path in enumerate(paths):
        # Get base name and create output path
        base_name = Path(structure_path).stem
        output_file = output_path / f"{base_name}.npy"

        try:
            # Resolve chain and encode structure
            chain_idx = _resolve_chain(structure_path, chain)
            result = _align_cpp._encode_structure(
                str(structure_path),
                chain=chain_idx,
                k_neighbors=k_neighbors,
            )

            # Save embeddings
            result.save(str(output_file))
            output_paths.append(str(output_file))

        except Exception as e:
            warnings.warn(
                f"Failed to encode {structure_path}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            # Update progress bar regardless of success/failure
            progress_bar.update(i + 1)

    # Finish the progress bar
    progress_bar.finish()

    return output_paths


def similarity(embeddings1, embeddings2):
    """Compute similarity matrix between two embedding sets."""
    emb1 = _extract_array(embeddings1)
    emb2 = _extract_array(embeddings2)
    return _align_cpp._similarity_impl(emb1, emb2)


def compute_distances(
    inputs: Sequence[object] | None = None,
    *,
    input_list: str | os.PathLike | None = None,
    input_dir: str | os.PathLike | None = None,
    gap_open: float | None = None,
    gap_extend: float | None = None,
    temperature: float | None = None,
):
    """Compute pairwise distance matrix between embeddings.

    Computes alignment-based distances for guide tree construction.
    Distance formula: d(i,j) = (1 - score_cosine) / 2

    Gap parameters (gap_open, gap_extend, temperature) default to the
    embedded trained model weights (gap_open=-2.544, gap_extend=0.194,
    temperature=1.0) but can be overridden if needed.

    Args:
        inputs: Sequence of embedding arrays or .npy paths
        input_list: Path to file containing list of embeddings (one per line)
        input_dir: Directory containing .npy embedding files
        gap_open: Gap opening penalty (default: -2.544 from trained model)
        gap_extend: Gap extension penalty (default: 0.194 from trained model)
        temperature: Softmax temperature (default: 1.0 from trained model)

    Returns:
        Distance matrix as NumPy array of shape (N, N)

    Example:
        >>> # From directory of .npy files
        >>> distances = pfalign.compute_distances(input_dir="embeddings/")
        >>>
        >>> # From list of arrays
        >>> emb1 = pfalign.encode("struct1.pdb")
        >>> emb2 = pfalign.encode("struct2.pdb")
        >>> distances = pfalign.compute_distances([emb1, emb2])
    """
    # Determine which input mode to use
    actual_input = None
    if inputs is not None:
        actual_input = inputs
    elif input_list is not None:
        actual_input = input_list
    elif input_dir is not None:
        actual_input = input_dir
    else:
        raise ValueError("Must provide one of: inputs, input_list, or input_dir")

    paths = _parse_msa_inputs(actual_input)

    # Load embeddings with progress bar
    embeddings = []
    progress_bar = _align_cpp.ProgressBar(len(paths), "Loading embeddings", width=20)
    for idx, entry in enumerate(paths):
        embeddings.append(_extract_array(entry))
        progress_bar.update(idx + 1)
    progress_bar.finish()

    return _align_cpp._compute_distances_from_embeddings(
        embeddings,
        gap_open=gap_open,
        gap_extend=gap_extend,
        temperature=temperature,
    )


def progressive_align(
    embeddings: Sequence[object],
    tree,
    *,
    gap_open: float | None = None,
    gap_extend: float | None = None,
    temperature: float | None = None,
    ecs_temperature: float = 5.0,
    arena_size_mb: int = 200,
    thread_count: int | None = None,
):
    """Perform progressive MSA using a custom guide tree.

    Allows fine-grained control over MSA construction by accepting a
    pre-built guide tree from pfalign.tree.upgma/nj/bionj/mst.

    Gap parameters (gap_open, gap_extend, temperature) default to the
    embedded trained model weights (gap_open=-2.544, gap_extend=0.194,
    temperature=1.0) but can be overridden if needed.

    Args:
        embeddings: List of embedding arrays (N, D) for each sequence
        tree: GuideTree from pfalign.tree.upgma/nj/bionj/mst
        gap_open: Gap opening penalty (default: -2.544 from trained model)
        gap_extend: Gap extension penalty (default: 0.194 from trained model)
        temperature: Softmax temperature (default: 1.0 from trained model)
        ecs_temperature: Temperature for profile similarity (default: 5.0)
        arena_size_mb: Memory arena size in MB (default: 200)
        thread_count: Number of threads (default: auto-detect)

    Returns:
        MSAResult object with aligned sequences

    Example:
        >>> # Encode structures
        >>> emb1 = pfalign.encode("protein1.pdb")
        >>> emb2 = pfalign.encode("protein2.pdb")
        >>> emb3 = pfalign.encode("protein3.pdb")
        >>>
        >>> # Compute distances and build custom tree
        >>> distances = pfalign.compute_distances([emb1, emb2, emb3])
        >>> tree = pfalign.tree.bionj(distances)  # or upgma/nj/mst
        >>>
        >>> # Progressive alignment with custom tree
        >>> msa = pfalign.progressive_align([emb1, emb2, emb3], tree)
        >>> msa.write_fasta("aligned.fasta")
    """
    # Extract arrays from inputs
    arrays = [_extract_array(entry) for entry in embeddings]

    return _align_cpp._progressive_align(
        arrays,
        tree,
        gap_open=gap_open,
        gap_extend=gap_extend,
        temperature=temperature,
        ecs_temperature=ecs_temperature,
        arena_size_mb=arena_size_mb,
        thread_count=thread_count,
    )


def reformat(
    input_file: str | os.PathLike,
    output_file: str | os.PathLike,
    *,
    input_format: str = "",
    output_format: str = "",
    match_mode: str = "",
    gap_threshold: int = 50,
    remove_inserts: bool = False,
    remove_gapped: int = 0,
    uppercase: bool = False,
    lowercase: bool = False,
    remove_secondary_structure: bool = False,
):
    """Convert between multiple sequence alignment formats.

    Supports conversion between FASTA, A2M, A3M, Stockholm, PSI-BLAST, and Clustal formats.
    Based on HHsuite's reformat.pl functionality.

    Args:
        input_file: Path to input alignment file
        output_file: Path to output alignment file
        input_format: Input format (auto-detect from extension if not specified).
            Supported: 'fas', 'a2m', 'a3m', 'sto', 'psi', 'clu'
        output_format: Output format (auto-detect from extension if not specified).
            Supported: 'fas', 'a2m', 'a3m', 'sto', 'psi', 'clu'
        match_mode: Match state assignment mode:
            - "": No match state assignment (default)
            - "first": Use first sequence to define match states
            - "gap": Use gap percentage threshold (requires gap_threshold)
        gap_threshold: Gap percentage threshold for match state assignment (0-100).
            Used when match_mode="gap". Columns with <threshold% gaps become match states.
        remove_inserts: Remove insert states (lowercase residues)
        remove_gapped: Remove columns with >=N% gaps (0-100, 0=disabled)
        uppercase: Convert all residues to uppercase
        lowercase: Convert all residues to lowercase
        remove_secondary_structure: Remove secondary structure annotations

    Format descriptions:
        - FASTA (fas): Standard aligned FASTA format
        - A2M (a2m): Match/insert states (uppercase=match, lowercase=insert, '.'=insert-gap)
        - A3M (a3m): Compressed A2M (insert-gaps omitted, variable-length sequences)
        - Stockholm (sto): HMMER/Pfam format with header/footer
        - PSI-BLAST (psi): PSI-BLAST input format
        - Clustal (clu): Clustal alignment format

    Examples:
        >>> # Convert FASTA to A3M (auto-detect formats)
        >>> reformat("alignment.fas", "alignment.a3m")

        >>> # Convert A2M to Stockholm with match states from first sequence
        >>> reformat("input.a2m", "output.sto", match_mode="first")

        >>> # Convert with gap threshold: columns with <50% gaps = match states
        >>> reformat("input.fas", "output.a2m", match_mode="gap", gap_threshold=50)

        >>> # Remove gapped columns and convert to uppercase
        >>> reformat("input.a3m", "output.fas", remove_gapped=90, uppercase=True)

        >>> # Remove insert states (lowercase) from A2M
        >>> reformat("input.a2m", "output.fas", remove_inserts=True)
    """
    return _align_cpp._reformat(
        str(input_file),
        str(output_file),
        inform=input_format,
        outform=output_format,
        match_mode=match_mode,
        gap_threshold=gap_threshold,
        remove_inserts=remove_inserts,
        remove_gapped=remove_gapped,
        uppercase=uppercase,
        lowercase=lowercase,
        remove_ss=remove_secondary_structure,
    )


def info(
    path: str | os.PathLike,
    *,
    show_chains: bool = False,
) -> dict:
    """Get information about a structure file or directory.

    Inspects structure files to show chain information, residue counts,
    and atom counts without running alignment. For directories, lists
    all available structure files.

    Args:
        path: Path to structure file or directory
        show_chains: Include detailed chain information

    Returns:
        Dictionary with structure or directory information:
        - For structure file: {
            'path': str,
            'format': str,
            'num_chains': int,
            'chains': List[dict] (if show_chains=True)
          }
        - For directory: {
            'path': str,
            'is_directory': True,
            'num_structures': int,
            'structures': List[str]
          }

    Examples:
        >>> # Inspect structure file
        >>> info = pfalign.info("protein.pdb")
        >>> print(f"Found {info['num_chains']} chains")

        >>> # Show detailed chain info
        >>> info = pfalign.info("protein.pdb", show_chains=True)
        >>> for chain in info['chains']:
        ...     print(f"Chain {chain['id']}: {chain['length']} residues")

        >>> # List directory
        >>> info = pfalign.info("proteins/")
        >>> print(f"Found {info['num_structures']} structures")
    """
    path_obj = Path(path)

    # Check if directory
    if path_obj.is_dir():
        structures = []
        for item in path_obj.iterdir():
            if item.is_file() and item.suffix.lower() in _STRUCTURE_EXTENSIONS:
                structures.append(str(item))
        structures.sort()

        return {
            'path': str(path),
            'is_directory': True,
            'num_structures': len(structures),
            'structures': structures,
        }

    # Structure file - get chain info from C++
    chain_info = _align_cpp._get_structure_info(str(path), show_chains)
    return chain_info


def version_info() -> dict:
    """Get detailed version and build information.

    Returns comprehensive version information including build type,
    git commit, compiler details, and backend capabilities.

    Returns:
        Dictionary with version information:
        {
            'version': str,
            'git_commit': str,
            'git_branch': str,
            'build_type': str,
            'compiler': str,
            'build_date': str,
            'python_version': str,
        }

    Example:
        >>> info = pfalign.version_info()
        >>> print(f"pfalign {info['version']}")
        >>> print(f"Git: {info['git_commit'][:7]} ({info['git_branch']})")
    """
    import sys
    import platform

    # Get C++ build info
    cpp_info = _align_cpp._version_info() if hasattr(_align_cpp, '_version_info') else {}

    return {
        'version': __version__,
        'git_commit': cpp_info.get('git_commit', 'unknown'),
        'git_branch': cpp_info.get('git_branch', 'unknown'),
        'build_type': cpp_info.get('build_type', 'unknown'),
        'compiler': cpp_info.get('compiler', 'unknown'),
        'build_date': cpp_info.get('build_date', 'unknown'),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.platform(),
    }


def batch_pairwise(
    queries: str | os.PathLike | Sequence[str],
    targets: str | os.PathLike | Sequence[str],
    *,
    output_dir: str | os.PathLike,
    mode: str = "all-vs-all",
    k_neighbors: int = 30,
    gap_open: float | None = None,
    gap_extend: float | None = None,
    temperature: float | None = None,
    output_format: str = "fas",
) -> List[dict]:
    """Batch pairwise alignments between query and target sets.

    Performs multiple pairwise alignments without shell scripts.
    Supports all-vs-all and one-to-one modes.

    Args:
        queries: Query structures (directory path, list file, or list of paths)
        targets: Target structures (directory path, list file, or list of paths)
        output_dir: Directory to save alignment results
        mode: Alignment mode:
            - "all-vs-all": Align every query against every target
            - "one-to-one": Align queries[i] with targets[i] (requires same count)
        k_neighbors: Number of nearest neighbors for graph construction
        gap_open: Gap opening penalty (default: -2.544 from trained model)
        gap_extend: Gap extension penalty (default: 0.194 from trained model)
        temperature: Softmax temperature (default: 1.0 from trained model)
        output_format: Output format for alignments (default: "fas")

    Returns:
        List of dictionaries with alignment information:
        [{
            'query': str,
            'target': str,
            'output': str,
            'score': float,
        }, ...]

    Examples:
        >>> # All-vs-all alignment
        >>> results = pfalign.batch_pairwise(
        ...     queries="query_proteins/",
        ...     targets="target_proteins/",
        ...     output_dir="alignments/",
        ...     mode="all-vs-all"
        ... )
        >>> print(f"Completed {len(results)} alignments")

        >>> # One-to-one alignment
        >>> results = pfalign.batch_pairwise(
        ...     queries=["p1.pdb", "p2.pdb"],
        ...     targets=["t1.pdb", "t2.pdb"],
        ...     output_dir="alignments/",
        ...     mode="one-to-one"
        ... )
    """
    # Parse inputs
    query_paths = _parse_msa_inputs(queries)
    target_paths = _parse_msa_inputs(targets)

    # Validate mode
    if mode not in ["all-vs-all", "one-to-one"]:
        raise ValueError(f"mode must be 'all-vs-all' or 'one-to-one', got '{mode}'")

    # Build pairs
    pairs = []
    if mode == "all-vs-all":
        for q in query_paths:
            for t in target_paths:
                pairs.append((q, t))
    elif mode == "one-to-one":
        if len(query_paths) != len(target_paths):
            raise ValueError(
                f"one-to-one mode requires equal number of queries and targets, "
                f"got {len(query_paths)} queries and {len(target_paths)} targets"
            )
        pairs = list(zip(query_paths, target_paths))

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process all pairs
    results = []
    progress_bar = _align_cpp.ProgressBar(len(pairs), "Aligning pairs", width=20)
    for idx, (query, target) in enumerate(pairs):
        # Generate output filename
        query_name = Path(query).stem
        target_name = Path(target).stem
        output_file = output_path / f"{query_name}_vs_{target_name}.{output_format}"

        # Run pairwise alignment
        result = pairwise(
            query,
            target,
            k_neighbors=k_neighbors,
            gap_open=gap_open,
            gap_extend=gap_extend,
            temperature=temperature,
        )

        # Save result
        result.save(str(output_file), format=output_format)

        # Record results
        results.append({
            'query': query,
            'target': target,
            'output': str(output_file),
            'score': result.score(),
        })

        # Update progress
        progress_bar.update(idx + 1)

    progress_bar.finish()
    return results
