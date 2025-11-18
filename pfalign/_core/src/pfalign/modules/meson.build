# SoftAlign v2 Modules
# High-level components built from primitives

# MSA (Multiple Sequence Alignment) - needed by similarity
subdir('msa')

# MPNN encoder
subdir('mpnn')

# Similarity computation (depends on msa)
subdir('similarity')

# Alignment computation
subdir('alignment')

# Decode module (MAP decoding from posteriors)
subdir('decode')

# Pairwise alignment pipeline
subdir('pairwise')

# Combined modules library
modules_dep = declare_dependency(
  dependencies: [
    mpnn_dep,
    similarity_dep,
    alignment_dep,
    decode_dep,
    pairwise_dep,
    msa_dep,
  ],
)
