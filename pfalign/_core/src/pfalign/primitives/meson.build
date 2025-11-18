# SoftAlign v2 Primitives
# Low-level operations with backend support

# GEMM (Matrix multiply)
subdir('gemm')

# RBF (Radial Basis Function)
subdir('rbf')

# KNN (K-Nearest Neighbors)
subdir('knn')

# Layer Normalization
subdir('layer_norm')

# Gather operations
subdir('gather')

# Smith-Waterman (Dynamic Programming)
subdir('smith_waterman')

# Reduce operations (elementwise, sum, norm)
subdir('reduce')

# Alignment decoding (MAP from posteriors to discrete path)
subdir('alignment_decode')

# Structural alignment primitives
subdir('svd3x3')
subdir('kabsch')
subdir('structural_metrics')

# Combined primitives library (INTERFACE)
primitives_dep = declare_dependency(
  dependencies: [
    gemm_dep,
    rbf_dep,
    knn_dep,
    layer_norm_dep,
    gather_dep,
    reduce_dep,
    smith_waterman_dep,
    alignment_decode_dep,
    svd3x3_dep,
    kabsch_dep,
    structural_metrics_dep,
  ],
)
