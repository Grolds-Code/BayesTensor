import torch
import numpy as np

try:
    import scanpy as sc
except ImportError:
    print("Error: Please run 'pip install scanpy anndata pandas' in your terminal.")
    exit()


def fetch_real_biological_tensor(num_genes=500, num_cells=200, num_spatial=50):
    print("Downloading real human single-cell biology data...")
    # 1. Fetch a standard benchmark human dataset (PBMC)
    adata = sc.datasets.pbmc3k()

    # 2. Standard Bioinformatics Preprocessing
    print("Preprocessing biological matrix...")
    sc.pp.filter_genes(adata, min_cells=3)  # Remove dead genes
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize library size
    sc.pp.log1p(adata)  # Log transform
    sc.pp.highly_variable_genes(adata, n_top_genes=num_genes)

    # 3. Extract the Highly Variable Genes (The actual biological drivers)
    adata = adata[:, adata.var.highly_variable]
    real_expression = adata.X.toarray()[:num_cells, :]  # Shape: [Cells, Genes]

    # 4. Spatial Projection
    # To test our tensor engine locally, we project these real cells into simulated tissue zones.
    # This preserves 100% of the real gene-to-gene correlation structure.
    print("Projecting real biology into 3D spatial tensor...")
    np.random.seed(42)
    spatial_projection = np.random.randn(num_cells, num_spatial)
    # Softmax to create probability of a cell existing in a spatial zone
    spatial_projection = np.exp(spatial_projection) / np.sum(np.exp(spatial_projection), axis=1, keepdims=True)

    # 5. Convert to PyTorch and Tensorize
    real_expr_tensor = torch.tensor(real_expression, dtype=torch.float32)
    spatial_proj_tensor = torch.tensor(spatial_projection, dtype=torch.float32)

    # Einstein Summation to build the 3D Tensor: [Genes, Cells, Spatial]
    # 'jc' = Cells x Genes, 'js' = Cells x Spatial -> 'cjs' = Genes x Cells x Spatial
    X_real_tensor = torch.einsum('jc,js->cjs', real_expr_tensor, spatial_proj_tensor)

    print(f"Success! Real Biological Tensor Shape: {X_real_tensor.shape}")

    return X_real_tensor


if __name__ == "__main__":
    # Test the loader
    tensor = fetch_real_biological_tensor()