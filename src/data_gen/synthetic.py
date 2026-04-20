import torch


def generate_spatial_tensor(num_genes, num_cells, num_spatial, rank, seed=42):
    """Generates a synthetic 3D spatial transcriptomics tensor."""
    torch.manual_seed(seed)

    # 1. Generate the 'True' Latent Factors
    A_true = torch.randn(num_genes, rank)
    B_true = torch.randn(num_cells, rank)
    C_true = torch.randn(num_spatial, rank)

    # 2. Construct the Tensor using Einstein Summation
    # 'ir' = Genes, 'jr' = Cells, 'kr' = Spatial -> 'ijk'
    X_clean = torch.einsum('ir,jr,kr->ijk', A_true, B_true, C_true)

    # 3. Add biological noise
    X_observed = X_clean + 0.1 * torch.randn_like(X_clean)

    return X_observed, A_true, B_true, C_true


if __name__ == "__main__":
    # Simulate: 500 Genes, 200 Cells, 50 Spatial zones, Rank 5
    X_data, A, B, C = generate_spatial_tensor(500, 200, 50, rank=5)
    print(f"Success! Tensor built. Shape: {X_data.shape}")