import torch
import torch.nn as nn
from src.models.encoder import TensorSliceEncoder


class AmortizedCPTensor(nn.Module):
    """
    The full Scalable Bayesian Tensor Factorization Model.
    """

    def __init__(self, num_genes, num_cells, num_spatial, latent_rank, hidden_dim=128):
        super(AmortizedCPTensor, self).__init__()

        self.latent_rank = latent_rank

        # 1. The Local Amortized Encoder (Learns Cell Factors: B)
        # We input the gene expression profile for a cell
        self.encoder = TensorSliceEncoder(input_dim=num_genes,
                                          hidden_dim=hidden_dim,
                                          latent_rank=latent_rank)

        # 2. The Global Learnable Factors (Genes: A, Spatial: C)
        # Unlike cells, we treat these as global parameters for the whole tissue
        self.global_A = nn.Parameter(torch.randn(num_genes, latent_rank))
        self.global_C = nn.Parameter(torch.randn(num_spatial, latent_rank))

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick: z = mu + std * epsilon
        """
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * logvar)
        # Sample random noise (epsilon)
        eps = torch.randn_like(std)
        # Compute the sampled latent variable
        return mu + eps * std

    def decode(self, z_B):
        """
        Reconstructs the tensor slice using the CP Decomposition logic.
        z_B: The sampled latent factors for a batch of cells [Batch, Rank]
        A: Global gene factors [Genes, Rank]
        C: Global spatial factors [Spatial, Rank]
        """
        # We want to reconstruct a slice: [Batch_Cells, Genes, Spatial]
        # Using Einstein Summation: 'br' (Cells), 'ir' (Genes), 'kr' (Spatial) -> 'bik'
        X_reconstructed = torch.einsum('br,ir,kr->bik', z_B, self.global_A, self.global_C)
        return X_reconstructed

    def forward(self, x_cell_data):
        """
        The full forward pass: Encode -> Sample -> Decode
        """
        # 1. Predict distributions
        mu, logvar = self.encoder(x_cell_data)

        # 2. Sample latent factors using the trick
        z_B = self.reparameterize(mu, logvar)

        # 3. Reconstruct the biological data
        X_hat = self.decode(z_B)

        return X_hat, mu, logvar


if __name__ == "__main__":
    # --- System Test ---
    NUM_GENES = 500
    NUM_CELLS = 200
    NUM_SPATIAL = 50
    RANK = 5
    BATCH_SIZE = 32  # We process 32 cells at a time to save memory

    # Initialize the full framework
    model = AmortizedCPTensor(num_genes=NUM_GENES,
                              num_cells=NUM_CELLS,
                              num_spatial=NUM_SPATIAL,
                              latent_rank=RANK)

    # Create fake gene expression data for 32 cells
    mock_cell_input = torch.randn(BATCH_SIZE, NUM_GENES)

    # Push it through the pipeline
    reconstructed_tensor, mu, logvar = model(mock_cell_input)

    print("--- Architecture Verification ---")
    print(f"Input Cell Data: {mock_cell_input.shape}")
    print(f"Sampled Latent Means: {mu.shape}")
    print(f"Reconstructed Tensor Slice: {reconstructed_tensor.shape}")
    # Expected output: [32, 500, 50] -> (Batch Cells, Genes, Spatial Zones)