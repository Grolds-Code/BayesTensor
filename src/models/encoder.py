import torch
import torch.nn as nn


class TensorSliceEncoder(nn.Module):
    """
    Amortized Inference Network.
    Takes a slice of the tensor (e.g., the gene expression profile of a single cell)
    and predicts the mean (mu) and log-variance (logvar) of its latent factors.
    """

    def __init__(self, input_dim, hidden_dim, latent_rank):
        super(TensorSliceEncoder, self).__init__()

        # Rationale: We use a simple Multi-Layer Perceptron (MLP) as our encoder.
        # It compresses the high-dimensional biological data into a smaller hidden space.
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalization helps Bayesian networks converge faster
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Rationale: In Bayesian ML, we don't output a single number.
        # We output a probability distribution. Therefore, the network splits into two heads:
        # One predicts the mean, the other predicts the uncertainty (log variance).
        self.fc_mu = nn.Linear(hidden_dim, latent_rank)
        self.fc_logvar = nn.Linear(hidden_dim, latent_rank)

    def forward(self, x):
        """
        x: A 2D matrix representing slices of the tensor (e.g., [Batch_Size, Genes])
        """
        hidden = self.shared_network(x)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar


if __name__ == "__main__":
    # --- Rationale Check: Let's test if the architecture compiles ---

    # Imagine we are looking at 1 batch of 32 cells.
    # Each cell has expression data for 500 genes.
    batch_size = 32
    num_genes = 500
    latent_rank = 5  # The 5 underlying biological pathways we are trying to find

    # Create fake data just to test the pipes
    mock_data_slice = torch.randn(batch_size, num_genes)

    # Initialize our model
    encoder = TensorSliceEncoder(input_dim=num_genes, hidden_dim=128, latent_rank=latent_rank)

    # Pass the data through the network
    mu, logvar = encoder(mock_data_slice)

    print(f"Input Shape: {mock_data_slice.shape}")
    print(f"Predicted Mean (mu) Shape: {mu.shape}")
    print(f"Predicted Log-Variance (logvar) Shape: {logvar.shape}")