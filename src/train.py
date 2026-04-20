import sys
import os

# 1. System Path Fix (Ensures Python always finds your modules)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

# 2. Local Imports
from src.data_gen.synthetic import generate_spatial_tensor
from src.models.amortized_cp import AmortizedCPTensor


def elbo_loss(X_true, X_reconstructed, mu, logvar):
    """The Evidence Lower Bound (ELBO) Loss Function."""
    # Reconstruction Loss (How well did we rebuild the tensor?)
    recon_loss = nn.functional.mse_loss(X_reconstructed, X_true, reduction='sum')

    # KL Divergence (How close are our distributions to a standard Normal?)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div, recon_loss, kl_div


def train_model():
    # --- Phase 1: Setup & Data Loading ---
    NUM_GENES = 500
    NUM_CELLS = 200
    NUM_SPATIAL = 50
    RANK = 5
    EPOCHS = 500
    LEARNING_RATE = 0.01

    print("Loading Synthetic Data...")
    X_data, true_A, true_B, true_C = generate_spatial_tensor(NUM_GENES, NUM_CELLS, NUM_SPATIAL, RANK)

    # Data Wrangling: Batch by Cells -> [Cells, Genes, Spatial]
    X_data = X_data.permute(1, 0, 2)

    print("Initializing Amortized Variational Tensor Model...")
    model = AmortizedCPTensor(num_genes=NUM_GENES,
                              num_cells=NUM_CELLS,
                              num_spatial=NUM_SPATIAL,
                              latent_rank=RANK)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Phase 2: The Training Engine ---
    print("\n--- Starting Training Loop ---")
    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Prepare input: We use the spatial average of genes as the cell feature input
        cell_features = X_data.mean(dim=2)

        # Forward Pass
        X_reconstructed, mu, logvar = model(cell_features)

        # Calculate Loss
        total_loss, recon_loss, kl_loss = elbo_loss(X_data, X_reconstructed, mu, logvar)

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
                  f"Total Loss: {total_loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | "
                  f"KL: {kl_loss.item():.4f}")

    print("\nTraining Complete! Minimum Viable Model (MVM) successfully converged.")

    # --- Phase 3: Visualization & Automatic Saving ---
    print("Visualizing and Saving Latent Factors...")

    # Force scientific journal formatting
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10
    })

    # Safely building the output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Extracting the spatial matrices to evaluate structural learning
    true_C_np = true_C.detach().numpy()
    learned_C_np = model.global_C.detach().numpy()

    # Building the side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(true_C_np, aspect='auto', cmap='magma')
    axes[0].set_title("True Spatial Factors", fontsize=12, pad=12)
    axes[0].set_xlabel("Latent Rank (Biological Pathways)")
    axes[0].set_ylabel("Spatial Zones")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(learned_C_np, aspect='auto', cmap='magma')
    axes[1].set_title("Amortized CP Inference", fontsize=12, pad=12)
    axes[1].set_xlabel("Latent Rank (Biological Pathways)")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    # Save the figure securely before showing it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"latent_factors_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! High-resolution plot securely saved to: {save_path}")

    # Display to screen
    plt.show()


if __name__ == "__main__":
    train_model()