import sys
import os

# 1. System Path Fix (Ensures Python finds your root folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime

# 2. Local Imports
from src.real_data import fetch_real_biological_tensor
from src.models.amortized_cp import AmortizedCPTensor


def elbo_loss(X_true, X_reconstructed, mu, logvar):
    """The Evidence Lower Bound (ELBO) Loss Function."""
    recon_loss = nn.functional.mse_loss(X_reconstructed, X_true, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div, recon_loss, kl_div


def train_real_data():
    # --- Phase 1: Setup & Data Loading ---
    NUM_GENES = 500
    NUM_CELLS = 200
    NUM_SPATIAL = 50
    RANK = 5
    EPOCHS = 500
    LEARNING_RATE = 0.01

    print("--- INITIATING REAL DATA PIPELINE ---")

    # Fetch the Real Biological Matrix (Downloads from 10x Genomics)
    X_data = fetch_real_biological_tensor(NUM_GENES, NUM_CELLS, NUM_SPATIAL)

    # Data Wrangling: Batch by Cells -> [Cells, Genes, Spatial]
    X_data = X_data.permute(1, 0, 2)

    print("Initializing Amortized Variational Tensor Model...")
    model = AmortizedCPTensor(num_genes=NUM_GENES,
                              num_cells=NUM_CELLS,
                              num_spatial=NUM_SPATIAL,
                              latent_rank=RANK)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Tracking Arrays for the Plot ---
    history_recon = []
    history_kl = []

    # --- Phase 2: The Training Engine ---
    print("\n--- Starting Training Loop on Real Biology ---")
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

        # Save metrics for plotting
        history_recon.append(recon_loss.item())
        history_kl.append(kl_loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
                  f"Total: {total_loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | "
                  f"KL: {kl_loss.item():.4f}")

    print("\nTraining Complete! Model successfully adapted to real single-cell data.")

    # --- Phase 3: Convergence Visualization & Automatic Saving ---
    print("Generating and Saving Convergence Curves...")

    # Force scientific journal formatting
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10
    })

    # Safely build the output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Build the Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot Reconstruction Error
    ax.plot(history_recon, label="Reconstruction Error", color="#2c3e50", linewidth=2)

    # We scale KL up by 1000 just so it's visible on the same graph as the massive recon numbers
    ax.plot([kl * 1000 for kl in history_kl], label="KL Divergence (Scaled x1000)", color="#e74c3c", linewidth=2,
            linestyle="--")

    ax.set_title("Amortized Inference Convergence on Real scRNA-seq Data", pad=15)
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Loss Magnitude")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Save the figure securely before showing it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"real_data_convergence_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Convergence plot saved securely to: {save_path}")

    # Display to screen
    plt.show()


if __name__ == "__main__":
    train_real_data()