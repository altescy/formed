"""Example of using DistributedDataParallel (DDP) with TorchTrainer.

This example shows how to use DDP for multi-GPU training with the formed framework.
It demonstrates a simple multi-class classification task with synthetic data.

To run this example with 2 GPUs:
    torchrun --nproc_per_node=2 examples/torch_ddp_example.py

Or with 4 GPUs:
    torchrun --nproc_per_node=4 examples/torch_ddp_example.py

For CPU-only testing (using gloo backend):
    torchrun --nproc_per_node=2 examples/torch_ddp_example.py --backend=gloo

The torchrun command automatically sets the required environment variables:
    - RANK: Global rank of the process
    - LOCAL_RANK: Local rank on this machine
    - WORLD_SIZE: Total number of processes
    - MASTER_ADDR: Address of master node
    - MASTER_PORT: Port of master node
"""

import argparse
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from formed.integrations.torch import BaseTorchModel, DataLoader, DistributedDataParallelDistributor, TorchTrainer


class SimpleClassifier(BaseTorchModel):
    """Simple multi-layer classifier for demonstration."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs, params=None):
        """Forward pass.

        Args:
            inputs: Dict with 'features' (batch_size, input_dim) and optional 'labels' (batch_size,)
            params: Unused model parameters

        Returns:
            Dict with 'logits' and 'loss' (if labels provided)
        """
        logits = self.layers(inputs["features"])
        loss: Optional[torch.Tensor] = None
        if inputs.get("labels") is not None:
            loss = nn.functional.cross_entropy(logits, inputs["labels"])
        return {"logits": logits, "loss": loss}


def create_synthetic_dataset(num_samples: int, input_dim: int, num_classes: int, seed: int = 42):
    """Create a synthetic classification dataset.

    Args:
        num_samples: Number of samples to generate
        input_dim: Feature dimension
        num_classes: Number of classes
        seed: Random seed

    Returns:
        List of dicts with 'features' and 'labels'
    """
    rng = np.random.default_rng(seed)

    # Generate class centers
    centers = rng.normal(size=(num_classes, input_dim)) * 3.0

    dataset = []
    for _ in range(num_samples):
        # Random class
        label = rng.integers(0, num_classes)

        # Sample from class center with noise
        features = centers[label] + rng.normal(size=input_dim)

        dataset.append({"features": features.astype(np.float32), "labels": label})

    return dataset


def collate_fn(batch):
    """Collate function to batch samples.

    Args:
        batch: List of sample dicts

    Returns:
        Batched dict with tensors
    """
    features = torch.tensor(np.stack([item["features"] for item in batch]))
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return {"features": features, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="DDP training example")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend: nccl (GPU) or gloo (CPU)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per process")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Initialize DDP distributor
    # Environment variables are automatically set by torchrun
    distributor = DistributedDataParallelDistributor(
        backend=args.backend,  # Use nccl for GPU, gloo for CPU
        init_method="env://",
    )

    if distributor.is_main_process:
        print(f"Starting DDP training with {distributor.world_size} processes")
    print(f"Process rank {distributor.rank}/{distributor.world_size} on device {distributor.device}")

    # Create synthetic datasets
    # Use different seeds per process to ensure different data splits
    train_dataset = create_synthetic_dataset(num_samples=1000, input_dim=20, num_classes=5, seed=42 + distributor.rank)
    val_dataset = create_synthetic_dataset(num_samples=200, input_dim=20, num_classes=5, seed=100)

    if distributor.is_main_process:
        print(f"Dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    # Create model
    model = SimpleClassifier(input_dim=20, hidden_dim=64, num_classes=5)
    if distributor.is_main_process:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders using formed's DataLoader helper
    train_dataloader = DataLoader(batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Create trainer with DDP distributor
    trainer = TorchTrainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=partial(torch.optim.Adam, lr=args.lr),
        distributor=distributor,  # Use DDP distributor
        max_epochs=args.epochs,
        eval_strategy="epoch",
        logging_strategy="step",
        logging_interval=10,
    )

    # Train model (DDP handles gradient synchronization automatically)
    # Note: Trainer automatically handles cleanup of distributor resources
    if distributor.is_main_process:
        print("\nStarting training...")

    state = trainer.train(model, train_dataset, val_dataset)

    # Only rank 0 reports final results
    if distributor.is_main_process:
        print(f"\nTraining completed! Total steps: {state.step}")
        print("Final model can be saved from state.model.state_dict()")


if __name__ == "__main__":
    main()
