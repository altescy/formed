"""Example of using EarlyStoppingCallback with DDP.

This example demonstrates how early stopping works with distributed training,
including model broadcasting across processes.

To run this example:
    # Single process (for testing)
    python examples/torch_early_stopping_example.py

    # Multiple processes with DDP
    torchrun --nproc_per_node=2 examples/torch_early_stopping_example.py --backend=gloo
"""

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from formed.integrations.ml.metrics import Average
from formed.integrations.torch import (
    BaseTorchModel,
    DataLoader,
    DefaultTorchTrainingEngine,
    DistributedDataParallelDistributor,
    EarlyStoppingCallback,
    EvaluationCallback,
    TorchTrainer,
)
from formed.integrations.torch.types import IEvaluator


class SimpleModel(BaseTorchModel):
    """Simple regression model."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs, params=None):
        preds = self.layers(inputs["features"]).squeeze(-1)
        loss: Optional[torch.Tensor] = None
        if inputs.get("labels") is not None:
            loss = nn.functional.mse_loss(preds, inputs["labels"])
        return {"predictions": preds, "loss": loss}


class SimpleEvaluator(IEvaluator):
    """Simple evaluator that tracks loss."""

    def __init__(self):
        self._loss_metric = Average("loss")

    def update(self, inputs, output):
        if output.get("loss") is not None:
            self._loss_metric.update([output["loss"].item()])

    def compute(self):
        return self._loss_metric.compute()

    def reset(self):
        self._loss_metric.reset()


def create_dataset(num_samples: int, input_dim: int, seed: int = 42):
    """Create synthetic regression dataset."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(input_dim,))
    bias = rng.normal()

    X = rng.normal(size=(num_samples, input_dim))
    y = X @ weights + bias + rng.normal(scale=0.1, size=(num_samples,))

    return [{"features": X[i].astype(np.float32), "labels": float(y[i])} for i in range(num_samples)]


def collate_fn(batch):
    """Collate function to batch samples."""
    features = torch.tensor(np.stack([item["features"] for item in batch]))
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float32)
    return {"features": features, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="Early stopping example")
    parser.add_argument("--backend", type=str, default="gloo", help="Backend: nccl or gloo")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    # Initialize distributor (will be single device if not using torchrun)
    import os

    if "RANK" in os.environ:
        distributor = DistributedDataParallelDistributor(backend=args.backend, init_method="env://")
        if distributor.is_main_process:
            print(f"Running with DDP: {distributor.world_size} processes")
    else:
        from formed.integrations.torch import SingleDeviceDistributor

        distributor = SingleDeviceDistributor()
        print("Running on single device")

    # Create datasets
    train_dataset = create_dataset(num_samples=200, input_dim=10)
    val_dataset = create_dataset(num_samples=50, input_dim=10, seed=100)

    # Create model
    model = SimpleModel(input_dim=10, hidden_dim=32)

    # Create data loaders
    train_loader = DataLoader(batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Create trainer with early stopping
    trainer = TorchTrainer(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        engine=DefaultTorchTrainingEngine(
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        ),
        distributor=distributor,
        max_epochs=100,  # Large number, will stop early
        eval_strategy="epoch",
        logging_strategy="epoch",
        callbacks=[
            EvaluationCallback(SimpleEvaluator()),
            EarlyStoppingCallback(patience=args.patience, metric="-loss"),
        ],
    )

    if distributor.is_main_process:
        print(f"\nTraining with early stopping (patience={args.patience})...")

    state = trainer.train(model, train_dataset, val_dataset)

    if distributor.is_main_process:
        print(f"\nTraining stopped at step {state.step}")
        print("Early stopping successfully triggered and best model was loaded!")


if __name__ == "__main__":
    main()
