"""Example demonstrating PyTorch workflow integration.

This example shows how to use train_torch_model and evaluate_torch_model
workflow steps to train and evaluate a PyTorch model with automatic caching.

To run this example:
    python examples/torch_workflow_example.py
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from formed.integrations.ml.metrics import Average
from formed.integrations.torch import (
    BaseTorchModel,
    DataLoader,
    SingleDeviceDistributor,
    TorchTrainer,
    evaluate_torch_model,
    train_torch_model,
)
from formed.integrations.torch.types import IEvaluator


class SimpleRegressor(BaseTorchModel):
    """Simple regression model for demonstration."""

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


class RegressionEvaluator(IEvaluator):
    """Evaluator that tracks MSE loss."""

    def __init__(self):
        self._loss_metric = Average("mse")

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
    print("=== PyTorch Workflow Example ===\n")

    # Create datasets
    train_dataset = create_dataset(num_samples=200, input_dim=10, seed=42)
    val_dataset = create_dataset(num_samples=50, input_dim=10, seed=100)
    test_dataset = create_dataset(num_samples=50, input_dim=10, seed=200)

    # Create data loaders
    train_loader = DataLoader(batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Create a model instance for optimizer initialization
    temp_model = SimpleRegressor(input_dim=10, hidden_dim=32)

    # Create trainer
    trainer = TorchTrainer(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=torch.optim.Adam(temp_model.parameters(), lr=1e-3),
        distributor=SingleDeviceDistributor(),
        max_epochs=10,
        eval_strategy="epoch",
        logging_strategy="epoch",
    )

    # Step 1: Train model using workflow step
    print("Step 1: Training model with workflow step...")
    print("(This uses automatic caching - rerunning with same config will load from cache)")

    model = SimpleRegressor(input_dim=10, hidden_dim=32)

    trained_model = train_torch_model(
        model=model, trainer=trainer, train_dataset=train_dataset, val_dataset=val_dataset, random_seed=42
    )

    print(f"✓ Model trained: {type(trained_model).__name__}\n")

    # Step 2: Evaluate on test set using workflow step
    print("Step 2: Evaluating model with workflow step...")

    evaluator = RegressionEvaluator()
    test_metrics = evaluate_torch_model(
        model=trained_model, evaluator=evaluator, dataset=test_dataset, dataloader=test_loader
    )

    print(f"✓ Test metrics: {test_metrics}\n")

    # Step 3: Demonstrate caching behavior
    print("Step 3: Demonstrating caching...")
    print("Running train_torch_model again with same configuration...")

    model_2 = SimpleRegressor(input_dim=10, hidden_dim=32)
    trained_model_2 = train_torch_model(
        model=model_2, trainer=trainer, train_dataset=train_dataset, val_dataset=val_dataset, random_seed=42
    )

    print("✓ Second training completed (should be much faster due to caching!)\n")

    # Verify models are equivalent
    are_equal = all(
        torch.allclose(p1, p2) for p1, p2 in zip(trained_model.parameters(), trained_model_2.parameters(), strict=False)
    )
    print(f"✓ Models are equivalent: {are_equal}")

    print("\n=== Workflow Example Complete! ===")


if __name__ == "__main__":
    main()
