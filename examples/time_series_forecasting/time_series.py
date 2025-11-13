"""Time series forecasting example using PyTorch and formed workflow.

This example demonstrates:
1. Generating synthetic time series data (sin waves with noise)
2. Training an LSTM model to predict future values
3. Using formed's workflow system with automatic caching
4. Model evaluation and visualization

Task: Given a sequence of time series values, predict the next value.
"""

import dataclasses
import math
import random
from typing import Any, Generic

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import TypeVar

from formed import workflow
from formed.integrations import ml
from formed.integrations import torch as ft
from formed.integrations.ml import types as mlt
from formed.integrations.torch import modules as ftm
from formed.integrations.torch import types as ftt

InputT = TypeVar(
    "InputT",
    default=Any,
)


@dataclasses.dataclass
class TimeSeriesExample:
    """Example for time series forecasting."""

    id: str
    sequence: list[float]  # Input sequence
    target: float  # Target value to predict


class TimeSeriesDataModule(
    ml.DataModule[
        mlt.DataModuleModeT,
        InputT,
        "TimeSeriesDataModule[mlt.AsInstance, InputT]",
        "TimeSeriesDataModule[mlt.AsBatch, InputT]",
    ],
    Generic[mlt.DataModuleModeT, InputT],
):
    """Data module for time series data."""

    id: ml.MetadataTransform[Any, str]
    sequence: ml.Extra[ml.TensorTransform] = ml.Extra.default()
    target: ml.Extra[ml.ScalarTransform] = ml.Extra.default()


@dataclasses.dataclass
class ForecastOutput:
    """Output from time series forecasting model."""

    predictions: torch.Tensor
    loss: torch.Tensor | None = None


class TimeSeriesForecaster(ft.BaseTorchModel[TimeSeriesDataModule[mlt.AsBatch], ForecastOutput]):
    """LSTM-based time series forecasting model."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.feedforward = ftm.FeedForward(
            input_dim=hidden_dim, hidden_dim=hidden_dim // 2, output_dim=1, num_layers=2, dropout=dropout
        )

    def forward(
        self,
        inputs: TimeSeriesDataModule[mlt.AsBatch],
        params: None = None,
    ) -> ForecastOutput:
        # Convert inputs to tensors
        if inputs.sequence is None:
            raise ValueError("inputs.sequence must not be None")
        sequence = ft.ensure_torch_tensor(inputs.sequence, dtype=torch.float32)
        if sequence.ndim == 2:
            # Add feature dimension: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
            sequence = sequence.unsqueeze(-1)

        # LSTM forward
        lstm_out, _ = self.lstm(sequence)

        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Predict next value
        predictions = self.feedforward(last_output).squeeze(-1)  # (batch_size,)

        # Calculate loss if target is provided
        loss: torch.Tensor | None = None
        if inputs.target is not None:
            target = ft.ensure_torch_tensor(inputs.target, dtype=torch.float32)
            loss = nn.functional.mse_loss(predictions, target)

        return ForecastOutput(predictions=predictions, loss=loss)


class ForecastingEvaluator(ftt.IEvaluator[TimeSeriesDataModule[mlt.AsBatch], ForecastOutput]):
    """Evaluator for time series forecasting."""

    def __init__(self):
        self._mse = ml.Average("mse")
        self._mae = ml.Average("mae")

    def update(self, inputs: TimeSeriesDataModule[mlt.AsBatch], output: ForecastOutput) -> None:
        if output.loss is not None:
            self._mse.update([output.loss.item()])
        if inputs.target is not None:
            predictions = output.predictions.detach().cpu().numpy()
            targets = ft.ensure_torch_tensor(inputs.target, dtype=torch.float32).cpu().numpy()
            mae = np.abs(predictions - targets).mean()
            self._mae.update([mae])

    def compute(self) -> dict[str, float]:
        metrics = self._mse.compute()
        metrics.update(self._mae.compute())
        return metrics

    def reset(self) -> None:
        self._mse.reset()
        self._mae.reset()


@workflow.step
def generate_sinusoid_dataset(
    num_examples: int = 1000,
    sequence_length: int = 20,
    num_frequencies: int = 3,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> list[TimeSeriesExample]:
    """Generate synthetic time series data based on sin waves.

    Args:
        num_examples: Number of examples to generate.
        sequence_length: Length of input sequence.
        num_frequencies: Number of different frequency patterns to generate.
        noise_level: Standard deviation of gaussian noise to add.
        random_seed: Random seed for reproducibility.

    Returns:
        List of time series examples.
    """
    rng = np.random.default_rng(random_seed)
    examples = []

    for i in range(num_examples):
        # Random frequency, phase, and amplitude
        frequency = rng.choice(np.linspace(0.5, 3.0, num_frequencies))
        phase = rng.uniform(0, 2 * math.pi)
        amplitude = rng.uniform(0.5, 2.0)

        # Generate sequence + 1 extra point for target
        t = np.arange(sequence_length + 1)
        values = amplitude * np.sin(2 * math.pi * frequency * t / sequence_length + phase)

        # Add noise
        values += rng.normal(0, noise_level, size=len(values))

        # Split into sequence and target
        sequence = values[:sequence_length].tolist()
        target = float(values[sequence_length])

        examples.append(TimeSeriesExample(id=f"example_{i}", sequence=sequence, target=target))

    return examples


def main():
    """Run the time series forecasting example directly (without workflow)."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    # Generate datasets
    print("Generating datasets...")
    train_data = generate_sinusoid_dataset(num_examples=1000, random_seed=42)
    val_data = generate_sinusoid_dataset(num_examples=200, random_seed=123)
    test_data = generate_sinusoid_dataset(num_examples=200, random_seed=456)

    # Create data module
    datamodule = TimeSeriesDataModule(
        id=ml.MetadataTransform(),
        sequence=ml.TensorTransform(),
        target=ml.ScalarTransform(),
    )

    # Create instances
    with datamodule.train():
        train_instances = [datamodule.instance(ex) for ex in train_data]
    val_instances = [datamodule.instance(ex) for ex in val_data]
    test_instances = [datamodule.instance(ex) for ex in test_data]

    # Create model
    print("Creating model...")
    model = TimeSeriesForecaster(
        input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
    )

    # Create data loaders using DataModule's batch method
    train_loader = ml.DataLoader(
        sampler=ml.BasicBatchSampler(batch_size=args.batch_size, shuffle=True, drop_last=True),
        collator=datamodule.batch,
    )
    val_loader = ml.DataLoader(
        sampler=ml.BasicBatchSampler(batch_size=args.batch_size, shuffle=False, drop_last=False),
        collator=datamodule.batch,
    )
    test_loader = ml.DataLoader(
        sampler=ml.BasicBatchSampler(batch_size=args.batch_size, shuffle=False, drop_last=False),
        collator=datamodule.batch,
    )

    # Create evaluator
    evaluator = ForecastingEvaluator()

    # Create trainer
    trainer = ft.TorchTrainer(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        distributor=ft.SingleDeviceDistributor(),
        max_epochs=args.epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        callbacks=[
            ft.EvaluationCallback(evaluator),
            ft.EarlyStoppingCallback(patience=5, metric="-mse"),
        ],
    )

    # Train model
    print(f"\nTraining model for {args.epochs} epochs...")
    state = trainer.train(model, train_instances, val_instances)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model = state.model
    model.eval()
    evaluator.reset()

    with torch.no_grad():
        for batch in test_loader(test_instances):
            output = model(batch)
            evaluator.update(batch, output)

    test_metrics = evaluator.compute()
    print(f"Test Metrics: {test_metrics}")

    # Show some predictions
    print("\nSample Predictions:")
    model.eval()
    sample_instances = random.sample(test_instances, min(5, len(test_instances)))

    with torch.no_grad():
        for instance in sample_instances:
            batch = datamodule.batch([instance])
            output = model(batch)
            pred = output.predictions.item()
            target = instance.target if hasattr(instance, "target") else None
            if target is not None and instance.sequence is not None:
                error = abs(pred - target)
                print(
                    f"  Sequence: [{', '.join(f'{v:.2f}' for v in instance.sequence[-5:])}] "
                    f"-> Prediction: {pred:.3f}, Target: {target:.3f}, Error: {error:.3f}"
                )


if __name__ == "__main__":
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    main()
