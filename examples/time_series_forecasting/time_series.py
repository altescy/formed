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

import numpy
import torch
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
    sequence: ml.TensorTransform
    target: ml.Extra[ml.ScalarTransform] = ml.Extra.default()


@dataclasses.dataclass
class ForecastOutput:
    """Output from time series forecasting model."""

    predictions: torch.Tensor
    loss: torch.Tensor | None = None


class TimeSeriesForecaster(ft.BaseTorchModel[TimeSeriesDataModule[mlt.AsBatch], ForecastOutput]):
    """Time series forecasting model."""

    def __init__(
        self,
        encoder: ftm.BaseSequenceEncoder,
        vectorizer: ftm.BaseSequenceVectorizer,
        loss: ftm.BaseRegressionLoss | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        feature_dim = ft.determine_ndim(
            encoder.get_output_dim(),
            vectorizer.get_output_dim(),
        )

        self._encoder = encoder
        self._vectorizer = vectorizer
        self._dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        self._predictor = torch.nn.Linear(feature_dim, 1)
        self._loss = loss or ft.MeanSquaredErrorLoss()

    def forward(
        self,
        inputs: TimeSeriesDataModule[mlt.AsBatch],
        params: None = None,
    ) -> ForecastOutput:
        # Convert inputs to tensors
        sequence = ft.ensure_torch_tensor(inputs.sequence)
        if sequence.ndim == 2:
            # Add feature dimension: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
            sequence = sequence[:, :, None]

        # Encode sequence
        encodings = self._encoder(sequence)

        # Vectorize encodings
        features = self._vectorizer(encodings)

        # Apply dropout if specified
        if self._dropout is not None:
            features = self._dropout(features)

        # Predict next value
        predictions = self._predictor(features).squeeze(-1)

        # Calculate loss if target is provided
        loss: torch.Tensor | None = None
        if inputs.target is not None:
            loss = self._loss(predictions, inputs.target)

        return ForecastOutput(predictions=predictions, loss=loss)


class ForecastingEvaluator(ftt.IEvaluator[TimeSeriesDataModule[mlt.AsBatch], ForecastOutput]):
    """Evaluator for time series forecasting."""

    def __init__(self):
        self._loss = ml.Average("loss")
        self._mae = ml.MeanAbsoluteError()

    def update(self, inputs: TimeSeriesDataModule[mlt.AsBatch], output: ForecastOutput) -> None:
        if output.loss is not None:
            self._loss.update([output.loss.item()])
        if inputs.target is not None:
            self._mae.update(
                self._mae.Input(
                    predictions=output.predictions.detach().cpu().numpy().tolist(),
                    targets=inputs.target.tolist(),
                )
            )

    def compute(self) -> dict[str, float]:
        return {
            **self._loss.compute(),
            **self._mae.compute(),
        }

    def reset(self) -> None:
        self._loss.reset()
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
    rng = numpy.random.default_rng(random_seed)
    examples = []

    for i in range(num_examples):
        # Random frequency, phase, and amplitude
        frequency = rng.choice(numpy.linspace(0.5, 3.0, num_frequencies))
        phase = rng.uniform(0, 2 * math.pi)
        amplitude = rng.uniform(0.5, 2.0)

        # Generate sequence + 1 extra point for target
        t = numpy.arange(sequence_length + 1)
        values = amplitude * numpy.sin(2 * math.pi * frequency * t / sequence_length + phase)

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
        encoder=ftm.LSTMSequenceEncoder(
            input_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=False,
        ),
        vectorizer=ftm.BagOfEmbeddingsSequenceVectorizer(pooling="last"),
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
        engine=ft.DefaultTorchTrainingEngine(
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        ),
        distributor=ft.SingleDeviceDistributor(),
        max_epochs=args.epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        callbacks=[
            ft.EvaluationCallback(evaluator),
            ft.EarlyStoppingCallback(patience=5, metric="-mean_absolute_error"),
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
