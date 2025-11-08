import dataclasses
from collections.abc import Sequence
from typing import Generic, Optional

import jax
import numpy
import pytest
from flax import nnx, struct

from formed.integrations.flax import BaseFlaxModel, FlaxTrainer, ensure_jax_array
from formed.integrations.ml import BasicBatchSampler, DataLoader, DataModule, Extra, ScalarTransform, TensorTransform
from formed.integrations.ml.types import AsBatch, AsInstance, DataModuleModeT  # noqa: F401


@dataclasses.dataclass
class RegressionExample:
    features: numpy.ndarray
    target: float


class RegressionDataModule(
    DataModule[
        DataModuleModeT,
        RegressionExample,
        "RegressionDataModule[AsInstance, RegressionExample]",
        "RegressionDataModule[AsBatch, RegressionExample]",
    ],
    Generic[DataModuleModeT],
):
    features: TensorTransform
    target: Extra[ScalarTransform] = Extra.default()


@struct.dataclass
class RegressorInput:
    features: jax.Array
    target: Optional[jax.Array] = None


@struct.dataclass
class RegressorOutput:
    predictions: jax.Array
    loss: Optional[jax.Array] = None


@pytest.fixture
def regression_dataset() -> list[RegressionExample]:
    num_samples = 100
    feature_dim = 10

    rng = numpy.random.default_rng(42)
    weights = rng.normal(size=(feature_dim,))
    bias = rng.normal()

    X = rng.normal(size=(num_samples, feature_dim))
    y = X @ weights + bias + rng.normal(scale=0.1, size=(num_samples,))

    return [RegressionExample(features=X[i], target=y[i]) for i in range(num_samples)]


class TestTrainingWithRegressor:
    class FlaxRegressor(BaseFlaxModel[RegressorInput, RegressorOutput]):
        def __init__(self, rngs: nnx.Rngs, feature_dim: int) -> None:
            self.dense = nnx.Linear(in_features=feature_dim, out_features=1, rngs=rngs)

        def __call__(self, inputs: RegressorInput, params: None = None) -> RegressorOutput:
            del params
            preds = self.dense(inputs.features).squeeze(-1)
            loss: Optional[jax.Array] = None
            if inputs.target is not None:
                loss = jax.numpy.mean((preds - inputs.target) ** 2)
            return RegressorOutput(predictions=preds, loss=loss)

    @staticmethod
    def test_training_with_regressor(regression_dataset) -> None:
        train_data = regression_dataset[:80]
        val_data = regression_dataset[80:]

        datamodule = RegressionDataModule(
            features=TensorTransform(),
            target=ScalarTransform(),
        )

        def collator(batch: Sequence[RegressionDataModule[AsInstance]]) -> RegressorInput:
            inputs = datamodule.batch(batch)
            return RegressorInput(
                features=ensure_jax_array(inputs.features),
                target=ensure_jax_array(inputs.target) if inputs.target is not None else None,
            )

        with datamodule.train():
            train_instances = [datamodule.instance(example) for example in train_data]
        val_instances = [datamodule.instance(example) for example in val_data]

        train_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=True, drop_last=True),
            collator=collator,
        )
        val_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=collator,
        )

        rngs = nnx.Rngs(42)

        model = TestTrainingWithRegressor.FlaxRegressor(rngs=rngs, feature_dim=10)

        trainer = FlaxTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=5,
        )

        trainer.train(rngs, model, train_instances, val_instances)  # pyright: ignore[reportArgumentType]
