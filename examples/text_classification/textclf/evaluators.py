from collections.abc import Sequence

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt

from .datamodules import TextClassificationDataModule
from .flax import FlaxClassifierOutput
from .torch import TorchClassifierOutput


class ClassificationEvaluator:
    def __init__(
        self,
        metrics: Sequence[ml.MulticlassClassificationMetric],
    ) -> None:
        self._loss = ml.Average("loss")
        self._metrics = metrics

    def update(
        self,
        inputs: TextClassificationDataModule[mlt.AsBatch],
        output: FlaxClassifierOutput | TorchClassifierOutput,
    ) -> None:
        if output.loss is not None:
            self._loss.update([output.loss.item()])
        if inputs.label is not None:
            predictions = output.label.tolist()
            targets = inputs.label.tolist()
            for metric in self._metrics:
                metric.update(metric.Input(predictions=predictions, targets=targets))

    def compute(self) -> dict[str, float]:
        metrics = self._loss.compute()
        for metric in self._metrics:
            metrics.update(metric.compute())
        return metrics

    def reset(self) -> None:
        self._loss.reset()
        for metric in self._metrics:
            metric.reset()
