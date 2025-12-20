from collections.abc import Sequence
from copy import deepcopy

import numpy

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt

from .datamodules import TextClassificationDataModule
from .flax import FlaxClassifierOutput
from .torch import TorchClassifierOutput


class ClassificationEvaluator:
    def __init__(
        self,
        metrics: Sequence[ml.MulticlassClassificationMetric],
        label_metrics: Sequence[ml.BinaryClassificationMetric] = (),
        datamodule: TextClassificationDataModule[mlt.AsConverter] | None = None,
    ) -> None:
        assert not label_metrics or (datamodule is not None and datamodule.label), (
            "Label metrics require a datamodule to provide label information."
        )
        self._loss = ml.Average("loss")
        self._metrics = metrics
        self._datamodule = datamodule
        self._label_metrics: dict[mlt.Label, Sequence[ml.BinaryClassificationMetric]] = {}
        if datamodule is not None and datamodule.label:
            for label in datamodule.label.labels:
                self._label_metrics[label] = deepcopy(label_metrics)

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
            if self._label_metrics and self._datamodule is not None and self._datamodule.label:
                labels_pred = self._datamodule.label.reconstruct(numpy.array(predictions))
                labels_true = self._datamodule.label.reconstruct(numpy.array(targets))
                for label, metrics in self._label_metrics.items():
                    binarized_pred = [lbl == label for lbl in labels_pred]
                    binarized_true = [lbl == label for lbl in labels_true]
                    for metric in metrics:
                        metric.update(
                            metric.Input(
                                predictions=binarized_pred,
                                targets=binarized_true,
                            )
                        )

    def compute(self) -> dict[str, float]:
        metrics = self._loss.compute()
        for metric in self._metrics:
            metrics.update(metric.compute())
        for label, metrics_list in self._label_metrics.items():
            for metric in metrics_list:
                label_metric = metric.compute()
                for key, value in label_metric.items():
                    metrics[f"{label}/{key}"] = value
        return metrics

    def reset(self) -> None:
        self._loss.reset()
        for metric in self._metrics:
            metric.reset()
