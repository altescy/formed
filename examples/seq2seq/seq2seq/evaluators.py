import copy
from collections.abc import Sequence

import torch
from typing_extensions import Self

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt
from formed.integrations.torch.types import IEvaluator

from .datamodules import Seq2SeqDataModule
from .models import Seq2SeqTrainingOutput


class Seq2SeqEvaluator(IEvaluator[Seq2SeqDataModule[mlt.AsBatch], Seq2SeqTrainingOutput]):
    """Adapt teacher-forced model outputs to injected token-sequence metrics."""

    def __init__(self, target_pad_index: int, metrics: Sequence[ml.TokenSequenceMetric]) -> None:
        self._target_pad_index = target_pad_index
        self._metrics = metrics

    def update(
        self,
        inputs: Seq2SeqDataModule[mlt.AsBatch],
        output: Seq2SeqTrainingOutput,
    ) -> None:
        targets = torch.as_tensor(inputs.target.ids, device=output.predictions.device)[:, 1:]
        metric_input = ml.TokenSequenceInput(
            predictions=output.predictions.detach().cpu().tolist(),
            targets=targets.detach().cpu().tolist(),
            mask=targets.ne(self._target_pad_index).detach().cpu().tolist(),
            loss=output.loss.item(),
        )
        for metric in self._metrics:
            metric.update(metric_input)

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for metric in self._metrics:
            result.update(metric.compute())
        return result

    def reset(self) -> None:
        for metric in self._metrics:
            metric.reset()

    def clone(self) -> Self:
        return copy.deepcopy(self)
