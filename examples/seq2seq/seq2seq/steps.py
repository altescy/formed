import random
from collections.abc import Sequence
from typing import Annotated

import torch

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt
import formed.integrations.torch.modules as ftm
from formed import workflow
from formed.integrations.torch.types import IStreamingDataLoader

from .datamodules import Seq2SeqDataModule
from .models import LSTMSeq2SeqModel, ModelOutput, SamplingParams
from .types import Seq2SeqExample

WORDS = (
    "add",
    "build",
    "cache",
    "client",
    "config",
    "create",
    "data",
    "decode",
    "encode",
    "event",
    "fetch",
    "file",
    "find",
    "format",
    "get",
    "handler",
    "http",
    "index",
    "item",
    "json",
    "list",
    "load",
    "message",
    "model",
    "parse",
    "profile",
    "read",
    "request",
    "response",
    "save",
    "server",
    "token",
    "update",
    "url",
    "user",
    "value",
    "write",
)


@workflow.step("seq2seq::generate_camel_case_dataset")
def generate_camel_case_dataset(num_examples: int, random_seed: int) -> list[Seq2SeqExample]:
    rng = random.Random(random_seed)
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < num_examples:
        words = rng.sample(WORDS, rng.randint(2, 3))
        source = words[0] + "".join(word.capitalize() for word in words[1:])
        pairs.add((source, "_".join(words)))
    return [
        Seq2SeqExample(id=str(index), source=source, target=target)
        for index, (source, target) in enumerate(sorted(pairs))
    ]


@workflow.step("seq2seq::predict", format="json")
def predict(
    model: LSTMSeq2SeqModel,
    datamodule: Seq2SeqDataModule[mlt.AsConverter],
    dataset: Sequence[Seq2SeqExample],
    dataloader: IStreamingDataLoader[Seq2SeqExample, Seq2SeqDataModule[mlt.AsBatch]],
    sampler: ftm.BaseSequenceSampler[
        Seq2SeqDataModule[mlt.AsBatch], ModelOutput, SamplingParams, ftm.LSTMDecoderState, object
    ],
    print_results: bool = True,
) -> list[dict[str, str]]:
    model.eval()
    predictions: list[dict[str, str]] = []
    with torch.no_grad():
        for inputs in dataloader(dataset):
            output = sampler(model, inputs)
            sources = datamodule.source.reconstruct(inputs.source)
            targets = datamodule.target.reconstruct(inputs.target)
            generated = datamodule.target.reconstruct(output.best_sequences)
            for source_tokens, target_tokens, predicted_tokens in zip(sources, targets, generated):
                result = {
                    "source": source_tokens,
                    "target": target_tokens,
                    "prediction": predicted_tokens,
                }
                predictions.append(result)
                if print_results:
                    print(f"{result['source']:24s} -> {result['prediction']}")
    return predictions


@workflow.step("seq2seq::evaluate_generation", format="json")
def evaluate_generation(
    predictions: Sequence[dict[str, str]],
    metrics: Sequence[ml.MulticlassClassificationMetric[str]],
) -> Annotated[dict[str, float], workflow.WorkflowStepResultFlag.METRICS]:
    metric_input = ml.ClassificationInput(
        predictions=[result["prediction"] for result in predictions],
        targets=[result["target"] for result in predictions],
    )
    result: dict[str, float] = {}
    for metric in metrics:
        metric.reset()
        metric.update(metric_input)
        result.update(metric.compute())
    return result
