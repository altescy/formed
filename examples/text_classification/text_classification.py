import dataclasses
import random
from collections.abc import Sequence
from typing import Any, Generic

import jax
import optax
from flax import nnx, struct
from typing_extensions import TypeVar

from formed.integrations import flax as xf
from formed.integrations import ml
from formed.integrations.flax import modules as xfm
from formed.integrations.ml import types as mlt
from formed.workflow import step

InputT = TypeVar(
    "InputT",
    default=Any,
)
TextTransformT_co = TypeVar(
    "TextTransformT_co",
    bound=ml.BaseTransform[str | Sequence[str] | mlt.AnalyzedText],
    default=Any,
    covariant=True,
)


@dataclasses.dataclass
class ClassificationExample:
    id: str
    text: str | Sequence[str]
    label: int | str | None = None


class TextClassificationDataModule(
    ml.DataModule[
        mlt.DataModuleModeT,
        InputT,
        "TextClassificationDataModule[mlt.AsInstance, InputT, TextTransformT_co]",
        "TextClassificationDataModule[mlt.AsBatch, InputT, TextTransformT_co]",
    ],
    Generic[mlt.DataModuleModeT, InputT, TextTransformT_co],
):
    id: ml.MetadataTransform[Any, str]
    text: TextTransformT_co
    label: ml.Extra[ml.LabelIndexer] = ml.Extra.default()


@struct.dataclass
class ClassifierOutput:
    probs: jax.Array
    label: jax.Array
    loss: jax.Array | None = None


class TextClassifier(xf.BaseFlaxModel[TextClassificationDataModule[mlt.AsBatch], ClassifierOutput]):
    def __init__(
        self,
        num_classes: int,
        embedder: xfm.BaseEmbedder[mlt.BatchT],
        vectorizer: xfm.BaseSequenceVectorizer,
        encoder: xfm.BaseSequenceEncoder | None = None,
        feedforward: xfm.FeedForward | None = None,
        dropout: float = 0.1,
    ) -> None:
        rngs = xf.require_rngs()

        embedding_dim = embedder.get_output_dim()
        vector_dim = vectorizer.get_output_dim()
        encoding_dim = encoder.get_output_dim() if encoder is not None else None
        feedforward_dim = feedforward.get_output_dim() if feedforward is not None else None
        feature_dim = feedforward_dim or vector_dim or encoding_dim or embedding_dim

        self._embedder = embedder
        self._vectorizer = vectorizer
        self._encoder = encoder
        self._feedforward = feedforward
        self._dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
        self._classifier = nnx.Linear(feature_dim, num_classes, rngs=rngs)

    def __call__(
        self,
        inputs: TextClassificationDataModule[mlt.AsBatch],
        params: None = None,
    ) -> ClassifierOutput:
        embeddings, mask = self._embedder(inputs.text)

        if self._encoder is not None:
            embeddings = self._encoder(embeddings, mask=mask)

        features = self._vectorizer(embeddings, mask=mask)

        if self._feedforward is not None:
            features = self._feedforward(features)

        if self._dropout is not None:
            features = self._dropout(features)

        logits = self._classifier(features)
        probs = jax.nn.softmax(logits, axis=-1)
        label = probs.argmax(axis=-1)

        loss: jax.Array | None = None
        if inputs.label is not None:
            one_hot_labels = jax.nn.one_hot(inputs.label, logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()

        return ClassifierOutput(probs=probs, label=label, loss=loss)


class ClassificationEvaluator:
    def __init__(self, metrics: Sequence[ml.MulticlassClassificationMetric]) -> None:
        self._loss = ml.Average("loss")
        self._metrics = metrics

    def update(self, inputs: TextClassificationDataModule[mlt.AsBatch], output: ClassifierOutput) -> None:
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


@step
def generate_sort_detection_dataset(
    vocab: Sequence[str] = "abcdefghijklmnopqrstuvwxyz",
    num_examples: int = 100,
    max_tokens: int = 10,
    random_seed: int = 42,
) -> list[ClassificationExample]:
    rng = random.Random(random_seed)
    examples = []
    for _ in range(num_examples):
        num_tokens = rng.randint(1, max_tokens)
        label = rng.choice(["sorted", "not_sorted"])
        tokens = rng.choices(vocab, k=num_tokens)
        if label == "sorted":
            tokens.sort()
        examples.append(ClassificationExample(id=str(len(examples)), text=tokens, label=label))
    return examples
