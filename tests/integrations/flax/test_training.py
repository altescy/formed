import dataclasses
from typing import Generic, Optional

import jax
import numpy
import optax
import pytest
from flax import nnx, struct

from formed.integrations.flax import (
    BaseFlaxModel,
    EvaluationCallback,
    FlaxTrainer,
    determine_ndim,
    ensure_jax_array,
    require_rngs,
    use_rngs,
)
from formed.integrations.flax.modules import (
    AnalyzedTextEmbedder,
    BagOfEmbeddingsSequenceVectorizer,
    BaseSequenceVectorizer,
    TokenEmbedder,
)
from formed.integrations.ml import (
    BasicBatchSampler,
    DataLoader,
    DataModule,
    Extra,
    LabelIndexer,
    ScalarTransform,
    TensorTransform,
    Tokenizer,
    TokenSequenceIndexer,
)
from formed.integrations.ml.metrics import Average, MeanSquaredError, MulticlassAccuracy
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
class RegressorOutput:
    predictions: jax.Array
    loss: Optional[jax.Array] = None


class RegressionEvaluator:
    def __init__(self) -> None:
        self._loss = Average("loss")
        self._mse = MeanSquaredError()

    def update(self, inputs: RegressionDataModule[AsBatch], output: RegressorOutput) -> None:
        if output.loss is not None:
            self._loss.update([output.loss.item()])
        if inputs.target is not None:
            self._mse.update(
                self._mse.Input(
                    predictions=output.predictions.tolist(),
                    targets=inputs.target.tolist(),
                )
            )

    def compute(self) -> dict[str, float]:
        return {
            **self._loss.compute(),
            **self._mse.compute(),
        }

    def reset(self) -> None:
        self._loss.reset()
        self._mse.reset()


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
    class FlaxRegressor(BaseFlaxModel[RegressionDataModule[AsBatch], RegressorOutput]):
        def __init__(self, feature_dim: int) -> None:
            rngs = require_rngs()
            self.dense = nnx.Linear(in_features=feature_dim, out_features=1, rngs=rngs)

        def __call__(self, inputs: RegressionDataModule[AsBatch], params: None = None) -> RegressorOutput:
            del params
            preds = self.dense(ensure_jax_array(inputs.features)).squeeze(-1)
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

        with datamodule.train():
            train_instances = [datamodule.instance(example) for example in train_data]
        val_instances = [datamodule.instance(example) for example in val_data]

        train_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=True, drop_last=True),
            collator=datamodule.batch,
        )
        val_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=datamodule.batch,
        )

        with use_rngs(42):
            model = TestTrainingWithRegressor.FlaxRegressor(feature_dim=10)

            trainer = FlaxTrainer(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                max_epochs=5,
                callbacks=[EvaluationCallback(RegressionEvaluator())],
            )

            trainer.train(model, train_instances, val_instances)


@dataclasses.dataclass
class TextClassificationExample:
    text: str
    label: str


class TextClassificationDataModule(
    DataModule[
        DataModuleModeT,
        TextClassificationExample,
        "TextClassificationDataModule[AsInstance, TextClassificationExample]",
        "TextClassificationDataModule[AsBatch, TextClassificationExample]",
    ],
    Generic[DataModuleModeT],
):
    text: Tokenizer
    label: Extra[LabelIndexer] = Extra.default()


@struct.dataclass
class ClassifierOutput:
    probs: jax.Array
    label: jax.Array
    loss: Optional[jax.Array] = None


class ClassificationEvaluator:
    def __init__(self) -> None:
        self._loss = Average("loss")
        self._accuracy = MulticlassAccuracy()

    def update(self, inputs: TextClassificationDataModule[AsBatch], output: ClassifierOutput) -> None:
        if output.loss is not None:
            self._loss.update([output.loss.item()])
        if inputs.label is not None:
            self._accuracy.update(
                self._accuracy.Input(
                    predictions=output.label.tolist(),
                    targets=inputs.label.tolist(),
                )
            )

    def compute(self) -> dict[str, float]:
        return {
            **self._loss.compute(),
            **self._accuracy.compute(),
        }

    def reset(self) -> None:
        self._loss.reset()
        self._accuracy.reset()


@pytest.fixture
def text_classification_dataset() -> list[TextClassificationExample]:
    num_samples = 100
    rng = numpy.random.default_rng(42)
    common_words = ["the", "a", "an", "in", "on", "and", "is", "are", "was", "were"]
    sports_words = ["game", "team", "score", "player", "win", "lose", "match", "tournament", "coach", "league"]
    politics_words = [
        "election",
        "government",
        "policy",
        "vote",
        "senate",
        "law",
        "president",
        "campaign",
        "debate",
        "congress",
    ]
    science_words = [
        "research",
        "experiment",
        "data",
        "theory",
        "study",
        "scientist",
        "discovery",
        "analysis",
        "result",
        "method",
    ]

    labels = ["sports", "politics", "science"]
    dataset: list[TextClassificationExample] = []

    for _ in range(num_samples):
        label = rng.choice(labels)
        num_tokens = rng.integers(5, 15)
        tokens = rng.choice(common_words, size=num_tokens // 2).tolist()
        if label == "sports":
            tokens += rng.choice(sports_words, size=num_tokens // 2).tolist()
        elif label == "politics":
            tokens += rng.choice(politics_words, size=num_tokens // 2).tolist()
        else:
            tokens += rng.choice(science_words, size=num_tokens // 2).tolist()
        rng.shuffle(tokens)
        text = " ".join(tokens)
        dataset.append(TextClassificationExample(text=text, label=label))

    return dataset


class TestTrainingWithTextClassifier:
    class TextClassifier(BaseFlaxModel[TextClassificationDataModule[AsBatch], ClassifierOutput]):
        def __init__(
            self,
            num_classes: int,
            embedder: AnalyzedTextEmbedder,
            vectorizer: BaseSequenceVectorizer,
        ) -> None:
            rngs = require_rngs()
            self._num_classes = num_classes
            self._embedder = embedder
            self._vectorizer = vectorizer

            feature_dim = determine_ndim(
                self._embedder.get_output_dim(),
                self._vectorizer.get_output_dim(),
            )

            self._classifier = nnx.Linear(feature_dim, num_classes, rngs=rngs)

        def __call__(
            self,
            inputs: TextClassificationDataModule[AsBatch],
            params: None = None,
        ) -> ClassifierOutput:
            embeddings, mask = self._embedder(inputs.text)
            vectorized = self._vectorizer(embeddings, mask=mask)

            logits = self._classifier(vectorized)
            probs = jax.nn.softmax(logits, axis=-1)
            label = probs.argmax(axis=-1)

            loss: Optional[jax.Array] = None
            if inputs.label is not None:
                one_hot_labels = jax.nn.one_hot(inputs.label, self._num_classes)
                loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()

            return ClassifierOutput(probs=probs, label=label, loss=loss)

    def test_training_with_text_classifier(self, text_classification_dataset) -> None:
        train_data = text_classification_dataset[:80]
        val_data = text_classification_dataset[80:]

        datamodule = TextClassificationDataModule(
            text=Tokenizer(surfaces=TokenSequenceIndexer()),
            label=LabelIndexer(),
        )

        with datamodule.train():
            train_instances = [datamodule.instance(example) for example in train_data]
        val_instances = [datamodule.instance(example) for example in val_data]

        train_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=True, drop_last=True),
            collator=datamodule.batch,
        )
        val_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=datamodule.batch,
        )

        with use_rngs(42):
            model = self.TextClassifier(
                num_classes=datamodule.label.num_labels,
                embedder=AnalyzedTextEmbedder(
                    surface=TokenEmbedder(
                        vocab_size=datamodule.text.surfaces.vocab_size,
                        embedding_dim=64,
                    )
                ),
                vectorizer=BagOfEmbeddingsSequenceVectorizer(),
            )

            trainer = FlaxTrainer(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                max_epochs=5,
                callbacks=[EvaluationCallback(ClassificationEvaluator())],
            )

        trainer.train(model, train_instances, val_instances)
