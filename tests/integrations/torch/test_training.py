import dataclasses
from typing import Generic, Optional

import numpy
import pytest
import torch
import torch.nn as nn

from formed.integrations.ml import (
    BasicBatchSampler,
    DataLoader,
    DataModule,
    Extra,
    LabelIndexer,
    ScalarTransform,
    TensorTransform,
)
from formed.integrations.ml.metrics import Average, MeanSquaredError, MulticlassAccuracy
from formed.integrations.ml.types import AsBatch, AsInstance, DataModuleModeT  # noqa: F401
from formed.integrations.torch import (
    AnalyzedTextEmbedder,
    BagOfEmbeddingsSequenceVectorizer,
    BaseTorchModel,
    CrossEntropyLoss,
    EvaluationCallback,
    MeanSquaredErrorLoss,
    TokenEmbedder,
    TorchTrainer,
    ensure_torch_tensor,
)


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


@dataclasses.dataclass
class RegressorOutput:
    predictions: torch.Tensor
    loss: Optional[torch.Tensor] = None


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
                    predictions=output.predictions.detach().cpu().tolist(),
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
    class TorchRegressor(BaseTorchModel[RegressionDataModule[AsBatch], RegressorOutput, None]):
        def __init__(self, feature_dim: int) -> None:
            super().__init__()
            self.dense = nn.Linear(in_features=feature_dim, out_features=1)
            self.loss = MeanSquaredErrorLoss()

        def forward(self, inputs: RegressionDataModule[AsBatch], params: None = None) -> RegressorOutput:
            del params
            features = ensure_torch_tensor(inputs.features)
            preds = self.dense(features).squeeze(-1)
            loss: Optional[torch.Tensor] = None
            if inputs.target is not None:
                loss = self.loss(preds, inputs.target)
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

        model = TestTrainingWithRegressor.TorchRegressor(feature_dim=10)

        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
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
    from formed.integrations.ml import Tokenizer, TokenSequenceIndexer

    text: Tokenizer
    label: Extra[LabelIndexer] = Extra.default()


@dataclasses.dataclass
class ClassifierOutput:
    probs: torch.Tensor
    label: torch.Tensor
    loss: Optional[torch.Tensor] = None


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
                    predictions=output.label.detach().cpu().tolist(),
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
    class SimpleTextClassifier(BaseTorchModel[TextClassificationDataModule[AsBatch], ClassifierOutput, None]):
        def __init__(
            self,
            num_classes: int,
            vocab_size: int,
            embedding_dim: int = 64,
        ) -> None:
            super().__init__()
            self._num_classes = num_classes
            self._embedder = AnalyzedTextEmbedder(
                surface=TokenEmbedder(vocab_size=vocab_size, embedding_dim=embedding_dim),
            )
            self._vectorizer = BagOfEmbeddingsSequenceVectorizer()
            self._classifier = nn.Linear(embedding_dim, num_classes)
            self._loss = CrossEntropyLoss()

        def forward(
            self,
            inputs: TextClassificationDataModule[AsBatch],
            params: None = None,
        ) -> ClassifierOutput:
            # Simple bag-of-embeddings model
            embeddings, mask = self._embedder(inputs.text)  # (batch_size, seq_len, embedding_dim)
            features = self._vectorizer(embeddings, mask=mask)  # (batch_size, embedding_dim)

            # Average pooling with mask
            logits = self._classifier(features)
            probs = torch.softmax(logits, dim=-1)
            label = probs.argmax(dim=-1)

            loss: Optional[torch.Tensor] = None
            if inputs.label is not None:
                loss = self._loss(logits, inputs.label)

            return ClassifierOutput(probs=probs, label=label, loss=loss)

    def test_training_with_text_classifier(self, text_classification_dataset) -> None:
        from formed.integrations.ml import Tokenizer, TokenSequenceIndexer

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

        model = self.SimpleTextClassifier(
            num_classes=datamodule.label.num_labels,
            vocab_size=datamodule.text.surfaces.vocab_size,
            embedding_dim=64,
        )

        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            max_epochs=5,
            callbacks=[EvaluationCallback(ClassificationEvaluator())],
        )

        trainer.train(model, train_instances, val_instances)
