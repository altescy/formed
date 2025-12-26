"""Tests for PyTorch workflow steps."""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Generic, Optional, cast

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
    Tokenizer,
    TokenSequenceIndexer,
)
from formed.integrations.ml.metrics import Average, MeanSquaredError, MulticlassAccuracy
from formed.integrations.ml.types import AsBatch, AsInstance, DataModuleModeT  # noqa: F401
from formed.integrations.torch import (
    BaseTorchModel,
    DefaultTorchTrainingEngine,
    EvaluationCallback,
    TorchTrainer,
    XavierUniformTensorInitializer,
    ensure_torch_tensor,
)
from formed.integrations.torch.modules import (
    AnalyzedTextEmbedder,
    BagOfEmbeddingsSequenceVectorizer,
    CrossEntropyLoss,
    MeanSquaredErrorLoss,
    TokenEmbedder,
)
from formed.integrations.torch.workflow import (
    TorchModelFormat,
    evaluate_torch_model,
    predict,
)
from formed.workflow import DefaultWorkflowExecutor, MemoryWorkflowCache, WorkflowGraph


@dataclasses.dataclass
class RegressionExample:
    features: numpy.ndarray
    target: float


class RegressionDataModule(
    DataModule[
        DataModuleModeT,
        RegressionExample,
        "RegressionDataModule[AsInstance]",
        "RegressionDataModule[AsBatch]",
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


@BaseTorchModel.register("torch_regressor_test")
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


class RegressionPostProcessor:
    def __call__(self, inputs: RegressionDataModule[AsBatch], output: RegressorOutput) -> list[float]:
        return output.predictions.detach().cpu().tolist()


@dataclasses.dataclass
class TextClassificationExample:
    text: str
    label: str


class TextClassificationDataModule(
    DataModule[
        DataModuleModeT,
        TextClassificationExample,
        "TextClassificationDataModule[AsInstance]",
        "TextClassificationDataModule[AsBatch]",
    ],
    Generic[DataModuleModeT],
):
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


@BaseTorchModel.register("simple_text_classifier_test")
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
            surface=TokenEmbedder(initializer=XavierUniformTensorInitializer((vocab_size, embedding_dim))),
        )
        self._vectorizer = BagOfEmbeddingsSequenceVectorizer()
        self._classifier = nn.Linear(embedding_dim, num_classes)
        self._loss = CrossEntropyLoss()

    def forward(
        self,
        inputs: TextClassificationDataModule[AsBatch],
        params: None = None,
    ) -> ClassifierOutput:
        embeddings, mask = self._embedder(inputs.text)
        features = self._vectorizer(embeddings, mask=mask)

        logits = self._classifier(features)
        probs = torch.softmax(logits, dim=-1)
        label = probs.argmax(dim=-1)

        loss: Optional[torch.Tensor] = None
        if inputs.label is not None:
            loss = self._loss(logits, inputs.label)

        return ClassifierOutput(probs=probs, label=label, loss=loss)


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


class TestTorchModelFormat:
    """Test the TorchModelFormat class."""

    @staticmethod
    def test_write_and_read_model_with_pickle(tmp_path: Path) -> None:
        """Test writing and reading a model using pickle format."""
        format = TorchModelFormat()
        model = TorchRegressor(feature_dim=5)

        # No config set, should use pickle
        model.__model_config__ = None

        format.write(model, tmp_path)
        restored_model = format.read(tmp_path)

        assert isinstance(restored_model, TorchRegressor)
        assert restored_model.dense.in_features == 5
        assert restored_model.dense.out_features == 1

    @staticmethod
    def test_write_model_with_config_saves_files(tmp_path: Path) -> None:
        """Test that writing a model with config creates expected files."""
        format = TorchModelFormat()
        model = TorchRegressor(feature_dim=10)

        # Set model config
        model.__model_config__ = {
            "type": "torch_regressor_test",
            "feature_dim": 10,
        }

        format.write(model, tmp_path)

        # Check that config and state files are created
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "state.pth").exists()
        assert not (tmp_path / "model.pkl").exists()


class TestTrainTorchModel:
    """Test the train_torch_model workflow step."""

    @staticmethod
    def test_train_regression_model_basic(regression_dataset: list[RegressionExample]) -> None:
        """Test training a regression model directly (not through workflow system)."""
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

        # Direct test without using Lazy
        model = TorchRegressor(feature_dim=10)
        engine = DefaultTorchTrainingEngine(
            optimizer=torch.optim.Adam,
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
            max_epochs=2,
            callbacks=[EvaluationCallback(RegressionEvaluator())],
        )

        state = trainer.train(model, train_instances, val_instances)
        assert isinstance(state.model, TorchRegressor)


class TestEvaluateTorchModel:
    """Test the evaluate_torch_model workflow step."""

    @staticmethod
    def test_evaluate_regression_model_basic(regression_dataset: list[RegressionExample]) -> None:
        """Test evaluating a regression model basic functionality."""
        test_data = regression_dataset[:20]

        datamodule = RegressionDataModule(
            features=TensorTransform(),
            target=ScalarTransform(),
        )

        test_instances = [datamodule.instance(example) for example in test_data]

        test_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=datamodule.batch,
        )

        model = TorchRegressor(feature_dim=10)
        model.eval()

        evaluator = RegressionEvaluator()

        # Evaluate manually (not through workflow step)
        with torch.inference_mode():
            for inputs in test_dataloader(test_instances):
                output = model(inputs)
                evaluator.update(inputs, output)

        metrics = evaluator.compute()
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "mean_squared_error" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    @staticmethod
    def test_evaluate_text_classifier(text_classification_dataset: list[TextClassificationExample]) -> None:
        """Test evaluating a text classification model."""
        test_data = text_classification_dataset[:20]

        datamodule = TextClassificationDataModule(
            text=Tokenizer(surfaces=TokenSequenceIndexer()),
            label=LabelIndexer(),
        )

        with datamodule.train():
            _ = [datamodule.instance(example) for example in text_classification_dataset]

        test_instances = [datamodule.instance(example) for example in test_data]

        test_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=datamodule.batch,
        )

        model = SimpleTextClassifier(
            num_classes=datamodule.label.num_labels,
            vocab_size=datamodule.text.surfaces.vocab_size,
            embedding_dim=32,
        )

        # Test the workflow step
        metrics = evaluate_torch_model(
            model=model,
            evaluator=ClassificationEvaluator(),
            dataset=test_instances,
            dataloader=cast(Callable, test_dataloader),
            random_seed=42,
        )

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestPredictStep:
    """Test the predict workflow step."""

    @staticmethod
    def test_predict_regression(regression_dataset: list[RegressionExample]) -> None:
        """Test prediction with regression model."""
        test_data = regression_dataset[:10]

        datamodule = RegressionDataModule(
            features=TensorTransform(),
            target=ScalarTransform(),
        )

        test_instances = [datamodule.instance(example) for example in test_data]

        test_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=datamodule.batch,
        )

        model = TorchRegressor(feature_dim=10)

        def postprocessor(_: RegressionDataModule[AsBatch], output: RegressorOutput):
            return output.predictions.detach().cpu().tolist()

        # Test the workflow step
        predictions = list(
            predict(
                dataset=test_instances,
                dataloader=cast(Callable, test_dataloader),
                model=model,
                postprocessor=postprocessor,
                random_seed=42,
            )
        )

        assert len(predictions) > 0
        assert all(isinstance(pred, float) for pred in predictions)

    @staticmethod
    def test_predict_text_classifier(text_classification_dataset: list[TextClassificationExample]) -> None:
        """Test prediction with text classification model."""
        test_data = text_classification_dataset[:10]

        datamodule = TextClassificationDataModule(
            text=Tokenizer(surfaces=TokenSequenceIndexer()),
            label=LabelIndexer(),
        )

        with datamodule.train():
            _ = [datamodule.instance(example) for example in text_classification_dataset]

        test_instances = [datamodule.instance(example) for example in test_data]

        test_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=False),
            collator=datamodule.batch,
        )

        model = SimpleTextClassifier(
            num_classes=datamodule.label.num_labels,
            vocab_size=datamodule.text.surfaces.vocab_size,
            embedding_dim=32,
        )

        def postprocessor(inputs: TextClassificationDataModule[AsBatch], output: ClassifierOutput):
            return output.label.detach().cpu().tolist()

        # Test the workflow step
        predictions = list(
            predict(
                dataset=test_instances,
                dataloader=cast(Callable, test_dataloader),
                model=model,
                postprocessor=postprocessor,
                random_seed=42,
            )
        )

        assert len(predictions) > 0
        assert all(isinstance(pred, int) for pred in predictions)


class TestTorchWorkflowIntegration:
    """Test integration of torch workflow steps with the workflow system."""

    @staticmethod
    def test_workflow_graph_with_train_step(regression_dataset: list[RegressionExample]) -> None:
        """Test that train_torch_model can be used in a workflow graph."""
        from formed.workflow import step

        train_data = regression_dataset[:80]
        val_data = regression_dataset[80:]

        datamodule = RegressionDataModule(
            features=TensorTransform(),
            target=ScalarTransform(),
        )

        with datamodule.train():
            train_instances = [datamodule.instance(example) for example in train_data]
        val_instances = [datamodule.instance(example) for example in val_data]

        @step("test_torch_workflow::get_train_data")
        def get_train_data() -> list:
            return train_instances

        @step("test_torch_workflow::get_val_data")
        def get_val_data() -> list:
            return val_instances

        @step("test_torch_workflow::get_trainer")
        def get_trainer() -> TorchTrainer:
            train_dataloader = DataLoader(
                sampler=BasicBatchSampler(batch_size=16, shuffle=True, drop_last=True),
                collator=datamodule.batch,
            )
            val_dataloader = DataLoader(
                sampler=BasicBatchSampler(batch_size=16, shuffle=False),
                collator=datamodule.batch,
            )

            engine = DefaultTorchTrainingEngine(
                optimizer=torch.optim.Adam,
            )
            return TorchTrainer(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                engine=engine,
                max_epochs=2,
                callbacks=[EvaluationCallback(RegressionEvaluator())],
            )

        # Create workflow graph
        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "train_data": {"type": "test_torch_workflow::get_train_data"},
                    "val_data": {"type": "test_torch_workflow::get_val_data"},
                    "trainer": {"type": "test_torch_workflow::get_trainer"},
                    "trained_model": {
                        "type": "torch::train",
                        "model": {
                            "type": "torch_regressor_test",
                            "feature_dim": 10,
                        },
                        "trainer": {"type": "ref", "ref": "trainer"},
                        "train_dataset": {"type": "ref", "ref": "train_data"},
                        "val_dataset": {"type": "ref", "ref": "val_data"},
                        "random_seed": 42,
                    },
                    "predictions": {
                        "type": "torch::predict",
                        "dataset": {"type": "ref", "ref": "val_data"},
                        "dataloader": {
                            "type": "formed.integrations.ml:DataLoader",
                            "sampler": {
                                "type": "basic",
                                "batch_size": 16,
                                "shuffle": False,
                            },
                            "collator": datamodule.batch,  # type: ignore
                        },
                        "model": {"type": "ref", "ref": "trained_model"},
                        "postprocessor": RegressionPostProcessor(),  # type: ignore
                        "random_seed": 42,
                    },
                }
            }
        )

        # Execute workflow
        cache = MemoryWorkflowCache()
        executor = DefaultWorkflowExecutor()
        context = executor(graph, cache=cache)

        # Check result
        trained_model = context.cache[context.info.graph["trained_model"]]
        assert isinstance(trained_model, TorchRegressor)
        predictions = list(context.cache[context.info.graph["predictions"]])
        assert len(predictions) == 20
