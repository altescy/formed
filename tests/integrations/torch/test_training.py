import dataclasses
from typing import Any, Generic, Optional, cast

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
    BalancedByDistributionLabelWeighter,
    BaseClassificationLoss,
    CrossEntropyLoss,
    MeanSquaredErrorLoss,
    TokenEmbedder,
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

        engine = DefaultTorchTrainingEngine(
            optimizer=torch.optim.Adam,
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
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
            loss: Optional[BaseClassificationLoss] = None,
        ) -> None:
            super().__init__()
            self._num_classes = num_classes
            self._embedder = AnalyzedTextEmbedder(
                surface=TokenEmbedder(initializer=XavierUniformTensorInitializer((vocab_size, embedding_dim))),
            )
            self._vectorizer = BagOfEmbeddingsSequenceVectorizer()
            self._classifier = nn.Linear(embedding_dim, num_classes)
            self._loss = loss or CrossEntropyLoss()

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
            embedding_dim=32,
            loss=CrossEntropyLoss(weighter=BalancedByDistributionLabelWeighter(datamodule.label.distribution)),
        )

        engine = DefaultTorchTrainingEngine(
            optimizer=torch.optim.Adam,
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
            max_epochs=5,
            callbacks=[EvaluationCallback(ClassificationEvaluator())],
        )

        trainer.train(model, train_instances, val_instances)

    def test_training_with_lr_scheduler(self, text_classification_dataset) -> None:
        """Test training with learning rate scheduler."""
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
            embedding_dim=32,
        )

        # Track initial learning rate
        initial_lr = 1e-2

        engine = DefaultTorchTrainingEngine(
            optimizer=lambda params: torch.optim.Adam(params, lr=initial_lr),
            lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                cast(torch.optim.Optimizer, optimizer), step_size=10, gamma=0.5
            ),
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
            max_epochs=3,
            callbacks=[EvaluationCallback(ClassificationEvaluator())],
        )

        state = trainer.train(model, train_instances, val_instances)

        # Verify that lr_scheduler was used
        assert state.lr_scheduler is not None
        # Learning rate should have changed after training
        # Cast to access param_groups which is not in IOptimizer protocol
        optimizer_with_groups = cast(Any, state.optimizer)
        final_lr = optimizer_with_groups.param_groups[0]["lr"]
        # After 3 epochs with batch_size=16 and 80 samples, we have 15 steps (5 per epoch)
        # With step_size=10, lr should change once: initial_lr * 0.5
        assert final_lr < initial_lr

    def test_training_with_lr_scheduler_callable(self, text_classification_dataset) -> None:
        """Test training with learning rate scheduler via callable."""
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
            embedding_dim=32,
        )

        initial_lr = 1e-2

        # Use callable for lr_scheduler initialization
        engine = DefaultTorchTrainingEngine(
            optimizer=lambda params: torch.optim.Adam(params, lr=initial_lr),
            lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(
                cast(torch.optim.Optimizer, optimizer), gamma=0.9
            ),
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
            max_epochs=3,
            callbacks=[EvaluationCallback(ClassificationEvaluator())],
        )

        state = trainer.train(model, train_instances, val_instances)

        # Verify scheduler was initialized and used
        assert state.lr_scheduler is not None
        # Cast to access param_groups which is not in IOptimizer protocol
        optimizer_with_groups = cast(Any, state.optimizer)
        final_lr = optimizer_with_groups.param_groups[0]["lr"]
        # With ExponentialLR and gamma=0.9, lr should decrease
        assert final_lr < initial_lr

    def test_training_without_lr_scheduler(self, text_classification_dataset) -> None:
        """Test that training works without lr_scheduler (backward compatibility)."""
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
            embedding_dim=32,
        )

        initial_lr = 1e-3

        engine = DefaultTorchTrainingEngine(
            optimizer=lambda params: torch.optim.Adam(params, lr=initial_lr),
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
            max_epochs=3,
            callbacks=[EvaluationCallback(ClassificationEvaluator())],
        )

        state = trainer.train(model, train_instances, val_instances)

        # Verify that lr_scheduler is None
        assert state.lr_scheduler is None
        # Learning rate should remain unchanged
        # Cast to access param_groups which is not in IOptimizer protocol
        optimizer_with_groups = cast(Any, state.optimizer)
        final_lr = optimizer_with_groups.param_groups[0]["lr"]
        assert final_lr == initial_lr


class TestTrainStateWithLRScheduler:
    """Tests for TrainState serialization with lr_scheduler."""

    def test_state_dict_with_lr_scheduler(self) -> None:
        """Test that state_dict includes lr_scheduler state."""
        from formed.integrations.torch.training.state import TrainState

        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        state = TrainState(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=10,
        )

        state_dict = state.state_dict()

        # Verify all components are in state_dict
        assert "model_state" in state_dict
        assert "optimizer_state" in state_dict
        assert "lr_scheduler_state" in state_dict
        assert "step" in state_dict
        assert state_dict["step"] == 10

    def test_state_dict_without_lr_scheduler(self) -> None:
        """Test that state_dict works without lr_scheduler."""
        from formed.integrations.torch.training.state import TrainState

        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        state = TrainState(
            model=model,
            optimizer=optimizer,
            step=10,
        )

        state_dict = state.state_dict()

        # Verify lr_scheduler_state is not in state_dict
        assert "model_state" in state_dict
        assert "optimizer_state" in state_dict
        assert "lr_scheduler_state" not in state_dict
        assert "step" in state_dict
        assert state_dict["step"] == 10

    def test_load_state_dict_with_lr_scheduler(self) -> None:
        """Test loading state_dict with lr_scheduler."""
        from formed.integrations.torch.training.state import TrainState

        # Create initial state
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        state = TrainState(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=10,
        )

        # Take some steps to change scheduler state
        for _ in range(7):
            optimizer.step()
            lr_scheduler.step()

        # Save state
        saved_state_dict = state.state_dict()
        saved_lr = optimizer.param_groups[0]["lr"]

        # Create new state and load
        new_model = nn.Linear(10, 1)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_lr_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.5)

        new_state = TrainState(
            model=new_model,
            optimizer=new_optimizer,
            lr_scheduler=new_lr_scheduler,
            step=0,
        )

        new_state.load_state_dict(saved_state_dict)

        # Verify state was restored
        assert new_state.step == 10
        loaded_lr = new_optimizer.param_groups[0]["lr"]
        assert loaded_lr == saved_lr

    def test_load_state_dict_without_lr_scheduler(self) -> None:
        """Test loading state_dict without lr_scheduler doesn't crash."""
        from formed.integrations.torch.training.state import TrainState

        # Create state without scheduler
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        state = TrainState(
            model=model,
            optimizer=optimizer,
            step=10,
        )

        saved_state_dict = state.state_dict()

        # Create new state with scheduler and load state without scheduler
        new_model = nn.Linear(10, 1)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_lr_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.5)

        new_state = TrainState(
            model=new_model,
            optimizer=new_optimizer,
            lr_scheduler=new_lr_scheduler,
            step=0,
        )

        # This should not crash
        new_state.load_state_dict(saved_state_dict)

        # Verify step was loaded
        assert new_state.step == 10
        # Scheduler state should remain at initial state (not loaded)
        assert new_state.lr_scheduler is not None


class TestGradientClipping:
    """Tests for gradient clipping functionality."""

    def test_training_with_gradient_clipping(self, text_classification_dataset) -> None:
        """Test that gradient clipping is applied during training."""
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

        model = TestTrainingWithTextClassifier.SimpleTextClassifier(
            num_classes=datamodule.label.num_labels,
            vocab_size=datamodule.text.surfaces.vocab_size,
            embedding_dim=32,
        )

        # Create engine with gradient clipping
        engine = DefaultTorchTrainingEngine(
            optimizer=lambda params: torch.optim.Adam(params, lr=1e-2),
            max_grad_norm=1.0,
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            engine=engine,
            max_epochs=2,
            callbacks=[EvaluationCallback(ClassificationEvaluator())],
        )

        # Train model
        state = trainer.train(model, train_instances, val_instances)

        # Verify training completed successfully
        assert state.step > 0

    def test_gradient_clipping_with_grad_scaler(self, text_classification_dataset) -> None:
        """Test gradient clipping works with gradient scaling (mixed precision)."""
        from formed.integrations.ml import Tokenizer, TokenSequenceIndexer

        train_data = text_classification_dataset[:80]

        datamodule = TextClassificationDataModule(
            text=Tokenizer(surfaces=TokenSequenceIndexer()),
            label=LabelIndexer(),
        )

        with datamodule.train():
            train_instances = [datamodule.instance(example) for example in train_data]

        train_dataloader = DataLoader(
            sampler=BasicBatchSampler(batch_size=16, shuffle=True, drop_last=True),
            collator=datamodule.batch,
        )

        model = TestTrainingWithTextClassifier.SimpleTextClassifier(
            num_classes=datamodule.label.num_labels,
            vocab_size=datamodule.text.surfaces.vocab_size,
            embedding_dim=32,
        )

        # Create engine with gradient clipping and grad_scaler
        engine = DefaultTorchTrainingEngine(
            optimizer=torch.optim.Adam,
            max_grad_norm=1.0,
            dtype="float32",
            grad_scaler=torch.amp.grad_scaler.GradScaler(device="cpu"),
        )
        trainer = TorchTrainer(
            train_dataloader=train_dataloader,
            engine=engine,
            max_epochs=1,
        )

        # Train model
        state = trainer.train(model, train_instances)

        # Verify training completed successfully
        assert state.step > 0
        assert state.grad_scaler is not None
