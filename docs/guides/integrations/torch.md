# PyTorch Integration Guide

## Overview

The `formed.integrations.torch` module provides comprehensive PyTorch integration for training deep learning models within the formed workflow system. It combines model definition, training infrastructure, and workflow integration with automatic caching and reproducibility.

**Design philosophy:**

This integration emphasizes **declarative model composition** through reusable, configurable modules. Rather than implementing models from scratch, you compose them from pre-built components (embedders, encoders, vectorizers, etc.) that can be fully specified in configuration files. This approach separates model architecture from implementation details and enables rapid experimentation.

**Key capabilities:**

- **Declarative Model Composition**: Build models from reusable modules (`formed.integrations.torch.modules`)
- **Configuration-Driven**: Define complete models and training pipelines in Jsonnet/JSON
- **Training Infrastructure**: Complete training pipeline with callbacks, evaluation, and distributed training
- **Workflow Integration**: Built-in workflow steps for seamless integration with formed's caching system
- **Distributed Training**: Support for data parallelism across multiple devices

## Core Concepts

### Architecture Overview

The PyTorch integration is organized around several key components that work together:

```
BaseTorchModel              # Model definition with forward pass
     ↓
TorchTrainingEngine         # Loss computation and optimization logic
     ↓
TorchTrainer                # Training loop coordination
│    ↓ (uses)
├─ DataLoader               # Batch iteration
├─ BaseDistributor          # Device management and parallelism
├─ TorchTrainingCallback    # Hooks for monitoring and control
└─ Evaluator                # Metric computation
     ↓
train_torch_model           # Workflow step for training
```

**Component relationships:**

1. **BaseTorchModel** defines model architecture and forward pass
2. **TorchTrainingEngine** implements training logic (loss, gradients, optimization)
3. **TorchTrainer** orchestrates the training loop, calling the engine for each batch
4. **BaseDistributor** manages device placement and distributed training
5. **TorchTrainingCallback** provides hooks for custom behavior at various training stages
6. **train_torch_model** wraps everything as a workflow step with caching

### Training Flow

The typical training flow:

1. **Initialization**: TorchTrainer creates TrainState from model and engine
2. **Epoch Loop**: For each epoch:
   - Load batches from train DataLoader
   - Execute train_step via engine (forward + backward + optimize)
   - Optionally evaluate on validation data
   - Execute callbacks at appropriate points
   - Log metrics according to logging strategy
3. **Completion**: Return final model and training state

## Model Definition

### BaseTorchModel

`BaseTorchModel` is the base class for all PyTorch models in the framework. It combines `torch.nn.Module` with the registrable pattern for configuration-based instantiation.

**Type parameters:**

```python
BaseTorchModel[ModelInputT, ModelOutputT, ModelParamsT]
```

- **ModelInputT**: Type of batched input (typically from DataModule)
- **ModelOutputT**: Type of model output (dict, NamedTuple, or custom dataclass)
- **ModelParamsT**: Type of additional parameters (usually `None` or a dataclass)

**Key features:**

- Automatically compatible with TorchTrainer
- Can be registered and instantiated from configuration
- Supports automatic model serialization with `TorchModelFormat`

### Declarative Model Composition

The `formed.integrations.torch.modules` package provides reusable neural network modules that can be composed declaratively:

**Example - Text Classification Model:**

```python
from formed.integrations.torch import BaseTorchModel
from formed.integrations.torch import modules as ftm
from formed.integrations.ml import types as mlt
import dataclasses
import torch

@dataclasses.dataclass
class ClassifierOutput:
    probs: torch.Tensor
    label: torch.Tensor
    loss: torch.Tensor | None = None

@BaseTorchModel.register("text_classifier")
class TextClassifier(BaseTorchModel):
    def __init__(
        self,
        num_classes: int,
        embedder: ftm.BaseEmbedder,
        encoder: ftm.BaseSequenceEncoder | None = None,
        vectorizer: ftm.BaseSequenceVectorizer | None = None,
        feedforward: ftm.FeedForward | None = None,
        sampler: ftm.BaseLabelSampler | None = None,
        loss: ftm.BaseClassificationLoss | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Use defaults if not provided
        vectorizer = vectorizer or ftm.BagOfEmbeddingsSequenceVectorizer()
        sampler = sampler or ftm.ArgmaxLabelSampler()
        loss = loss or ftm.CrossEntropyLoss()

        # Determine output dimension through the pipeline
        feature_dim = self._determine_feature_dim(
            embedder, encoder, vectorizer, feedforward
        )

        self._embedder = embedder
        self._encoder = encoder
        self._vectorizer = vectorizer
        self._feedforward = feedforward
        self._dropout = torch.nn.Dropout(dropout)
        self._classifier = torch.nn.Linear(feature_dim, num_classes)
        self._sampler = sampler
        self._loss = loss

    def forward(self, inputs, params=None):
        # inputs: batch from DataModule (e.g., TextClassificationDataModule[AsBatch])
        embeddings, mask = self._embedder(inputs.text)

        if self._encoder is not None:
            embeddings = self._encoder(embeddings, mask=mask)

        vector = self._vectorizer(embeddings, mask=mask)

        if self._feedforward is not None:
            vector = self._feedforward(vector)

        vector = self._dropout(vector)
        logits = self._classifier(vector)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        label = self._sampler(logits)

        loss = None
        if inputs.label is not None:
            loss = self._loss(logits, inputs.label)

        return ClassifierOutput(probs=probs, label=label, loss=loss)
```

### Available Modules

The `formed.integrations.torch.modules` package provides building blocks for composing models. These modules form a typical NLP pipeline:

1. **Embedders** - Convert tokens to embeddings (e.g., `TokenEmbedder`, `AnalyzedTextEmbedder`)
2. **Encoders** - Process sequences with context (e.g., `LSTMSequenceEncoder`, `TransformerEncoder`)
3. **Vectorizers** - Aggregate sequences to fixed-size vectors (e.g., `BagOfEmbeddingsSequenceVectorizer`)
4. **FeedForward** - Additional transformation layers with configurable depth and activations
5. **Losses** - Task-specific loss functions (e.g., `CrossEntropyLoss`, `BCEWithLogitsLoss`)
6. **Samplers** - Convert logits to labels (e.g., `ArgmaxLabelSampler`, `MultinomialLabelSampler`)

Additional modules include positional encoders, attention masks, and label weighters. See the API reference for complete details.

### Configuration-Based Model Definition

Models composed from these modules can be fully specified in configuration:

```jsonnet
{
  model: {
    type: 'text_classifier',
    num_classes: 2,
    embedder: {
      type: 'token',
      embedding_dim: 128,
      vocab_size: 10000,
    },
    encoder: {
      type: 'lstm',
      hidden_size: 256,
      num_layers: 2,
      bidirectional: true,
    },
    vectorizer: {
      type: 'bag_of_embeddings',
    },
    feedforward: {
      input_dim: 512,  # 256 * 2 (bidirectional)
      hidden_dims: [256, 128],
      activations: 'relu',
    },
    dropout: 0.2,
  },
}
```

This declarative approach:

- Separates model architecture from implementation
- Enables easy experimentation with different configurations
- Maintains type safety through the pipeline
- Reduces boilerplate code

## Training Infrastructure

### TorchTrainingEngine

`TorchTrainingEngine` defines how models are trained by implementing state creation, training steps, and evaluation steps.

**DefaultTorchTrainingEngine** - Standard training with automatic differentiation:

```python
from formed.integrations.torch import DefaultTorchTrainingEngine
import torch.optim as optim

engine = DefaultTorchTrainingEngine(
    optimizer=optim.Adam,           # Optimizer class or factory
    optimizer_params={"lr": 1e-3},  # Optimizer parameters
    lr_scheduler=optim.lr_scheduler.StepLR,  # Optional scheduler
    lr_scheduler_params={"step_size": 10, "gamma": 0.1},
    loss="loss",                    # Accessor for loss in model output
    max_grad_norm=1.0,             # Optional gradient clipping
    accumulation_steps=1,          # Gradient accumulation steps
)
```

**Loss specification:**

The `loss` parameter can be:

- A string accessor: `"loss"` extracts `output["loss"]` or `output.loss`
- A callable: `lambda output: output.loss + 0.1 * output.regularization`

**Custom engines:**

Implement `TorchTrainingEngine` for custom training logic:

```python
class CustomTrainingEngine(TorchTrainingEngine):
    def create_state(self, trainer, model):
        # Initialize optimizer, scheduler, etc.
        optimizer = torch.optim.Adam(model.parameters())
        return TrainState(model=model, optimizer=optimizer)

    def train_step(self, inputs, state, trainer):
        state.model.train()
        state.optimizer.zero_grad()

        output = state.model(inputs)
        loss = output["loss"]
        loss.backward()

        state.optimizer.step()
        return output

    def eval_step(self, inputs, state, trainer):
        state.model.eval()
        with torch.no_grad():
            output = state.model(inputs)
        return output
```

### TorchTrainer

`TorchTrainer` orchestrates the complete training process, coordinating data loading, training steps, evaluation, callbacks, and logging.

**Basic usage:**

```python
from formed.integrations.torch import TorchTrainer, DefaultTorchTrainingEngine
from formed.integrations.ml import DataLoader, BasicBatchSampler
import torch.optim as optim

# Setup data loaders
train_loader = DataLoader(
    sampler=BasicBatchSampler(batch_size=32, shuffle=True),
    collator=datamodule.batch,
)

val_loader = DataLoader(
    sampler=BasicBatchSampler(batch_size=64),
    collator=datamodule.batch,
)

# Create engine
engine = DefaultTorchTrainingEngine(
    optimizer=optim.Adam,
    optimizer_params={"lr": 1e-3},
)

# Create trainer
trainer = TorchTrainer(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    engine=engine,
    max_epochs=10,
    eval_strategy="epoch",      # Evaluate every N epochs
    eval_interval=1,
    logging_strategy="step",    # Log every N steps
    logging_interval=100,
)

# Train model
state = trainer.train(model, train_instances, val_instances)
```

**Configuration options:**

- **max_epochs**: Maximum number of training epochs
- **eval_strategy**: When to evaluate - `"epoch"` or `"step"`
- **eval_interval**: Evaluation frequency (number of epochs or steps)
- **logging_strategy**: When to log metrics - `"epoch"` or `"step"`
- **logging_interval**: Logging frequency (number of epochs or steps)
- **logging_first_step**: Whether to log after the first training step
- **train_prefix**: Prefix for training metrics (default: `"train/"`)
- **val_prefix**: Prefix for validation metrics (default: `"val/"`)

### TrainState

`TrainState` encapsulates the training state, including model, optimizer, and counters:

```python
@dataclasses.dataclass
class TrainState:
    model: BaseTorchModel           # The model being trained
    optimizer: IOptimizer           # Optimizer instance
    lr_scheduler: ILRScheduler | None  # Optional LR scheduler
    step: int = 0                   # Global step counter
    epoch: int = 0                  # Current epoch
    best_metric: float | None = None   # Best validation metric
    metadata: dict[str, Any] = field(default_factory=dict)  # Custom metadata
```

The state is updated in-place during training and can be accessed in callbacks.

## Callbacks

### TorchTrainingCallback

Callbacks provide hooks to execute custom logic at various points during training.

**Hook execution order:**

1. `on_training_start` - Once at the beginning
2. `on_epoch_start` - At the start of each epoch
3. `on_batch_start` - Before each training batch
4. `on_batch_end` - After each training batch
5. `on_eval_start` - Before evaluation (returns evaluator)
6. `on_eval_end` - After evaluation with computed metrics
7. `on_log` - When metrics are logged
8. `on_epoch_end` - At the end of each epoch
9. `on_training_end` - Once at the end

### Built-in Callbacks

#### EvaluationCallback

Computes metrics using a custom evaluator:

```python
from formed.integrations.torch import EvaluationCallback
from formed.integrations.ml import MulticlassAccuracy

# Define evaluator
class MyEvaluator:
    def __init__(self):
        self.accuracy = MulticlassAccuracy()

    def update(self, inputs, output):
        predictions = output["logits"].argmax(dim=-1).tolist()
        targets = inputs.label.tolist()
        self.accuracy.update(
            self.accuracy.Input(predictions=predictions, targets=targets)
        )

    def compute(self):
        return self.accuracy.compute()

    def reset(self):
        self.accuracy.reset()

# Use in trainer
trainer = TorchTrainer(
    ...,
    callbacks=[EvaluationCallback(MyEvaluator())],
)
```

#### EarlyStoppingCallback

Stops training when a metric stops improving:

```python
from formed.integrations.torch import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=5,              # Number of evaluations without improvement
    metric="-loss",          # Metric to monitor (- prefix for minimization)
    min_delta=0.0,          # Minimum change to qualify as improvement
    restore_best_weights=True,  # Restore model to best state
)

trainer = TorchTrainer(
    ...,
    callbacks=[callback],
)
```

**Metric specification:**

- Prefix with `-` for metrics to minimize (e.g., `"-loss"`)
- No prefix for metrics to maximize (e.g., `"accuracy"`)

#### MlflowCallback

Logs metrics to MLflow:

```python
from formed.integrations.torch import MlflowCallback

callback = MlflowCallback()

trainer = TorchTrainer(
    ...,
    callbacks=[callback],
)
```

This callback automatically logs:

- Training and validation metrics
- Model parameters and hyperparameters
- System metrics (GPU usage, etc.)

### Custom Callbacks

Implement `TorchTrainingCallback` for custom behavior:

```python
from formed.integrations.torch import TorchTrainingCallback

@TorchTrainingCallback.register("my_callback")
class MyCallback(TorchTrainingCallback):
    def on_epoch_end(self, trainer, model, state, epoch):
        print(f"Completed epoch {epoch}")
        # Save checkpoint, log custom metrics, etc.

    def on_eval_end(self, trainer, model, state, metrics):
        print(f"Validation metrics: {metrics}")
```

## Distributed Training

### BaseDistributor

`BaseDistributor` manages device placement and distributed training strategies.

**SingleDeviceDistributor** - No distribution (default):

```python
from formed.integrations.torch import SingleDeviceDistributor

distributor = SingleDeviceDistributor(device="cuda:0")
```

**DataParallelDistributor** - Data parallelism across multiple GPUs:

```python
from formed.integrations.torch import DataParallelDistributor

distributor = DataParallelDistributor(
    device_ids=[0, 1, 2, 3],  # GPUs to use (None = all available)
)
```

**DistributedDataParallelDistributor** - Distributed data parallelism:

```python
from formed.integrations.torch import DistributedDataParallelDistributor

distributor = DistributedDataParallelDistributor(
    backend="nccl",           # Backend for distributed communication
    init_method="env://",     # Initialization method
)
```

**Using distributors:**

```python
trainer = TorchTrainer(
    ...,
    distributor=distributor,
)
```

The distributor:

- Wraps the model for distributed training
- Handles device placement
- Reduces metrics across devices
- Manages process synchronization

## Workflow Integration

### train_torch_model

The `torch::train` workflow step trains a PyTorch model and caches the result:

```jsonnet
{
  steps: {
    # Prepare data
    datamodule_and_instances: {
      type: 'ml::train_datamodule_with_instances',
      dataset: { type: 'ref', ref: 'dataset' },
      datamodule: { type: 'my_datamodule', ... },
    },

    # Train model
    trained_model: {
      type: 'torch::train',
      model: {
        type: 'text_classifier',
        vocab_size: 10000,
        embedding_dim: 128,
        hidden_dim: 256,
        num_classes: 2,
      },
      trainer: {
        train_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: {
            type: 'basic',
            batch_size: 32,
            shuffle: true,
          },
          collator: { type: 'ref', ref: 'datamodule_and_instances.datamodule.batch' },
        },
        engine: {
          type: 'default',
          optimizer: { type: 'torch.optim:Adam', lr: 0.001 },
          loss: 'loss',
        },
        max_epochs: 10,
        eval_strategy: 'epoch',
        callbacks: [
          { type: 'evaluation', evaluator: { type: 'my_evaluator' } },
          { type: 'early_stopping', patience: 3, metric: '-loss' },
        ],
      },
      train_dataset: { type: 'ref', ref: 'datamodule_and_instances.instances' },
      val_dataset: { type: 'ref', ref: 'val_instances' },
      random_seed: 42,
      device: 'cuda:0',
    },
  },
}
```

**Parameters:**

- **model**: Model configuration (Lazy-loaded)
- **trainer**: TorchTrainer configuration
- **train_dataset**: Training instances
- **val_dataset**: Optional validation instances
- **random_seed**: Random seed for reproducibility
- **device**: Device for training (optional, can be set via context)

**Model caching:**

Models are cached using `TorchModelFormat`:

- If model has `__model_config__`, saves config + state_dict separately
- Otherwise, pickles the entire model
- Enables efficient caching and model reuse across workflow runs

### evaluate_torch_model

The `torch::evaluate` workflow step evaluates a trained model:

```jsonnet
{
  steps: {
    evaluation: {
      type: 'torch::evaluate',
      model: { type: 'ref', ref: 'trained_model' },
      dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        sampler: { type: 'basic', batch_size: 64 },
        collator: { type: 'ref', ref: 'datamodule.batch' },
      },
      evaluator: { type: 'my_evaluator' },
      dataset: { type: 'ref', ref: 'test_instances' },
      device: 'cuda:0',
    },
  },
}
```

## Complete Example

Here's a complete example combining all components with declarative model composition:

```python
from formed.integrations.torch import (
    BaseTorchModel,
    TorchTrainer,
    DefaultTorchTrainingEngine,
    EvaluationCallback,
    EarlyStoppingCallback,
    modules as ftm,
)
from formed.integrations.ml import (
    DataLoader,
    BasicBatchSampler,
    MulticlassAccuracy,
    DataModule,
    Tokenizer,
    LabelIndexer,
    Extra,
)
import dataclasses
import torch

# 1. Define DataModule
class TextClassificationDataModule(DataModule):
    text: Tokenizer
    label: Extra[LabelIndexer] = Extra.default()

# 2. Define model using reusable modules
@dataclasses.dataclass
class ClassifierOutput:
    probs: torch.Tensor
    label: torch.Tensor
    loss: torch.Tensor | None = None

@BaseTorchModel.register("text_classifier")
class TextClassifier(BaseTorchModel):
    def __init__(
        self,
        num_classes: int,
        embedder: ftm.BaseEmbedder,
        encoder: ftm.BaseSequenceEncoder,
        vectorizer: ftm.BaseSequenceVectorizer,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._vectorizer = vectorizer
        self._dropout = torch.nn.Dropout(dropout)
        self._classifier = torch.nn.Linear(
            vectorizer.get_output_dim(), num_classes
        )
        self._loss = ftm.CrossEntropyLoss()

    def forward(self, inputs, params=None):
        embeddings, mask = self._embedder(inputs.text)
        embeddings = self._encoder(embeddings, mask=mask)
        vector = self._vectorizer(embeddings, mask=mask)
        vector = self._dropout(vector)
        logits = self._classifier(vector)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        label = logits.argmax(dim=-1)

        loss = None
        if inputs.label is not None:
            loss = self._loss(logits, inputs.label)

        return ClassifierOutput(probs=probs, label=label, loss=loss)

# 3. Prepare data
datamodule = TextClassificationDataModule(
    text=Tokenizer(),
    label=LabelIndexer(),
)

with datamodule.train():
    train_instances = [datamodule.instance(ex) for ex in train_data]
val_instances = [datamodule.instance(ex) for ex in val_data]

# 4. Create data loaders
train_loader = DataLoader(
    sampler=BasicBatchSampler(batch_size=32, shuffle=True),
    collator=datamodule.batch,
)

# 5. Define evaluator
class Evaluator:
    def __init__(self):
        self.accuracy = MulticlassAccuracy()

    def update(self, inputs, output):
        preds = output.label.tolist()
        targets = inputs.label.tolist()
        self.accuracy.update(
            self.accuracy.Input(predictions=preds, targets=targets)
        )

    def compute(self):
        return self.accuracy.compute()

    def reset(self):
        self.accuracy.reset()

# 6. Create model (can also be done from configuration)
model = TextClassifier(
    num_classes=2,
    embedder=ftm.TokenEmbedder(
        embedding_dim=128,
        vocab_size=10000,
    ),
    encoder=ftm.LSTMSequenceEncoder(
        input_size=128,
        hidden_size=256,
        num_layers=2,
        bidirectional=True,
    ),
    vectorizer=ftm.BagOfEmbeddingsSequenceVectorizer(),
    dropout=0.2,
)

# 7. Create trainer
trainer = TorchTrainer(
    train_dataloader=train_loader,
    engine=DefaultTorchTrainingEngine(
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        loss="loss",  # Extract loss from ClassifierOutput
    ),
    max_epochs=10,
    eval_strategy="epoch",
    callbacks=[
        EvaluationCallback(Evaluator()),
        EarlyStoppingCallback(patience=3, metric="-loss"),
    ],
)

# 8. Train model
state = trainer.train(model, train_instances, val_instances)
```

## Best Practices

### Model Design

- **Use reusable modules**: Leverage `formed.integrations.torch.modules` for common components
- **Return structured output**: Use dataclasses instead of dicts for type safety
- **Include loss in output**: Return loss from forward pass for automatic training
- **Make components optional**: Allow optional encoder/feedforward for flexibility
- **Keep forward pass composable**: Chain modules in a clear pipeline (embed → encode → vectorize → classify)

### Training Configuration

- Start with `DefaultTorchTrainingEngine` for standard training
- Use `eval_strategy="epoch"` for small datasets, `"step"` for large ones
- Set `logging_interval` based on dataset size (more frequent for larger datasets)
- Always use EvaluationCallback for metric tracking

### Distributed Training

- Use `DataParallelDistributor` for single-machine multi-GPU training
- Use `DistributedDataParallelDistributor` for multi-machine training
- Ensure batch size is divisible by number of GPUs
- Only save checkpoints from main process (distributor.is_main_process)

### Workflow Integration

- Use `torch::train` for training with automatic caching
- Reference DataModule's batch method as collator
- Set random seed for reproducibility
- Use device context managers for device management
