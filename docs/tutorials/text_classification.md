# Text Classification with PyTorch

This tutorial guides you through building a text classification system using formed's PyTorch integration. You'll learn how to compose models from reusable components, manage data transformations, and train models within formed's workflow system.

**What you'll build**: A binary text classifier that detects whether a sequence of characters is sorted or not (a toy task for demonstration).

**What you'll learn**:
- Defining DataModules for type-safe data transformation
- Composing models from pre-built torch modules
- Training models with callbacks and evaluation
- Integrating everything into a reproducible workflow

## Prerequisites

Install formed with PyTorch integration:

```bash
pip install formed[torch,mlflow]
```

## Project Structure

Create a new directory for this tutorial:

```bash
mkdir text_classification_tutorial
cd text_classification_tutorial
```

We'll create:

- `textclf.py` - DataModule, model, and evaluator definitions
- `config.jsonnet` - Workflow configuration
- `formed.yml` - Project settings

## Step 1: Define the DataModule

The DataModule handles data transformation from raw examples to model-ready batches. It provides a structured, type-safe way to define how each field should be processed.

Create `textclf.py`:

```python
from typing import Any
from collections.abc import Sequence
import dataclasses

from formed.integrations import ml
from formed.integrations.ml import types as mlt

# Define the raw data structure
@dataclasses.dataclass
class ClassificationExample:
    id: str
    text: str | Sequence[str]  # Can be string or tokens
    label: int | str | None = None

# Define the DataModule for text classification
@ml.DataModule.register("textclf::text_classification")
class TextClassificationDataModule(
    ml.DataModule[
        mlt.DataModuleModeT,
        Any,
        "TextClassificationDataModule[mlt.AsInstance]",
        "TextClassificationDataModule[mlt.AsBatch]",
    ]
):
    """DataModule for text classification tasks.

    Fields:
        id: Example identifier (metadata, not batched)
        text: Text to classify, processed through tokenization
        label: Classification label, indexed to integers
    """
    id: ml.MetadataTransform[Any, str] = ml.MetadataTransform()
    text: ml.Tokenizer  # Tokenizes and indexes text
    label: ml.Extra[ml.LabelIndexer] = ml.Extra.default()  # Optional during inference
```

**Key concepts**:

- **Field transforms**: Each field specifies its transformation type
  - `MetadataTransform`: Pass through metadata without batching
  - `Tokenizer`: Tokenize text and build vocabulary
  - `LabelIndexer`: Index labels to integers
- **Extra fields**: `label` is marked with `Extra` since it's absent during inference
- **Type parameters**: Generic parameters track types through transformation stages (AsConverter → AsInstance → AsBatch)

**How it works**:

1. During training, `with datamodule.train():` builds vocabularies from data
2. `datamodule(example)` converts raw examples to instances
3. `datamodule.batch(instances)` collates instances into batches
4. The DataModule structure is preserved at each stage

## Step 2: Define the Model

Models in formed are composed from reusable modules. This declarative approach separates architecture from implementation and enables configuration-driven experimentation.

Add to `textclf.py`:

```python
import torch
from formed.integrations import torch as ft
from formed.integrations.torch import modules as ftm

@dataclasses.dataclass
class ClassifierOutput:
    """Model output structure."""
    probs: torch.Tensor   # Class probabilities
    label: torch.Tensor   # Predicted labels
    loss: torch.Tensor | None = None  # Loss (if labels provided)

@ft.BaseTorchModel.register("textclf::torch_text_classifier")
class TextClassifier(ft.BaseTorchModel[
    TextClassificationDataModule[mlt.AsBatch],  # Input type
    ClassifierOutput,                            # Output type
]):
    """LSTM-based text classifier.

    Architecture:
        text → embedder → encoder → vectorizer → feedforward → classifier

    Args:
        num_classes: Number of classification labels
        embedder: Converts tokens to embeddings
        encoder: Processes token sequences with context
        vectorizer: Aggregates sequence to fixed-size vector
        feedforward: Optional additional transformation
        dropout: Dropout probability
        loss: Loss function for training
    """

    def __init__(
        self,
        num_classes: int,
        embedder: ftm.BaseEmbedder,
        vectorizer: ftm.BaseSequenceVectorizer,
        encoder: ftm.BaseSequenceEncoder | None = None,
        feedforward: ftm.FeedForward | None = None,
        sampler: ftm.BaseLabelSampler | None = None,
        loss: ftm.BaseClassificationLoss | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Use defaults for optional components
        sampler = sampler or ftm.ArgmaxLabelSampler()
        loss = loss or ftm.CrossEntropyLoss()

        # Calculate feature dimension through the pipeline
        # determine_ndim chains output dimensions, handling optional components
        feature_dim = ft.determine_ndim(
            embedder.get_output_dim(),
            encoder.get_output_dim() if encoder is not None else None,
            vectorizer.get_output_dim(),
            feedforward.get_output_dim() if feedforward is not None else None,
        )

        # Store components
        self._embedder = embedder
        self._encoder = encoder
        self._vectorizer = vectorizer
        self._feedforward = feedforward
        self._dropout = torch.nn.Dropout(dropout)
        self._classifier = torch.nn.Linear(feature_dim, num_classes)
        self._sampler = sampler
        self._loss = loss

    def forward(
        self,
        inputs: TextClassificationDataModule[mlt.AsBatch],
        params: None = None,
    ) -> ClassifierOutput:
        """Forward pass through the model.

        Args:
            inputs: Batched data from DataModule
            params: Additional parameters (unused)

        Returns:
            ClassifierOutput with predictions and loss
        """
        # Embed tokens: (batch, seq_len) → (batch, seq_len, embed_dim)
        embeddings, mask = self._embedder(inputs.text)

        # Encode sequence with context (optional)
        if self._encoder is not None:
            embeddings = self._encoder(embeddings, mask=mask)

        # Vectorize sequence: (batch, seq_len, dim) → (batch, dim)
        vector = self._vectorizer(embeddings, mask=mask)

        # Apply feedforward (optional)
        if self._feedforward is not None:
            vector = self._feedforward(vector)

        # Apply dropout and classify
        vector = self._dropout(vector)
        logits = self._classifier(vector)

        # Get probabilities and predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        label = self._sampler(logits)

        # Compute loss if labels provided
        loss = None
        if inputs.label is not None:
            loss = self._loss(logits, inputs.label)

        return ClassifierOutput(probs=probs, label=label, loss=loss)
```

**Key concepts**:

- **Module composition**: Models are built from reusable components
  - `BaseEmbedder`: Token → embedding
  - `BaseSequenceEncoder`: Contextual sequence processing (LSTM, Transformer, etc.)
  - `BaseSequenceVectorizer`: Sequence → fixed vector
  - `FeedForward`: Additional transformation layers
- **Structured output**: Return dataclass instead of dict for type safety
- **Loss in forward**: Including loss in output enables automatic training
- **Optional components**: Make encoder/feedforward optional for flexibility

## Step 3: Define the Evaluator

Evaluators compute metrics during training and evaluation. They follow a standard update-compute-reset pattern.

Add to `textclf.py`:

```python
class ClassificationEvaluator:
    """Evaluator for classification tasks.

    Tracks loss and configurable classification metrics (accuracy, F-beta, etc.)

    Args:
        metrics: List of classification metrics to compute
    """

    def __init__(
        self,
        metrics: Sequence[ml.MulticlassClassificationMetric],
    ) -> None:
        self._loss = ml.Average("loss")
        self._metrics = metrics

    def update(
        self,
        inputs: TextClassificationDataModule[mlt.AsBatch],
        output: ClassifierOutput,
    ) -> None:
        """Update metrics with a batch of predictions.

        Args:
            inputs: Input batch (contains labels)
            output: Model predictions and loss
        """
        # Track loss
        if output.loss is not None:
            self._loss.update([output.loss.item()])

        # Track classification metrics
        if inputs.label is not None:
            predictions = output.label.tolist()
            targets = inputs.label.tolist()
            for metric in self._metrics:
                metric.update(
                    metric.Input(predictions=predictions, targets=targets)
                )

    def compute(self) -> dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary of metric names to values
        """
        metrics = self._loss.compute()
        for metric in self._metrics:
            metrics.update(metric.compute())
        return metrics

    def reset(self) -> None:
        """Reset metrics for next evaluation round."""
        self._loss.reset()
        for metric in self._metrics:
            metric.reset()
```

**Key concepts**:

- **Standard interface**: All evaluators implement update/compute/reset
- **Incremental computation**: Metrics accumulate over batches
- **Configurable metrics**: Accept list of metrics for flexibility

## Step 4: Create Sample Data

Before defining the workflow, let's create a simple data generation function.

Add to `textclf.py`:

```python
import random
from formed import workflow

@workflow.step("textclf::generate_sort_detection_dataset")
def generate_sort_detection_dataset(
    vocab: Sequence[str] = "abcdefghijklmnopqrstuvwxyz",
    num_examples: int = 100,
    max_tokens: int = 10,
    random_seed: int = 42,
) -> list[ClassificationExample]:
    """Generate synthetic dataset for sort detection.

    Creates examples with random character sequences labeled as
    'sorted' or 'not_sorted' based on alphabetical order.

    Args:
        vocab: Characters to sample from
        num_examples: Number of examples to generate
        max_tokens: Maximum sequence length
        random_seed: Random seed for reproducibility

    Returns:
        List of ClassificationExample instances
    """
    rng = random.Random(random_seed)
    examples = []

    for _ in range(num_examples):
        num_tokens = rng.randint(1, max_tokens)
        label = rng.choice(["sorted", "not_sorted"])
        tokens = rng.choices(vocab, k=num_tokens)

        if label == "sorted":
            tokens.sort()

        examples.append(
            ClassificationExample(
                id=str(len(examples)),
                text=tokens,
                label=label,
            )
        )

    return examples
```

**Key concepts**:

- **Workflow steps**: Functions decorated with `@workflow.step` become cacheable workflow steps
- **Deterministic**: Use explicit random seed for reproducibility
- **Typed output**: Return structured data that DataModule can process

## Step 5: Configure the Workflow

Now define the complete workflow in Jsonnet. This configuration specifies all steps from data generation to model training.

Create `config.jsonnet`:

```jsonnet
// Helper for step references
local ref(name) = { type: 'ref', ref: name };

// Define evaluator (reused across steps)
local evaluator = {
  type: 'textclf:ClassificationEvaluator',
  metrics: [
    { type: 'accuracy' },
    { type: 'fbeta' },  // F1 score by default
  ],
};

{
  steps: {
    // 1. Generate datasets
    train_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 1000,
      random_seed: 1,
    },

    val_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 100,
      random_seed: 2,
    },

    test_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 100,
      random_seed: 3,
    },

    // 2. Build DataModule and create instances
    datamodule: {
      type: 'ml::train_datamodule',
      datamodule: {
        type: 'textclf::text_classification',
        id: {},  // Use defaults
        text: {
          surfaces: {},  // Build vocabulary from training data
        },
        label: {},  // Build label set from training data
      },
      dataset: ref('train_dataset'),
    },

    // 3. Train the model
    model: {
      type: 'torch::train',
      model: {
        type: 'textclf::torch_text_classifier',
        num_classes: ref('datamodule.label.num_labels'),

        // Embedder: token IDs → embeddings
        embedder: {
          type: 'analyzed_text',  // Handles AnalyzedText from Tokenizer
          surface: {
            type: 'token',
            initializer: {
              type: 'xavier_uniform',
              shape: [
                ref('datamodule.text.surfaces.vocab_size'),
                32,  // embedding dimension
              ],
            },
            padding_idx: ref('datamodule.text.surfaces.pad_index'),
          },
        },

        // Encoder: sequence processing
        encoder: {
          type: 'lstm',
          input_dim: 32,
          hidden_dim: 32,
          bidirectional: false,
        },

        // Vectorizer: sequence → vector
        vectorizer: {
          type: 'boe',  // Bag of embeddings
          pooling: 'last',  // Use last hidden state
        },

        dropout: 0.1,
      },

      // Training configuration
      trainer: {
        // Data loaders
        train_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: {
            type: 'basic',
            batch_size: 32,
            shuffle: true,
            drop_last: true,
          },
          collator: ref('datamodule.batch'),
        },

        val_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: {
            type: 'basic',
            batch_size: 32,
            drop_last: false,
          },
          collator: ref('datamodule.batch'),
        },

        // Training engine
        engine: {
          type: 'default',
          optimizer: {
            type: 'torch.optim:Adam',
            lr: 1e-3,
          },
        },

        // Callbacks
        callbacks: [
          // Log metrics to MLflow
          { type: 'mlflow' },

          // Compute evaluation metrics
          {
            type: 'evaluation',
            evaluator: evaluator,
          },

          // Early stopping on validation F-beta
          {
            type: 'early_stopping',
            patience: 3,
            metric: '+val/fbeta',  // '+' means maximize
          },
        ],

        // Training settings
        max_epochs: 10,
        logging_strategy: 'step',
        logging_interval: 5,
      },

      // Reference datasets
      train_dataset: ref('train_dataset'),
      val_dataset: ref('val_dataset'),
    },

    // 4. Evaluate on test set
    test_metrics: {
      type: 'torch::evaluate',
      model: ref('model'),
      evaluator: evaluator,
      dataset: ref('test_dataset'),
      dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        sampler: {
          type: 'basic',
          batch_size: 32,
          shuffle: false,
          drop_last: false,
        },
        collator: ref('datamodule.batch'),
      },
      random_seed: 0,
    },
  },
}
```

**Key concepts**:

- **Step dependencies**: Use `ref()` to reference other steps' outputs
- **Nested references**: Access fields with dot notation (e.g., `datamodule.label.num_labels`)
- **Method references**: Reference DataModule methods as collators (`datamodule.batch`)
- **Metric specifications**: Prefix with `+` to maximize, `-` to minimize
- **Declarative configuration**: Entire architecture specified in configuration

## Step 6: Configure Project Settings

Create `formed.yml` to specify required modules:

```yaml
workflow:
  organizer:
    type: mlflow
    log_execution_metrics: true

required_modules:
  - textclf
  - formed.integrations.datasets
  - formed.integrations.ml
  - formed.integrations.mlflow
  - formed.integrations.torch
```

**Key concepts**:

- **MLflow organizer**: Automatically logs experiments and artifacts
- **Required modules**: Import modules containing step definitions
- **Execution metrics**: Track training progress in MLflow

## Step 7: Run the Workflow

Execute the workflow:

```bash
formed workflow run config.jsonnet --execution-id sort-classifier-v1
```

**What happens**:

1. **Data generation**: Creates train/val/test datasets
2. **DataModule training**: Builds vocabulary and label index from training data
3. **Model training**: Trains with early stopping and metric logging
4. **Test evaluation**: Computes final metrics on held-out test set
5. **Caching**: Results are cached by fingerprint for reproducibility

**View results**:

```bash
mlflow ui
```

Then open http://localhost:5000 to see:

- Training curves (loss, accuracy, F-beta)
- Hyperparameters
- Model artifacts
- System metrics

You can access cached results later with the same execution ID:

```python
from formed.settings import load_formed_settings
from formed.workflow import WorkflowExecutionID

settings = load_formed_settings("./formed.yml")
organizer = settings.workflow.organizer

context = organizer.get(WorkflowExecutionID("004e597f"))
model = context.cache[context.info.graph["model"]]
```

## Next Steps

### Experiment with Configuration

**Try different architectures**:

```jsonnet
// Bidirectional LSTM
encoder: {
  type: 'lstm',
  input_dim: 32,
  hidden_dim: 64,
  num_layers: 2,
  bidirectional: true,
}
```

**Add feedforward layers**:

```jsonnet
model: {
  type: 'textclf::torch_text_classifier',
  // ...
  feedforward: {
    type: 'feedforward',
    input_dim: 32,
    hidden_dims: [64, 32],
    activations: 'relu',
    dropout: 0.2,
  },
}
```

**Use different optimizers**:

```jsonnet
engine: {
  type: 'default',
  optimizer: {
    type: 'torch.optim:AdamW',
    lr: 1e-3,
    weight_decay: 0.01,
  },
  lr_scheduler: {
    type: 'formed.integrations.torch:CosineLRScheduler',
    t_initial: 100,
    lr_min: 1e-5,
  },
}
```

### Real-World Dataset

Replace synthetic data with actual text classification:

```jsonnet
train_dataset: {
  type: 'datasets::load',
  path: 'dair-ai/emotion',
  split: 'train',
}
```

Update the DataModule to use pre-trained tokenizer:

```jsonnet
datamodule: {
  type: 'textclf::text_classification',
  text: {
    type: 'transformers::convert_tokenizer',
    tokenizer: 'bert-base-uncased',
  },
  label: {},
}
```

Use pre-trained models:

```jsonnet
embedder: {
  type: 'analyzed_text',
  surface: {
    type: 'pretrained_transformer',
    model: 'bert-base-uncased',
  },
}
```

## Key Takeaways

**DataModule**:

- Provides type-safe, composable data transformation
- Structure preserved through transformation pipeline
- Training mode builds vocabularies automatically

**Model Composition**:

- Build models from reusable, configurable modules
- Declarative architecture specification
- Automatic dimension tracking through pipeline

**Training**:

- Integrated callbacks for evaluation and early stopping
- Automatic metric logging with MLflow
- Flexible training strategies (epoch/step-based)

**Workflow**:

- Content-based caching ensures reproducibility
- Automatic dependency tracking
- Configuration-driven experimentation

## Further Reading

- **[ML Integration Guide](../guides/integrations/ml.md)**: Deep dive into DataModule and metrics
- **[PyTorch Integration Guide](../guides/integrations/torch.md)**: Complete PyTorch module reference
- **[Workflow Guide](../guides/workflow.md)**: Advanced workflow patterns and caching
- **[API Reference](../api_reference/index.md)**: Detailed API documentation

For the complete example with more features, see `examples/text_classification/` in the repository.
