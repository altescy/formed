# ML Integration Guide

## Overview

The `formed.integrations.ml` module provides framework-agnostic machine learning utilities organized into three main components:

- **Data Transformation**: Type-safe, composable transformations with automatic batching using `BaseTransform` and `DataModule`
- **Metrics**: Various metrics for classification, regression, and ranking tasks with a consistent interface
- **DataLoader**: Utilities for efficient data loading and batching in training pipelines

## Data Transformation

### Overview

The data transformation system provides a structured way to convert raw data into model-ready batches:

- **BaseTransform**: Base class for field-level transformations (e.g., tokenization, label indexing)
- **DataModule**: Container for composing multiple transforms with nested structure preservation
- **Transformation pipeline**: Raw data → `instance()` → batchable instances → `batch()` → batches

### Basic Example

```python
from typing import Any, Generic

from formed.common.iterutils import batched
from formed.integrations import ml

# Define field-level transformations using DataModule
# Fields with BaseTransform types are automatically batched
class TextClassificationDataModule(ml.DataModule):
    id: ml.MetadataTransform
    text: ml.Tokenizer
    label: ml.Extra[ml.LabelIndexer] = ml.Extra.default()  # Use Extra for optional fields (e.g., absent during inference)


train_dataset = [
    {"id": "1", "text": "I love programming.", "label": "positive"},
    {"id": "2", "text": "I hate bugs.", "label": "negative"},
]
test_dataset = [
    {"id": "3", "text": "Debugging is fun.", "label": "positive"},
    {"id": "4", "text": "I dislike errors.", "label": "negative"},
]

datamodule = TextClassificationDataModule(
    id=ml.MetadataTransform(),
    text=ml.Tokenizer(),
    label=ml.LabelIndexer(),
)

# Build vocabularies and label sets
with datamodule.train():
    # Convert raw data to batchable instances using datamodule.instance()
    train_instances = [datamodule.instance(example) for example in train_dataset]

test_instances = [datamodule.instance(example) for example in test_dataset]

# Create batches using datamodule.batch()
# The DataModule structure is preserved in batched form
for batch in map(datamodule.batch, batched(train_instances, batch_size=2)):
    print("Train Batch:", batch)
    # Train Batch: TextClassificationDataModule(
    #   id=['1', '2'],
    #   text=Tokenizer(
    #     surfaces=IDSequenceBatch(
    #       ids=array([[1, 4, 5], [1, 3, 2]]),
    #       mask=array([[ True,  True,  True], [ True,  True,  True]])
    #     ),
    #   ),
    #   label=array([0, 1]),
    # )
```

### BaseTransform

`BaseTransform` is the base class for all transformations, providing two key methods:

- **`instance(...)`**: Converts raw input data into a batchable instance
- **`batch(...)`**: Converts a sequence of instances into a batch

**Preprocessing with accessors:**

Use the `accessor` parameter to specify which field to process or apply preprocessing:

- `SomeTransform(..., accessor="path.to.field")` extracts data from nested structures using dot notation
- `SomeTransform(..., accessor=lambda raw: ...)` applies custom extraction logic with a function

This allows transforms to work with complex nested data structures without modifying the transform implementation.

### DataModule

`DataModule` is a transformation module for handling nested data structures:

- Inherit from `DataModule` and assign `BaseTransform` or nested `DataModule` instances to fields
- Each field defines how that data element should be transformed
- The structure is preserved through the transformation pipeline (raw → instance → batch)

**Key features:**

- **Composability**: Nest DataModules to handle complex hierarchical data
- **Type preservation**: Field structure is maintained across transformations
- **Training mode**: Use `with datamodule.train():` context to build vocabularies and other statistics

### Type System

DataModule has three modes that control field types and behavior:

- **`AsConverter`**: Data transformation mode providing `instance()` and `batch()` methods
  - This is the default mode when you create a DataModule instance
  - Provides methods for transforming data
- **`AsInstance`**: Data container mode for pre-batch data
  - `instance()` returns `DataModule[AsInstance]`
  - Represents individual examples ready for batching
- **`AsBatch`**: Data container mode for batched data
  - `batch()` returns `DataModule[AsBatch]`
  - Represents batched data ready for model input

Proper type annotations allow type checkers to track types across transformation stages:

```python
from typing import TypeVar
from formed.integrations.ml import types as mlt

T = TypeVar("T")

class TextClassificationDataModule(
    ml.DataModule[
        mlt.DataModuleModeT,                                # DataModule mode
        T,                                                  # Type accepted by instance()
        "TextClassificationDataModule[mlt.AsInstance]",     # Type returned by instance() / accepted by batch()
        "TextClassificationDataModule[mlt.AsBatch]",        # Type returned by batch()
    ]
):
    id: ml.MetadataTransform
    text: ml.Tokenizer
    label: ml.Extra[ml.LabelIndexer] = ml.Extra.default()

...

datamodule = TextClassificationDataModule(...)
datamodule.text.surfaces      # Type checker recognizes this as TokenSequenceIndexer
datamodule.text.surfaces.ids  # Type error!

batch = datamodule.batch(instances)
batch.text.surfaces     # Now recognized as IDSequenceBatch
batch.text.surfaces.ids # OK: numpy.ndarray
```

### Special Field Types

The ML integration provides special field descriptors for common patterns:

- **`Extra`**: Marks fields that may be absent in some contexts (e.g., labels during inference)
  - Use `Extra[SomeTransform]` to declare optional fields
  - Fields marked with `Extra` can be omitted from input data without errors
  - Example: `label: ml.Extra[ml.LabelIndexer] = ml.Extra.default()`

- **`Param`**: Marks fields that should not be transformed (passed through as-is)
  - Use `Param` for metadata or configuration that doesn't need transformation
  - These fields bypass the transformation pipeline

## Metrics

### Overview

The `ml.metrics` module implements various metrics for machine learning tasks with a consistent interface:

**Metric lifecycle:**
- **`update(...)`**: Update internal state with predictions and targets
- **`compute()`**: Calculate final metrics and return as a dictionary
- **`reset()`**: Reset internal state for the next evaluation round

This pattern supports incremental metric computation across multiple batches.

### Basic Usage

```python
from formed.integrations import ml

metric = ml.MulticlassAccuracy(average="macro")
metric.update(
    ml.MulticlassAccuracy.Input(
        predictions=[0, 2, 1, 3],
        targets=[0, 1, 2, 3],
    ),
)
metrics = metric.compute()  # => {"accuracy": 0.5}
metric.reset()
```

Available metrics include:

- **Classification**: Accuracy, Precision, Recall, F-beta, etc.
- **Regression**: MAE, MSE, RMSE, R², etc.
- **Ranking**: MRR, NDCG, etc.
- **Utilities**: Average (for tracking losses and other scalars)

### Building Evaluators

Use base metric classes (e.g., `MulticlassClassificationMetric`) to build reusable, configurable evaluators:

```python
from typing import NamedTuple, Sequence
import numpy

class ClassificationBatch(NamedTuple):
    data: ...
    label: numpy.ndarray | None


class ClassifierOutput(NamedTuple):
    label: numpy.ndarray
    loss: float | None


class ClassificationEvaluator:
    def __init__(self, metrics: Sequence[ml.MulticlassClassificationMetric]) -> None:
        self._loss = ml.Average("loss")
        self._metrics = metrics

    def update(
        self,
        inputs: ClassificationBatch,
        output: ClassifierOutput,
    ) -> None:
        if output.loss is not None:
            self._loss.update([output.loss])
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


evaluator = ClassificationEvaluator(
    metrics=[
        ml.MulticlassAccuracy(average="macro"),
        ml.MulticlassFBeta(average="macro"),
    ]
)
```

This pattern provides:

- **Common interface**: Unified API across different metric types
- **Configurable composition**: Mix and match metrics for different tasks
- **Task-specific logic**: Encapsulate evaluation logic per task

## DataLoader

### Overview

The `ml.DataLoader` class provides efficient data loading utilities for training pipelines. It combines three key components:

- **BatchSampler**: Generates batch indices (which items to include in each batch)
- **Collator**: Transforms a sequence of items into a batch
- **DataLoader**: Orchestrates sampling and collation with optional prefetching

This separation of concerns allows flexible configuration of batching strategies independent of data transformation logic.

### Core Components

#### BatchSampler

`BaseBatchSampler` generates sequences of indices for batching:

**BasicBatchSampler** - Standard batching with shuffle support:

```python
from formed.integrations import ml

sampler = ml.BasicBatchSampler(
    batch_size=32,
    shuffle=True,      # Shuffle data before batching
    drop_last=False,   # Keep incomplete final batch
    seed=0,            # Random seed for reproducibility
)
```

**SizeOrderedBucketBatchSampler** - Batch by item size for efficiency:

```python
sampler = ml.SizeOrderedBucketBatchSampler(
    attribute="text",  # Attribute to determine size (or callable)
    batch_size=32,
    shuffle=True,      # Shuffle batches (not items within batches)
    drop_last=False,
)
```

This sampler:

1. Sorts items by size (e.g., sequence length)
2. Groups consecutive items into batches
3. Optionally shuffles the batches

Benefits:

- Reduces padding in batches (items have similar sizes)
- Improves training efficiency and memory usage
- Particularly useful for variable-length sequences

#### Collator

The collator is a function that transforms a sequence of items into a batch. Common patterns:

**Using DataModule's batch method:**

```python
# DataModule.batch is a perfect collator
collator = datamodule.batch
```

**Custom collator function:**

```python
import numpy as np

def custom_collator(items):
    # Stack features and labels
    features = np.stack([item.features for item in items])
    labels = np.array([item.label for item in items])
    return {"features": features, "labels": labels}
```

#### DataLoader

`DataLoader` combines sampler and collator to create batched iterators:

```python
from formed.integrations import ml

loader = ml.DataLoader(
    sampler=ml.BasicBatchSampler(batch_size=32, shuffle=True),
    collator=datamodule.batch,
    buffer_size=0,  # Optional prefetch buffer (see below)
)

# Create batched iterator from dataset
for batch in loader(instances):
    # Process batch
    ...
```

### Basic Usage

Complete example with DataModule integration:

```python
from formed.integrations import ml
from formed.common.iterutils import batched

# 1. Define DataModule
class MyDataModule(ml.DataModule):
    features: ml.TensorTransform
    label: ml.LabelIndexer

datamodule = MyDataModule(
    features=ml.TensorTransform(),
    label=ml.LabelIndexer(),
)

# 2. Create instances
with datamodule.train():
    train_instances = [datamodule.instance(ex) for ex in train_data]
test_instances = [datamodule.instance(ex) for ex in test_data]

# 3. Create DataLoader
train_loader = ml.DataLoader(
    sampler=ml.BasicBatchSampler(batch_size=32, shuffle=True),
    collator=datamodule.batch,
)

# 4. Iterate over batches
for batch in train_loader(train_instances):
    # batch is MyDataModule[AsBatch] with properly batched fields
    ...
```

### Advanced Features

#### Prefetch Buffering

When collation is expensive, use buffering to prefetch batches in a background process:

```python
from formed.common.ctxutils import closing

loader = ml.DataLoader(
    sampler=ml.BasicBatchSampler(batch_size=32, shuffle=True),
    collator=datamodule.batch,
    buffer_size=10,  # Prefetch up to 10 batches in background
)

# Use with context manager for proper cleanup
with closing(loader(instances)) as batches:
    for batch in batches:
        # Process batch while next batches are prepared
        ...
```

**Important notes:**

- Collator and referenced objects must be picklable (for multiprocessing)
- Always use context manager (`closing`) to ensure background process cleanup
- Adjust `buffer_size` based on collation cost and memory constraints

#### Size-Ordered Batching for Sequences

For variable-length sequences, size-ordered batching reduces padding:

```python
# Instances have varying text lengths
sampler = ml.SizeOrderedBucketBatchSampler(
    attribute="text.surfaces",  # Path to sequence field
    batch_size=32,
    shuffle=True,
)

loader = ml.DataLoader(sampler=sampler, collator=datamodule.batch)

for batch in loader(instances):
    # Batch contains items with similar lengths
    # Less padding → more efficient training
    ...
```

Alternative: use callable for custom size extraction:

```python
sampler = ml.SizeOrderedBucketBatchSampler(
    attribute=lambda item: len(item.text.surfaces.ids),
    batch_size=32,
)
```

## Workflow Integration

### DataModule in Workflows

DataModules can be constructed and used in Jsonnet workflow configurations:

```jsonnet
{
  steps: {
    # 1. Load IMDB dataset from Huggingface Datasets
    dataset: {
      type: 'datasets::load_dataset',
      path: 'stanfordnlp/imdb',
      split: 'train[:100]',
    },

    # 2. Build DataModule and create instances
    datamodule_and_instances: {
      type: 'ml::train_datamodule_with_instances',
      dataset: { type: 'ref', ref: 'dataset' },
      datamodule: {
        type: 'classification:TextClassificationDataModule',
        id: { accessor: { type: 'classification:ExampleID' } },
        text: {
          type: 'formed.integrations.ml:Tokenizer',
          surfaces: { unk_token: '<UNK>', min_df: 3, max_vocab_size: 10000 },
          characters: { unk_token: '<UNK>', min_characters: 5 },
        },
        label: {},
      },
    },

    # 3. Use the constructed DataModule and instances
    model: {
      type: 'your_training_step',
      # Reference instances from previous step
      train_dataset: { type: 'ref', ref: 'datamodule_and_instances.instances' },
      train_dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        # Reference batch function from DataModule
        collator: { type: 'ref', ref: 'datamodule_and_instances.datamodule.batch' },
      },
      ...
    },
  },
}
```

### Workflow Patterns

**Using `ml::train_datamodule_with_instances`:**

This built-in step builds DataModule vocabularies and creates instances in one operation:

- Takes a dataset and a DataModule configuration
- Returns both the fitted DataModule and processed instances
- Ensures vocabularies are built correctly in training mode

**Field references:**

Access specific fields from previous steps using dot notation:

- `{ type: 'ref', ref: 'step_name.field_name' }` accesses nested fields
- Example: `'datamodule_and_instances.instances'` gets the instances field

**Method references:**

Reference DataModule methods directly in configurations:

- `{ type: 'ref', ref: 'step_name.datamodule.batch' }` references the batch method
- Allows passing transformation functions as parameters
- Enables configuration-driven pipeline construction
