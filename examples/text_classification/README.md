# Text Classification Example

This example demonstrates text classification using both PyTorch and Flax with formed's workflow system and MLflow integration.

## Task Description

The example includes two classification tasks:

1. **Sort Detection (Toy Task)**: Binary classification to detect whether a sequence of characters is sorted or not. Used for quick testing and development.
2. **Emotion Classification**: Multi-class classification on the DAIR-AI emotion dataset (6 emotions: sadness, joy, love, anger, fear, surprise).

**Key Features:**

- Support for both PyTorch and Flax backends
- Pre-trained transformer fine-tuning (BERT-based)
- Custom LSTM-based text classifier for toy tasks
- MLflow experiment tracking and artifact logging
- Workflow-based training with automatic content-based caching
- Early stopping and comprehensive evaluation metrics
- Balanced loss weighting by label distribution

## Quick Start

### Prerequisites

Install formed with required dependencies:

```bash
# For PyTorch backend
uv sync --extra torch --extra transformers --extra datasets --extra mlflow

# For Flax backend (in addition to above)
uv sync --extra flax --extra torch --extra transformers --extra datasets --extra mlflow
```

### Run Toy Example (PyTorch)

```bash
cd examples/text_classification
uv run formed workflow run configs/toy_torch.jsonnet --execution-id toy-torch-exp
```

### Run Toy Example (Flax)

```bash
cd examples/text_classification
uv run formed workflow run configs/toy_flax.jsonnet --execution-id toy-flax-exp
```

### Run Emotion Classification with Pre-trained Model

```bash
cd examples/text_classification
uv run formed workflow run configs/finetuning_torch.jsonnet --execution-id emotion-bert
```

### View MLflow Results

The example uses MLflow for experiment tracking. After running, you can view results:

```bash
cd examples/text_classification
uv run mlflow ui
```

Then open http://localhost:5000 in your browser.

## Configuration Files

### `configs/toy_torch.jsonnet`

Simple LSTM-based classifier for sort detection task:

- Custom embeddings (32-dim)
- Single-layer LSTM encoder (32 hidden units)
- Trained on 1000 synthetic examples
- Validates/tests on 50 examples each

### `configs/toy_flax.jsonnet`

Flax version of the toy example with similar architecture:

- Demonstrates Flax integration with formed
- Uses balanced loss weighting
- Early stopping on validation accuracy

### `configs/finetuning_torch.jsonnet`

Fine-tuning BERT-base-uncased on emotion classification:

- Loads DAIR-AI emotion dataset via Hugging Face datasets
- Uses pre-trained BERT transformer embeddings
- Batch size: 128 with size-ordered bucketing
- Learning rate: 3e-4 with cosine annealing and warmup
- Mixed precision training (bfloat16)
- Per-label metrics for detailed evaluation

## Workflow Structure

All configurations follow a similar workflow structure:

1. **Dataset Steps**: Generate or load train/validation/test datasets
2. **Datamodule**: Configure data preprocessing and transformations
3. **Model Training**: Train model using `torch::train` or `flax::train`
4. **Evaluation**: Compute test metrics using `torch::evaluate` or `flax::evaluate`

All steps are automatically cached based on their fingerprints (source code + parameters).

## Model Architecture

### Toy Task (LSTM-based)

**PyTorch/Flax:**

- **Embedder**: Token embeddings (vocab size × 32)
- **Encoder**: LSTM (32 hidden units, unidirectional)
- **Vectorizer**: Bag-of-embeddings with last-token pooling
- **Classifier**: Linear layer to 2 classes
- **Dropout**: 0.1-0.2

### Fine-tuning Task (Transformer-based)

**PyTorch:**

- **Embedder**: BERT-base-uncased (12 layers, 768 hidden units)
- **Vectorizer**: First token ([CLS]) pooling
- **Classifier**: Linear layer to 6 classes
- **Dropout**: 0.15
- **Loss**: Cross-entropy with balanced class weights

## Data Module

The `TextClassificationDataModule` handles data preprocessing:

```python
@ml.DataModule.register("textclf::text_classification")
class TextClassificationDataModule:
    text: TextTransform  # Tokenizer and text transformations
    id: MetadataTransform  # Example IDs (metadata only)
    label: LabelIndexer  # Label indexing and vocabulary
```

**Features:**

- Automatic vocabulary building from training data
- Support for pre-trained tokenizers (BERT, etc.)
- Label distribution tracking for balanced loss
- Batch collation with proper padding

## Evaluation Metrics

### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **F-beta Score**: Harmonic mean of precision and recall
- **Per-label Metrics**: Individual metrics for each class (fine-tuning only)

### Custom Evaluator

```python
ClassificationEvaluator(
    metrics=[accuracy, fbeta],       # Overall metrics
    label_metrics=[accuracy, fbeta],  # Per-class metrics
    datamodule=datamodule,            # For label reconstruction
)
```

## Customization

### Modify Dataset Size

Edit the configuration:

```jsonnet
train_dataset: {
  type: 'textclf::generate_sort_detection_dataset',
  num_examples: 5000,    // More training examples
  max_tokens: 20,        // Longer sequences
  random_seed: 42,
}
```

### Modify Model Architecture

For LSTM-based models:

```jsonnet
model: {
  type: 'torch::train',
  model: {
    type: 'textclf::torch_text_classifier',
    encoder: {
      type: 'lstm',
      input_dim: 32,
      hidden_dim: 64,     // Larger hidden size
      num_layers: 2,      // Deeper network
      bidirectional: true, // Use bidirectional LSTM
    },
    dropout: 0.3,         // More regularization
  },
}
```

For transformer-based models:

```jsonnet
local backbone = 'roberta-base';  // Use RoBERTa instead of BERT

model: {
  model: {
    embedder: {
      surface: {
        type: 'pretrained_transformer',
        model: backbone,
        layer_to_use: 'all',  // Use all layers
      },
    },
  },
}
```

### Modify Training Settings

```jsonnet
trainer: {
  engine: {
    optimizer: { type: 'torch.optim:AdamW', lr: 5e-5 },  // Different optimizer
    dtype: 'float32',  // Use full precision
  },
  callbacks: [
    { type: 'early_stopping', patience: 5, metric: '+val/accuracy' },
  ],
  max_epochs: 20,
  logging_interval: 10,
}
```

## Expected Results

### Toy Task (Sort Detection)

With default settings:

- Training accuracy: ~95-100%
- Validation accuracy: ~90-100%
- Test accuracy: ~90-100%
- Fast convergence (typically 3-5 epochs)

### Fine-tuning Task (Emotion Classification)

With BERT-base-uncased:

- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- Test F-beta score: ~0.85-0.90
- Per-class F-beta varies by emotion (joy and sadness typically highest)

## Workflow Benefits

1. **Content-based Caching**: Results are cached by fingerprint. Re-running with identical configuration loads cached results instantly.
2. **Automatic Dependency Tracking**: Only modified steps and their dependents are re-executed.
3. **MLflow Integration**: All experiments, metrics, and artifacts are automatically logged.
4. **Reproducibility**: Git info, environment details, and random seeds are tracked.
5. **Easy Experimentation**: Change any step configuration and re-run to see the impact.

## Implementation Details

### Custom Steps

The example defines a custom workflow step for data generation:

```python
@workflow.step("textclf::generate_sort_detection_dataset")
def generate_sort_detection_dataset(
    vocab: Sequence[str] = "abcdefghijklmnopqrstuvwxyz",
    num_examples: int = 100,
    max_tokens: int = 10,
    random_seed: int = 42,
) -> list[ClassificationExample]:
    # Generate synthetic sort detection examples
    ...
```

### Model Registration

Models are registered using formed's component system:

```python
@ft.BaseTorchModel.register("textclf::torch_text_classifier")
class TextClassifier(ft.BaseTorchModel):
    def forward(self, inputs, params=None):
        # Model forward pass
        ...
```

### Integration with MLflow

The workflow organizer is configured to use MLflow:

```yaml
# formed.yml
workflow:
  organizer:
    type: mlflow
    log_execution_metrics: true
```

This automatically logs:

- Training/validation metrics at each step
- Model checkpoints and best state
- Execution metadata (git commit, packages, etc.)
- Step configurations and outputs

## Troubleshooting

### Out of Memory

Reduce batch size in the configuration:

```jsonnet
trainer: {
  train_dataloader: {
    sampler: { batch_size: 32 },  // Smaller batches
  },
}
```

### Slow Training

For the toy task, ensure you're not using large models:

- Check embedding dimension (32 is sufficient)
- Check LSTM hidden size (32 is sufficient)
- Reduce dataset size for quick experiments

For fine-tuning:

- Use gradient accumulation if reducing batch size
- Consider using a smaller backbone (distilbert-base-uncased)
- Enable mixed precision training (bfloat16 or float16)

### MLflow UI Not Working

Make sure you're in the correct directory:

```bash
cd examples/text_classification
uv run mlflow ui
```

The MLflow tracking data is stored in `mlruns/` and `mlflow.db` in the example directory.
