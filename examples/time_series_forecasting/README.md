# Time Series Forecasting Example

This example demonstrates time series forecasting using PyTorch and formed's workflow system.

## Task Description

Given a sequence of time series values (sin wave with noise), predict the next value in the sequence.

**Key Features:**
- Synthetic data generation with configurable noise and patterns
- LSTM-based forecasting model
- Workflow-based training with automatic caching
- Early stopping and model evaluation

## Quick Start

### Method 1: Run directly (without workflow)

```bash
uv run python examples/time_series_forecasting/time_series.py
```

Options:
```bash
uv run python examples/time_series_forecasting/time_series.py \
    --epochs 20 \
    --batch-size 32 \
    --hidden-dim 64 \
    --num-layers 2 \
    --dropout 0.2
```

### Method 2: Run with workflow system

First, create `formed.yml` in the working directory:

```yaml
workflow:
  organizer:
    type: filesystem

required_modules:
  - examples.time_series_forecasting.time_series
```

Then run the workflow:

```bash
uv run formed workflow run examples/time_series_forecasting/config.jsonnet \
    --execution-id time-series-experiment
```

## Workflow Structure

The workflow consists of 5 steps:

1. **train_dataset**: Generate training data (1000 examples)
2. **val_dataset**: Generate validation data (200 examples)
3. **test_dataset**: Generate test data (200 examples)
4. **model**: Train LSTM model using `torch::train` step
5. **test_metrics**: Evaluate model using `torch::evaluate` step

All steps are automatically cached based on their configurations and dependencies.

## Model Architecture

- **Input**: Sequence of float values (default: 20 timesteps)
- **LSTM**: 2-layer bidirectional LSTM (64 hidden units)
- **FeedForward**: 2-layer MLP for final prediction
- **Output**: Single predicted value for next timestep

## Evaluation Metrics

- **MSE** (Mean Squared Error): Primary optimization metric
- **MAE** (Mean Absolute Error): Additional evaluation metric

## Customization

### Modify data generation

Edit `config.jsonnet`:

```jsonnet
train_dataset: {
  type: 'generate_sinusoid_dataset',
  num_examples: 2000,        // More examples
  sequence_length: 30,       // Longer sequences
  num_frequencies: 5,        // More patterns
  noise_level: 0.05,         // Less noise
  random_seed: 42,
}
```

### Modify model architecture

Edit `config.jsonnet`:

```jsonnet
model: {
  type: 'torch::train',
  model: {
    type: 'time_series:TimeSeriesForecaster',
    input_dim: 1,
    hidden_dim: 128,    // Larger model
    num_layers: 3,      // Deeper LSTM
    dropout: 0.3,       // More regularization
  },
  // ...
}
```

## Expected Results

With default settings, you should see:
- Training MSE: ~0.05-0.10
- Validation MSE: ~0.05-0.10
- Test MSE: ~0.05-0.10
- Prediction errors typically < 0.3 for individual samples

## Workflow Benefits

1. **Automatic Caching**: Re-running with same config loads cached results
2. **Dependency Tracking**: Only changed steps are re-executed
3. **Reproducibility**: All random seeds and configurations are tracked
4. **Experiment Management**: Easy comparison between different configurations
