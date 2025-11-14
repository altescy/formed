// Time Series Forecasting Workflow Configuration
//
// This workflow demonstrates:
// 1. Dataset generation with automatic caching
// 2. Model training using torch::train workflow step
// 3. Model evaluation using torch::evaluate workflow step
// 4. Dependency management between steps

local ref(name) = { type: 'ref', ref: name };

// Evaluator configuration
local evaluator = {
  type: 'time_series:ForecastingEvaluator',
};

{
  steps: {
    // Step 1: Generate training dataset
    train_dataset: {
      type: 'generate_sinusoid_dataset',
      num_examples: 1000,
      sequence_length: 20,
      num_frequencies: 3,
      noise_level: 0.1,
      random_seed: 42,
    },

    // Step 2: Generate validation dataset
    val_dataset: {
      type: 'generate_sinusoid_dataset',
      num_examples: 200,
      sequence_length: 20,
      num_frequencies: 3,
      noise_level: 0.1,
      random_seed: 123,
    },

    // Step 3: Generate test dataset
    test_dataset: {
      type: 'generate_sinusoid_dataset',
      num_examples: 200,
      sequence_length: 20,
      num_frequencies: 3,
      noise_level: 0.1,
      random_seed: 456,
    },

    // Step 4: Create datamodule (for collation)
    datamodule: {
      type: 'ml::train_datamodule',
      datamodule: {
        type: 'time_series:TimeSeriesDataModule',
        id: {},
        sequence: {},
        target: {},
      },
      dataset: ref('train_dataset'),
    },

    // Step 5: Train model using torch::train workflow step
    model: {
      type: 'torch::train',
      model: {
        type: 'time_series:TimeSeriesForecaster',
        encoder: {
          type: 'lstm',
          input_dim: 1,
          hidden_dim: 64,
          num_layers: 2,
          bidirectional: true,
        },
        vectorizer: {
          type: 'boe',
          pooling: 'last',
        },
        dropout: 0.2,
      },
      trainer: {
        type: 'formed.integrations.torch:TorchTrainer',
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
            shuffle: false,
            drop_last: false,
          },
          collator: ref('datamodule.batch'),
        },
        optimizer: {
          type: 'torch.optim:Adam',
          lr: 0.001,
        },
        distributor: {
          type: 'single',
        },
        max_epochs: 20,
        eval_strategy: 'epoch',
        logging_strategy: 'epoch',
        callbacks: [
          {
            type: 'evaluation',
            evaluator: evaluator,
          },
          {
            type: 'early_stopping',
            patience: 5,
            metric: '-val/loss',
          },
          {
            type: 'mlflow',
          },
        ],
      },
      train_dataset: ref('train_dataset'),
      val_dataset: ref('val_dataset'),
      random_seed: 0,
    },

    // Step 6: Evaluate model on test set using torch::evaluate workflow step
    test_metrics: {
      type: 'torch::evaluate',
      model: ref('model'),
      evaluator: evaluator,
      dataset: ref('test_dataset'),
      dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        sampler: {
          type: 'formed.integrations.ml:BasicBatchSampler',
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
