{
  steps: {
    train_dataset: { type: 'load_dataset', size: 200 },
    val_dataset: { type: 'load_dataset', size: 100 },
    datamodule: {
      type: 'formedml::build_datamodule',
      dataset: { type: 'ref', ref: 'train_dataset' },
    },
    model: {
      type: 'flax::train',
      model: {
        type: 'tiny-regressor',
        hidden_dim: 32,
      },
      trainer: {
        train_dataloader: {
          batch_sampler: {
            type: 'basic',
            batch_size: 64,
            shuffle: true,
            drop_last: true,
          },
        },
        optimizer: {
          type: 'optax:sgd',
          learning_rate: 0.01,
        },
        max_epochs: 120,
        callbacks: [
          { type: 'mlflow' },
          { type: 'early_stopping' },
        ],
      },
      train_dataset: { type: 'ref', ref: 'train_dataset' },
      val_dataset: { type: 'ref', ref: 'val_dataset' },
    },
  },
}
