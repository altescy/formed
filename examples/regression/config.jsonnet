{
  steps: {
    dataset: { type: 'load_dataset' },
    model: {
      type: 'flax::train',
      model: { type: 'tiny-regressor' },
      trainer: {
        train_dataloader: {
          collator: { type: 'regression:Collator' },
          batch_size: 64,
          shuffle: true,
          drop_last: true,
        },
        optimizer: {
          type: 'optax:sgd',
          learning_rate: 0.01,
        },
        max_epochs: 10,
        callbacks: [
          { type: 'mlflow' },
        ],
      },
      train_dataset: { type: 'ref', ref: 'dataset' },
    },
  },
}
