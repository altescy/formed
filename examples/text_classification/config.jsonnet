{
  steps: {
    train_dataset: {
      type: 'datasets::load_dataset',
      path: 'stanfordnlp/imdb',
      split: 'train',
    },
    model: {
      type: 'flax::train',
      train_dataset: { type: 'ref', ref: 'train_dataset' },
      val_dataset: 0.2,
      model: {
        type: 'text_classifier',
        hidden_dim: 32,
        dropout: 0.1,
      },
      trainer: {
        max_epochs: 10,
        callbacks: [
          { type: 'mlflow' },
          { type: 'early_stopping', patience: 5, metric: 'accuracy' },
        ],
        logging_strategy: 'step',
        logging_interval: 100,
        eval_strategy: 'step',
        eval_interval: 200,
      },
    },
  },
}
