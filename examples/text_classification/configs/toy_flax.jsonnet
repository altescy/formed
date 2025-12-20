local ref(name) = { type: 'ref', ref: name };
local evaluator = {
  type: 'textclf.evaluators:ClassificationEvaluator',
  metrics: [
    { type: 'accuracy' },
    { type: 'fbeta' },
  ],
};

{
  steps: {
    train_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 1000,
      random_seed: 1,
    },
    val_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 50,
      random_seed: 2,
    },
    test_dataset: {
      type: 'textclf::generate_sort_detection_dataset',
      num_examples: 50,
      random_seed: 3,
    },
    datamodule: {
      type: 'ml::train_datamodule',
      datamodule: {
        type: 'textclf::text_classification',
        id: {},
        text: {
          type: 'tokenizer',
          surfaces: {},
        },
        label: {},
      },
      dataset: ref('train_dataset'),
    },
    model: {
      type: 'flax::train',
      model: {
        type: 'text_classification:TextClassifier',
        num_classes: ref('datamodule.label.num_labels'),
        embedder: {
          type: 'analyzed_text',
          surface: {
            type: 'token',
            vocab_size: ref('datamodule.text.surfaces.vocab_size'),
            embedding_dim: 32,
          },
        },
        encoder: {
          type: 'lstm',
          features: 32,
          num_layers: 1,
          bidirectional: false,
        },
        vectorizer: {
          type: 'boe',
          pooling: 'last',
        },
        loss: {
          type: 'cross_entropy',
          weighter: {
            type: 'balanced_by_distribution',
            distribution: ref('datamodule.label.distribution'),
          },
        },
        dropout: 0.2,
      },
      trainer: {
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
            type: 'formed.integrations.ml:BasicBatchSampler',
            batch_size: 32,
            shuffle: false,
            drop_last: false,
          },
          collator: ref('datamodule.batch'),
        },
        callbacks: [
          {
            type: 'evaluation',
            evaluator: evaluator,
          },
          {
            type: 'early_stopping',
            patience: 5,
            metric: 'val/accuracy',
          },
          {
            type: 'mlflow',
          },
        ],
      },
      train_dataset: ref('train_dataset'),
      val_dataset: ref('val_dataset'),
    },
    test_metrics: {
      type: 'flax::evaluate',
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
    },
  },
}
