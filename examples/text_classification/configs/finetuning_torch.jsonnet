local ref(name) = { type: 'ref', ref: name };

local backbone = 'distilbert-base-uncased';
local batch_size = 128;
local learning_rate = 3e-4;

local model(num_classes, vocab_size) = {
  type: 'textclf::torch_text_classifier',
  num_classes: num_classes,
  embedder: {
    type: 'analyzed_text',
    surface: {
      type: 'pretrained_transformer',
      model: backbone,
      layer_to_use: 'all',
    },
  },
  vectorizer: { type: 'boe', pooling: ['first'] },
  dropout: 0.15,
  loss: {
    type: 'cross_entropy',
    weighter: {
      type: 'balanced_by_distribution',
      distribution: ref('datamodule.label.distribution'),
    },
  },
};
local evaluator = {
  type: 'textclf.evaluators:ClassificationEvaluator',
  metrics: [
    { type: 'accuracy' },
    { type: 'fbeta' },
  ],
  label_metrics: [
    { type: 'accuracy' },
    { type: 'fbeta' },
  ],
  datamodule: ref('datamodule'),
};

{
  steps: {
    //
    // Pretrained Artifacts
    //
    tokenizer: {
      type: 'transformers::convert_tokenizer',
      tokenizer: backbone,
      bos_token: '[CLS]',
      eos_token: null,
    },
    //
    // Prepare dataset and datamodule
    //
    train_dataset: {
      type: 'datasets::load',
      path: 'dair-ai/emotion',
      split: 'train',
    },
    val_dataset: {
      type: 'datasets::load',
      path: 'dair-ai/emotion',
      split: 'validation',
    },
    test_dataset: {
      type: 'datasets::load',
      path: 'dair-ai/emotion',
      split: 'test',
    },
    datamodule: {
      type: 'ml::train_datamodule',
      dataset: ref('train_dataset'),
      datamodule: {
        type: 'textclf::text_classification',
        text: ref('tokenizer'),
        label: {},
      },
    },
    //
    // Generate instances
    //
    train_instances: {
      type: 'ml::generate_instances_without_caching',
      datamodule: ref('datamodule'),
      dataset: ref('train_dataset'),
    },
    val_instances: {
      type: 'ml::generate_instances_without_caching',
      datamodule: ref('datamodule'),
      dataset: ref('val_dataset'),
    },
    //
    // Train model
    //
    model: {
      type: 'torch::train',
      model: model(
        ref('datamodule.label.num_labels'),
        ref('datamodule.text.surfaces.vocab_size'),
      ),
      trainer: {
        train_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: { type: 'size_ordered_bucket', attribute: 'text.surfaces', batch_size: batch_size, drop_last: true },
          collator: ref('datamodule.batch'),
        },
        val_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: { type: 'size_ordered_bucket', attribute: 'text.surfaces', batch_size: batch_size, drop_last: false },
          collator: ref('datamodule.batch'),
        },
        engine: {
          type: 'default',
          optimizer: { type: 'torch.optim:Adam', lr: learning_rate },
          lr_scheduler: {
            type: 'formed.integrations.torch:CosineLRScheduler',
            t_initial: 1800,
            lr_min: 1e-5,
            warmup_t: 200,
            warmup_lr_init: 1e-4,
          },
          dtype: 'bfloat16',
          grad_scaler: {},
        },
        callbacks: [
          { type: 'mlflow' },
          { type: 'evaluation', evaluator: evaluator },
          { type: 'early_stopping', patience: 3, metric: '+val/fbeta' },
        ],
        max_epochs: 10,
        logging_strategy: 'step',
        logging_interval: 5,
      },
      train_dataset: ref('train_instances'),
      val_dataset: ref('val_instances'),
    },
    test_metrics: {
      type: 'torch::evaluate',
      model: ref('model'),
      evaluator: evaluator,
      dataset: ref('test_dataset'),
      dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        sampler: {
          type: 'basic',
          batch_size: batch_size,
          shuffle: false,
          drop_last: false,
        },
        collator: ref('datamodule.batch'),
      },
      random_seed: 0,
    },
  },
}
