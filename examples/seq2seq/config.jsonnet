local ref(name) = { type: 'ref', ref: name };
local embeddingDim = 48;
local encoderDim = 128;
local decoderDim = 128;
local evaluator = {
  type: 'seq2seq.evaluators:Seq2SeqEvaluator',
  target_pad_index: ref('datamodule.target.pad_index'),
  metrics: [
    { type: 'loss' },
    { type: 'accuracy', name: 'teacher_forced_token_accuracy' },
    { type: 'exact_match', name: 'teacher_forced_sequence_exact_match' },
  ],
};
local inferenceDataloader = {
  type: 'formed.integrations.ml:DataLoader',
  sampler: { type: 'basic', batch_size: 64, shuffle: false },
  collator: ref('datamodule.batch'),
};
local sequenceSampler = {
  type: 'beam_search',
  adapter: { type: 'default' },
  max_steps: 64,
  beam_size: 4,
  num_return_sequences: 1,
  hypothesis_scorer: { type: 'length_penalty', alpha: 0.6 },
  termination_policy: { type: 'score_bound' },
  stopping_criteria: [
    {
      type: 'end_of_sequence',
      end_index: ref('datamodule.target.eos_index'),
    },
  ],
};

{
  steps: {
    train_dataset: {
      type: 'seq2seq::generate_camel_case_dataset',
      num_examples: 1000,
      random_seed: 42,
    },
    val_dataset: {
      type: 'seq2seq::generate_camel_case_dataset',
      num_examples: 100,
      random_seed: 123,
    },
    test_dataset: {
      type: 'seq2seq::generate_camel_case_dataset',
      num_examples: 100,
      random_seed: 456,
    },
    prediction_dataset: {
      type: 'seq2seq::generate_camel_case_dataset',
      num_examples: 8,
      random_seed: 789,
    },
    datamodule: {
      type: 'ml::train_datamodule',
      datamodule: {
        type: 'seq2seq::datamodule',
        source: {
          analyzer: { type: 'characters' },
          pad_token: '<PAD>',
        },
        target: {
          analyzer: { type: 'characters' },
          pad_token: '<PAD>',
          bos_token: '<BOS>',
          eos_token: '<EOS>',
        },
      },
      dataset: ref('train_dataset'),
    },
    model: {
      type: 'torch::train',
      model: {
        type: 'seq2seq::model',
        source_embedder: {
          type: 'torch.nn:Embedding',
          num_embeddings: ref('datamodule.source.vocab_size'),
          embedding_dim: embeddingDim,
          padding_idx: ref('datamodule.source.pad_index'),
        },
        target_embedder: {
          type: 'torch.nn:Embedding',
          num_embeddings: ref('datamodule.target.vocab_size'),
          embedding_dim: embeddingDim,
          padding_idx: ref('datamodule.target.pad_index'),
        },
        encoder: {
          type: 'lstm',
          input_dim: embeddingDim,
          hidden_dim: encoderDim,
        },
        decoder_state_initializer: {
          type: 'lstm',
          input_dim: encoderDim,
          hidden_dim: decoderDim,
        },
        decoder: {
          type: 'lstm',
          input_dim: embeddingDim,
          hidden_dim: decoderDim,
        },
        output_projection: {
          type: 'torch.nn:Linear',
          in_features: decoderDim,
          out_features: ref('datamodule.target.vocab_size'),
        },
        target_pad_index: ref('datamodule.target.pad_index'),
        target_bos_index: ref('datamodule.target.bos_index'),
      },
      trainer: {
        train_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: { type: 'basic', batch_size: 64, shuffle: true },
          collator: ref('datamodule.batch'),
        },
        val_dataloader: {
          type: 'formed.integrations.ml:DataLoader',
          sampler: { type: 'basic', batch_size: 64, shuffle: false },
          collator: ref('datamodule.batch'),
        },
        engine: {
          type: 'default',
          optimizer: { type: 'torch.optim:Adam', lr: 3e-3 },
          max_grad_norm: 1.0,
        },
        max_epochs: 30,
        logging_strategy: 'epoch',
        callbacks: [
          { type: 'evaluation', evaluator: evaluator },
          { type: 'mlflow' },
        ],
      },
      train_dataset: ref('train_dataset'),
      val_dataset: ref('val_dataset'),
      random_seed: 42,
    },
    test_metrics: {
      type: 'torch::evaluate',
      model: ref('model'),
      evaluator: evaluator,
      dataset: ref('test_dataset'),
      dataloader: {
        type: 'formed.integrations.ml:DataLoader',
        sampler: { type: 'basic', batch_size: 64, shuffle: false },
        collator: ref('datamodule.batch'),
      },
      random_seed: 42,
    },
    predictions: {
      type: 'seq2seq::predict',
      model: ref('model'),
      datamodule: ref('datamodule'),
      dataset: ref('prediction_dataset'),
      dataloader: inferenceDataloader,
      sampler: sequenceSampler,
    },
    test_predictions: {
      type: 'seq2seq::predict',
      model: ref('model'),
      datamodule: ref('datamodule'),
      dataset: ref('test_dataset'),
      dataloader: inferenceDataloader,
      sampler: sequenceSampler,
      print_results: false,
    },
    generation_metrics: {
      type: 'seq2seq::evaluate_generation',
      predictions: ref('test_predictions'),
      metrics: [
        { type: 'accuracy' },
      ],
    },
  },
}
