local base_model = {
  type: 'transformers:AutoModelForCausalLM.from_pretrained',
  pretrained_model_name_or_path: 'distilbert/distilgpt2',
};
local tokenizer = {
  type: 'transformers:AutoTokenizer.from_pretrained',
  pretrained_model_name_or_path: base_model.pretrained_model_name_or_path,
  pad_token: '<|endoftext|>',
};

{
  steps: {
    train_dataset: {
      type: 'datasets::load',
      path: 'sentence-transformers/eli5',
      split: 'train[:10000]',
    },
    tokenized_dataset: {
      type: 'transformers::tokenize',
      dataset: { type: 'ref', ref: 'train_dataset' },
      tokenizer: tokenizer,
      text_column: 'answer',
    },
    trained_model: {
      type: 'transformers::train_model',
      model: base_model,
      dataset: { type: 'ref', ref: 'tokenized_dataset' },
      args: {
        per_device_train_batch_size: 32,
        per_device_eval_batch_size: 32,
        learning_rate: 2e-5,
        warmup_ratio: 0.1,
        num_train_epochs: 3,
        fp16: false,
        bf16: true,
        report_to: 'none',
        do_train: true,
        do_eval: true,
        save_strategy: 'steps',
        save_steps: 100,
        save_total_limit: 2,
        eval_strategy: 'no',
        logging_strategy: 'steps',
        logging_first_step: true,
        logging_steps: 10,
      },
      data_collator: {
        type: 'transformers:DataCollatorForLanguageModeling',
        tokenizer: tokenizer,
        mlm: false,
      },
      processing_class: tokenizer,
      callbacks: [
        {
          type: 'formed.integrations.transformers.training:MlflowTrainerCallback',
        },
      ],

    },
  },
}
