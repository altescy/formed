# Causal Language Modeling with 🤗 Transformers

This tutorial shows you how to fine-tune a pre-trained language model using formed's Transformers integration. Unlike custom model development, this tutorial focuses on using formed's built-in workflow steps to orchestrate training with Hugging Face Transformers.

**What you'll build**: A causal language model fine-tuned on question-answer data, using DistilGPT-2 as the base model.

**What you'll learn**:

- Loading datasets with the `datasets` integration
- Tokenizing text data for language modeling
- Fine-tuning transformer models with `transformers::train_model`
- Configuring training arguments and data collators
- Tracking experiments with MLflow integration

## Prerequisites

Install formed with required integrations:

```bash
pip install formed[transformers,datasets,mlflow]
```

**Note**: The transformers integration provides seamless access to Hugging Face's ecosystem, including pre-trained models, tokenizers, and training utilities.

## What is Causal Language Modeling?

Causal language modeling (CLM) trains models to predict the next token given previous tokens. This is the training objective used by GPT-style models.

**Key characteristics**:

- Models see only left context (previous tokens)
- Commonly used for text generation tasks
- Training uses the language modeling head with cross-entropy loss

## Project Setup

Create a new directory:

```bash
mkdir causallm_tutorial
cd causallm_tutorial
```

We'll create two files:

- `config.jsonnet` - Workflow configuration
- `formed.yml` - Project settings

## Step 1: Configure Project Settings

Create `formed.yml`:

```yaml
workflow:
  organizer:
    type: mlflow
    log_execution_metrics: true

required_modules:
  - formed.integrations.mlflow
  - formed.integrations.datasets
  - formed.integrations.transformers
```

**What this does**:

- **MLflow organizer**: Automatically tracks all experiments, metrics, and model artifacts
- **Required modules**: Imports integrations that provide workflow steps
  - `datasets`: Load data from Hugging Face Hub or local files
  - `transformers`: Tokenization and model training
  - `mlflow`: Experiment tracking

## Step 2: Define the Workflow

Create `config.jsonnet`:

```jsonnet
// Define model and tokenizer configurations at the top for reusability
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
    // Step 1: Load dataset from Hugging Face Hub
    train_dataset: {
      type: 'datasets::load',
      path: 'sentence-transformers/eli5',
      split: 'train[:10000]',
    },

    // Step 2: Tokenize text data
    tokenized_dataset: {
      type: 'transformers::tokenize',
      dataset: { type: 'ref', ref: 'train_dataset' },
      tokenizer: tokenizer,
      text_column: 'answer',
    },

    // Step 3: Train the model
    trained_model: {
      type: 'transformers::train_model',
      model: base_model,
      dataset: { type: 'ref', ref: 'tokenized_dataset' },

      // Training arguments (passed to transformers.TrainingArguments)
      args: {
        per_device_train_batch_size: 8,
        per_device_eval_batch_size: 8,
        learning_rate: 2e-5,
        warmup_ratio: 0.1,
        num_train_epochs: 3,
        fp16: false,
        bf16: true,  // Use bfloat16 for training (requires compatible hardware)
        report_to: 'none',  // Don't report to external trackers (we use MLflow)
        do_train: true,
        do_eval: false,  // No validation set in this example
        save_strategy: 'steps',
        save_steps: 100,
        save_total_limit: 2,
        eval_strategy: 'no',
        logging_strategy: 'steps',
        logging_first_step: true,
        logging_steps: 10,
      },

      // Data collator for language modeling
      data_collator: {
        type: 'transformers:DataCollatorForLanguageModeling',
        tokenizer: tokenizer,
        mlm: false,  // Use causal LM (not masked LM)
      },

      // Processing class for tokenization during training
      processing_class: tokenizer,

      // Callbacks for experiment tracking
      callbacks: [
        {
          type: 'formed.integrations.transformers.training:MlflowTrainerCallback',
        },
      ],
    },
  },
}
```

Let's break down each component:

### Step 1: Loading Data with `datasets::load`

```jsonnet
train_dataset: {
  type: 'datasets::load',
  path: 'sentence-transformers/eli5',
  split: 'train[:10000]',
}
```

**What happens**:

- Loads the ELI5 (Explain Like I'm 5) dataset from Hugging Face Hub
- Takes first 10,000 examples from the training split
- Returns a `datasets.Dataset` object

**Key parameters**:

- `path`: Dataset name on Hugging Face Hub or path to local dataset
- `split`: Which split to load (supports slice notation like `train[:1000]`)
- Additional kwargs are passed to `datasets.load_dataset()`

**Dataset format**: The ELI5 dataset contains question-answer pairs. We'll use the `answer` field for language modeling.

### Step 2: Tokenizing with `transformers::tokenize`

```jsonnet
tokenized_dataset: {
  type: 'transformers::tokenize',
  dataset: { type: 'ref', ref: 'train_dataset' },
  tokenizer: tokenizer,
  text_column: 'answer',
}
```

**What happens**:

- Applies tokenization to the specified text column
- Converts text to token IDs compatible with the model
- Removes the original text column (keeps only token IDs)
- Returns a tokenized `datasets.Dataset`

**Key parameters**:

- `dataset`: Input dataset (reference to previous step)
- `tokenizer`: Tokenizer configuration or pre-loaded tokenizer
- `text_column`: Name of the column containing text to tokenize
- `padding`: Padding strategy (default: False, padding handled by data collator)
- `truncation`: Whether to truncate sequences
- `max_length`: Maximum sequence length

**Tokenizer configuration**:

```jsonnet
local tokenizer = {
  type: 'transformers:AutoTokenizer.from_pretrained',
  pretrained_model_name_or_path: 'distilbert/distilgpt2',
  pad_token: '<|endoftext|>',
};
```

This loads DistilGPT-2's tokenizer and sets the padding token (GPT-2 doesn't have one by default).

### Step 3: Training with `transformers::train_model`

```jsonnet
trained_model: {
  type: 'transformers::train_model',
  model: base_model,
  dataset: { type: 'ref', ref: 'tokenized_dataset' },
  args: { ... },
  data_collator: { ... },
  processing_class: tokenizer,
  callbacks: [ ... ],
}
```

**What happens**:

- Initializes the model from the pre-trained checkpoint
- Creates a `transformers.Trainer` with specified arguments
- Trains the model on the tokenized dataset
- Saves checkpoints according to `save_strategy`
- Returns the trained model

**Key parameters**:

#### Model Configuration

```jsonnet
model: {
  type: 'transformers:AutoModelForCausalLM.from_pretrained',
  pretrained_model_name_or_path: 'distilbert/distilgpt2',
}
```

Uses `AutoModelForCausalLM` to load a model with a causal language modeling head.

#### Training Arguments

The `args` field accepts any parameters from `transformers.TrainingArguments`:

**Batch size and epochs**:

- `per_device_train_batch_size`: Batch size per GPU/CPU
- `num_train_epochs`: Number of training epochs

**Optimization**:

- `learning_rate`: Learning rate for optimizer (default: 5e-5)
- `warmup_ratio`: Fraction of steps for learning rate warmup

**Mixed precision**:

- `fp16`: Use float16 (older GPUs)
- `bf16`: Use bfloat16 (newer GPUs, more stable)

**Checkpointing**:

- `save_strategy`: When to save ("steps", "epoch", "no")
- `save_steps`: Save checkpoint every N steps
- `save_total_limit`: Keep only N most recent checkpoints

**Logging**:

- `logging_strategy`: When to log ("steps", "epoch")
- `logging_steps`: Log every N steps
- `logging_first_step`: Whether to log after first step

#### Data Collator

```jsonnet
data_collator: {
  type: 'transformers:DataCollatorForLanguageModeling',
  tokenizer: tokenizer,
  mlm: false,
}
```

The data collator handles batching and prepares labels:

**`DataCollatorForLanguageModeling`**:

- `mlm: false`: Causal language modeling (predict next token)
- `mlm: true`: Masked language modeling (BERT-style, predict masked tokens)

For causal LM:

- Creates labels by shifting `input_ids` one position to the right
- Applies padding to create uniform batch size
- Handles attention masks automatically

#### Callbacks

```jsonnet
callbacks: [
  {
    type: 'formed.integrations.transformers.training:MlflowTrainerCallback',
  },
]
```

**`MlflowTrainerCallback`**:

- Logs training metrics to MLflow automatically
- Tracks loss, learning rate, and other training stats
- Integrates seamlessly with formed's MLflow organizer

## Step 3: Run the Workflow

Execute the workflow:

```bash
formed workflow run config.jsonnet --execution-id causallm-distilgpt2
```

**What happens during execution**:

1. **Dataset loading**: Downloads and caches the ELI5 dataset
2. **Tokenization**: Tokenizes all examples and caches results
3. **Training**: Runs training loop with Hugging Face Trainer
   - Logs metrics every 10 steps
   - Saves checkpoints every 100 steps
   - Uses bfloat16 mixed precision
4. **Model saving**: Caches the trained model by fingerprint

## Step 4: View Training Results

Launch MLflow UI:

```bash
mlflow ui
```

Open http://localhost:5000 to see:

**Metrics**:

- Training loss curve
- Learning rate schedule
- Steps per second

**Parameters**:

- All training arguments
- Model architecture
- Dataset configuration

**Artifacts**:

- Trained model checkpoints
- Tokenizer files
- Training logs

## Understanding the Components

### The `datasets` Integration

The `datasets` integration provides workflow steps for working with Hugging Face datasets:

**Available steps**:

- `datasets::load` - Load datasets from Hub or local files
- `datasets::compose` - Combine multiple datasets into DatasetDict
- `datasets::concatenate` - Concatenate datasets
- `datasets::train_test_split` - Split dataset into train/test

**Benefits**:

- Automatic caching of downloaded datasets
- Memory-efficient processing with Apache Arrow
- Seamless integration with transformers

### The `transformers` Integration

The `transformers` integration wraps Hugging Face Transformers for workflow use:

**Key steps**:

- `transformers::tokenize` - Tokenize text data
- `transformers::train_model` - Train models with Trainer API
- `transformers::load_model` - Load pre-trained models
- `transformers::load_tokenizer` - Load tokenizers
- `transformers::convert_tokenizer` - Convert to formed's Tokenizer format

**Benefits**:

- Access to thousands of pre-trained models
- Battle-tested training infrastructure
- Automatic gradient accumulation, mixed precision, and distributed training

### Data Collators

Data collators prepare batches during training:

**`DataCollatorForLanguageModeling`**:

- Handles causal and masked language modeling
- Creates labels automatically from inputs
- Applies dynamic padding for efficiency

**Other common collators**:

- `DataCollatorWithPadding` - Simple padding without label generation
- `DataCollatorForSeq2Seq` - For encoder-decoder models
- `DataCollatorForTokenClassification` - For NER and similar tasks

## Customization Examples

### Use a Different Model

Replace DistilGPT-2 with another model:

```jsonnet
local base_model = {
  type: 'transformers:AutoModelForCausalLM.from_pretrained',
  pretrained_model_name_or_path: 'gpt2',  // or 'gpt2-medium', 'EleutherAI/gpt-neo-125M', etc.
};
```

### Add Validation Set

Split the dataset and enable evaluation:

```jsonnet
{
  steps: {
    raw_dataset: {
      type: 'datasets::load',
      path: 'sentence-transformers/eli5',
      split: 'train[:10000]',
    },

    // Split into train and validation
    split_dataset: {
      type: 'datasets::train_test_split',
      dataset: { type: 'ref', ref: 'raw_dataset' },
      test_size: 0.1,
      seed: 42,
    },

    // Tokenize both splits
    tokenized_train: {
      type: 'transformers::tokenize',
      dataset: { type: 'ref', ref: 'split_dataset.train' },
      tokenizer: tokenizer,
      text_column: 'answer',
    },

    tokenized_val: {
      type: 'transformers::tokenize',
      dataset: { type: 'ref', ref: 'split_dataset.test' },
      tokenizer: tokenizer,
      text_column: 'answer',
    },

    // Combine for training
    dataset: {
      type: 'datasets::compose',
      train: { type: 'ref', ref: 'tokenized_train' },
      validation: { type: 'ref', ref: 'tokenized_val' },
    },

    trained_model: {
      type: 'transformers::train_model',
      // ...
      dataset: { type: 'ref', ref: 'dataset' },
      args: {
        // ...
        do_eval: true,
        eval_strategy: 'steps',
        eval_steps: 100,
      },
    },
  },
}
```

### Adjust Training Settings

**Longer training with more frequent evaluation**:

```jsonnet
args: {
  num_train_epochs: 5,
  eval_strategy: 'steps',
  eval_steps: 50,
  logging_steps: 5,
}
```

**Larger batch size with gradient accumulation**:

```jsonnet
args: {
  per_device_train_batch_size: 4,
  gradient_accumulation_steps: 4,  // Effective batch size: 16
  learning_rate: 1e-5,
}
```

**Different optimizer**:

```jsonnet
trained_model: {
  type: 'transformers::train_model',
  // ...
  args: {
    // ...
    optim: 'adamw_torch',  // or 'adafactor', 'adamw_8bit', etc.
    weight_decay: 0.01,
  },
}
```

### Custom Learning Rate Schedule

```jsonnet
args: {
  learning_rate: 5e-5,
  lr_scheduler_type: 'cosine',
  warmup_steps: 500,
}
```

### Truncate Long Sequences

```jsonnet
tokenized_dataset: {
  type: 'transformers::tokenize',
  dataset: { type: 'ref', ref: 'train_dataset' },
  tokenizer: tokenizer,
  text_column: 'answer',
  truncation: true,
  max_length: 512,
}
```

## Using the Trained Model

After training, you can load the cached model for inference:

```python
from formed.settings import load_formed_settings
from formed.workflow import WorkflowExecutionID

# Load the workflow execution
settings = load_formed_settings("./formed.yml")
organizer = settings.workflow.organizer

context = organizer.get(WorkflowExecutionID("your-execution-id"))

# Get the trained model from cache
model_step_id = context.info.graph["trained_model"]
model = context.cache[model_step_id]

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

# Generate text
inputs = tokenizer("The meaning of life is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

Or reference the model in a new workflow step:

```jsonnet
{
  steps: {
    // ... training steps ...

    // Generate text using trained model
    generated_text: {
      type: 'my_custom::generate',
      model: { type: 'ref', ref: 'trained_model' },
      prompts: ['The meaning of life is', 'Once upon a time'],
      max_length: 100,
    },
  },
}
```

## Key Takeaways

**Workflow Steps**:

- `datasets::load` loads datasets from Hugging Face Hub
- `transformers::tokenize` prepares text for model input
- `transformers::train_model` orchestrates training with Trainer API
- All steps are cached by fingerprint for reproducibility

**Training Configuration**:

- TrainingArguments control all aspects of training
- Data collators handle batch preparation and label creation
- Callbacks enable custom logging and monitoring

**MLflow Integration**:

- Automatic experiment tracking
- Metrics, parameters, and artifacts logged transparently
- Easy comparison across training runs

**Workflow Benefits**:

- No custom Python code needed for standard tasks
- Configuration-driven experimentation
- Automatic caching and dependency management
- Seamless integration with Hugging Face ecosystem

## Next Steps

### Fine-tune on Custom Data

Replace the dataset with your own:

```jsonnet
train_dataset: {
  type: 'datasets::load',
  path: '/path/to/your/dataset.jsonl',
}
```

Your data should be in a format supported by Hugging Face datasets (JSON, CSV, Parquet, etc.).

### Try Masked Language Modeling

For BERT-style models:

```jsonnet
local base_model = {
  type: 'transformers:AutoModelForMaskedLM.from_pretrained',
  pretrained_model_name_or_path: 'bert-base-uncased',
};

// ...

data_collator: {
  type: 'transformers:DataCollatorForLanguageModeling',
  tokenizer: tokenizer,
  mlm: true,
  mlm_probability: 0.15,
}
```

## Further Reading

- **[Text Classification Tutorial](./text_classification.md)**: Build custom models with PyTorch
- **[Transformers Documentation](https://huggingface.co/docs/transformers)**: Complete Hugging Face Transformers guide
- **[Workflow Guide](../guides/workflow.md)**: Advanced workflow patterns
- **[MLflow Integration](../reference/integrations/mlflow.md)**: Experiment tracking details

For more examples, see `examples/causallm/` in the repository.
