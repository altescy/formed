# Sequence-to-sequence workflow example

This example trains a character-level Transformer encoder-decoder to convert
`camelCase` identifiers into `snake_case`. All datasets are generated locally.

From this directory, run:

```bash
uv run formed workflow run config.jsonnet --execution-id camel-to-snake
```

`config.jsonnet` wires a Transformer encoder, the matching Transformer state
initializer, and a cross-attending Transformer decoder with a reorderable
key/value cache. Because `Seq2SeqModel` is decoder-agnostic, switching to an
LSTM stack (or any other) is purely a matter of swapping the `encoder`,
`decoder_state_initializer`, and `decoder` blocks in the config -- the model,
steps, and workflow are unchanged.

The workflow generates training, validation, and test datasets, trains the
character vocabularies in a DataModule, and trains the model with `torch::train`.
An evaluation callback reports validation loss and explicitly named
teacher-forced metrics to MLflow after each epoch. The evaluator receives its
task metrics through configuration, following the same DI pattern as the text
classification example. A `torch::evaluate` step records those metrics on the
test set, while a separate generation-evaluation step records autoregressive
exact match. Prediction uses an injected DataLoader and beam-search sampler, so
it does not materialize the entire dataset as one batch. The sampler keeps four
hypotheses per input, ranks them with an injected length-penalty scorer, and
returns the highest-scoring sequence. A score-bound termination policy stops
an input's search once no unfinished hypothesis can enter the requested n-best.

The source and target vocabularies are owned by the DataModule. Their sizes and
special-token indices are passed to torch components through Jsonnet `ref`
values, without introducing a dependency between the ML and torch integrations.
Both fields use `TextIndexer` with an injected character analyzer, so raw
examples and reconstructed predictions remain strings; character tokenization
and detokenization are entirely owned by the DataModule.

The model is only responsible for wiring together injected components. Source
and target embedders, the sequence encoder, decoder-state initializer, decoder,
and output projection are all selected in `config.jsonnet`. The initializer
converts encoded source vectors into the decoder's opaque state; that state may
also contain conditioning context such as encoder memory in other decoder
implementations.
