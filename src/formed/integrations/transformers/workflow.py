import importlib
import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

import datasets
import torch
import transformers
from colt import Lazy
from transformers import PreTrainedModel, TrainerCallback, TrainingArguments
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction

from formed.integrations.datasets.workflow import DatasetFormat
from formed.types import NotSpecified
from formed.workflow import Format, step, use_step_workdir

from .utils import load_pretrained_tokenizer, load_pretrained_transformer

DataCollator: TypeAlias = Callable  # NOTE: workaround for type mismatch in transformers
PretrainedModelT = TypeVar("PretrainedModelT", bound=PreTrainedModel)


@Format.register("transformers::model")
class TransformersPretrainedModelFormat(Generic[PretrainedModelT], Format[PretrainedModelT]):
    def write(self, artifact: PretrainedModelT, directory: Path) -> None:
        artifact.save_pretrained(str(directory / "model"))
        metadata = {
            "module": artifact.__class__.__module__,
            "class": artifact.__class__.__name__,
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False))

    def read(self, directory: Path) -> PretrainedModelT:
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        module_name = metadata["module"]
        class_name = metadata["class"]
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        if not issubclass(model_class, PreTrainedModel):
            raise ValueError(f"Class {class_name} is not a subclass of PreTrainedModel")
        model = model_class.from_pretrained(str(directory / "model"))
        return cast(PretrainedModelT, model)


@step("transformers::tokenize", format=DatasetFormat())
def tokenize_dataset(
    dataset: datasets.Dataset | datasets.DatasetDict,
    tokenizer: str | PathLike | PreTrainedTokenizerBase,
    text_column: str = "text",
    padding: bool | Literal["max_length", "longest", "do_not_pad"] = False,
    truncation: bool | Literal["only_first", "only_second", "longest_first", "do_not_truncate"] = False,
    return_special_tokens_mask: bool = False,
    max_length: int | None = None,
) -> datasets.Dataset | datasets.DatasetDict:
    """Tokenize a dataset using a pre-trained tokenizer.

    This step applies tokenization to a text column in the dataset,
    removing the original text column and adding tokenized features.

    Args:
        dataset: Dataset or DatasetDict to tokenize.
        tokenizer: Tokenizer identifier, path, or instance.
        text_column: Name of the text column to tokenize.
        padding: Padding strategy.
        truncation: Truncation strategy.
        return_special_tokens_mask: Whether to return special tokens mask.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset with the text column removed.
    """
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        tokenizer = load_pretrained_tokenizer(tokenizer)

    def tokenize_function(examples: Mapping[str, Any]) -> Any:
        return tokenizer(
            examples[text_column],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column],
    )


@step("transformers::load_model", cacheable=False)
def load_pretrained_model(
    model_name_or_path: str | PathLike,
    auto_class: str | type[_BaseAutoModelClass] = transformers.AutoModel,
    submodule: str | None = None,
    **kwargs: Any,
) -> transformers.PreTrainedModel:
    """Load a pre-trained transformer model.

    Args:
        model_name_or_path: Model identifier or path to model directory.
        auto_class: Auto model class to use for loading (name or class).
        submodule: Optional submodule to extract from the model.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        Loaded pre-trained transformer model.
    """
    if isinstance(auto_class, str):
        auto_class = getattr(transformers, auto_class)
    assert isinstance(auto_class, type) and issubclass(auto_class, _BaseAutoModelClass)
    return load_pretrained_transformer.__wrapped__(
        model_name_or_path=model_name_or_path,
        auto_class=auto_class,
        submodule=submodule,
        **kwargs,
    )


@step("transformers::load_tokenizer", cacheable=False)
def load_pretrained_tokenizer_step(
    pretrained_model_name_or_path: str | PathLike,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """Load a pre-trained tokenizer.

    Args:
        pretrained_model_name_or_path: Model identifier or path to model directory.
        **kwargs: Additional arguments to pass to the tokenizer constructor.

    Returns:
        Loaded pre-trained tokenizer.
    """
    return load_pretrained_tokenizer(pretrained_model_name_or_path, **kwargs)


@step("transformers::train_model", format=TransformersPretrainedModelFormat())
def train_transformer_model(
    model: PreTrainedModel,
    args: Lazy[TrainingArguments],
    data_collator: DataCollator | None = None,  # pyright: ignore[reportInvalidTypeForm]
    dataset: None
    | (datasets.Dataset | datasets.DatasetDict | Mapping[str, datasets.Dataset | datasets.DatasetDict]) = None,
    processing_class: None
    | (PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin) = None,
    model_init: Callable[[], PreTrainedModel] | None = None,
    compute_loss_func: Callable | None = None,
    compute_metrics: Callable[[EvalPrediction], dict] | None = None,
    callbacks: list[TrainerCallback] | None = None,
    optimizers: tuple[
        Lazy[torch.optim.Optimizer] | None,
        Lazy[torch.optim.lr_scheduler.LambdaLR] | None,
    ] = (None, None),
    optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
    preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    train_dataset_key: str = "train",
    eval_dataset_key: str = "validation",
) -> PreTrainedModel:
    """Train a transformer model using the Hugging Face Trainer.

    This step trains a transformer model on the provided datasets using
    the Hugging Face Trainer API.

    Args:
        model: Pre-trained model to train.
        args: Training arguments configuration.
        data_collator: Optional data collator for batching.
        dataset: Training/validation datasets.
        processing_class: Optional processing class (tokenizer, processor, etc.).
        model_init: Optional model initialization function.
        compute_loss_func: Optional custom loss computation function.
        compute_metrics: Optional metrics computation function.
        callbacks: Optional training callbacks.
        optimizers: Optional optimizer and learning rate scheduler.
        optimizer_cls_and_kwargs: Optional optimizer class and keyword arguments.
        preprocess_logits_for_metrics: Optional logits preprocessing function.
        train_dataset_key: Key for training dataset split.
        eval_dataset_key: Key for evaluation dataset split.

    Returns:
        Trained transformer model.
    """
    workdir = use_step_workdir()

    args_ = args.construct(output_dir=str(workdir))

    train_dataset: datasets.Dataset | datasets.DatasetDict | None = None
    eval_dataset: datasets.Dataset | datasets.DatasetDict | None = None
    if isinstance(dataset, datasets.Dataset):
        train_dataset = dataset
        eval_dataset = None
    else:
        train_dataset = dataset.get(train_dataset_key) if dataset and args_.do_train else None
        eval_dataset = dataset.get(eval_dataset_key) if dataset and args_.do_eval else None

    lazy_optimizer, lazy_lr_scheduler = optimizers
    optimizer = lazy_optimizer.construct(params=model.parameters()) if lazy_optimizer else None
    lr_scheduler = lazy_lr_scheduler.construct(optimizer=optimizer) if lazy_lr_scheduler else None

    trainer = transformers.Trainer(
        model=model,
        args=args_,
        data_collator=data_collator,  # pyright: ignore[reportArgumentType]
        train_dataset=train_dataset,  # pyright: ignore[reportArgumentType]
        eval_dataset=eval_dataset,  # pyright: ignore[reportArgumentType]
        processing_class=processing_class,
        model_init=model_init,
        compute_loss_func=compute_loss_func,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizer_cls_and_kwargs or (optimizer, lr_scheduler),  # type: ignore[reportArgumentType]
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    return model


with suppress(ImportError):
    from formed.integrations.ml import Param, Tokenizer, TokenSequenceIndexer

    from .analyzers import PretrainedTransformerAnalyzer

    @step("transformers::convert_tokenizer", format="json")
    def convert_tokenizer(
        tokenizer: str | PathLike | PreTrainedTokenizerBase,
        pad_token: str | None | NotSpecified = NotSpecified.VALUE,
        unk_token: str | None | NotSpecified = NotSpecified.VALUE,
        bos_token: str | None | NotSpecified = NotSpecified.VALUE,
        eos_token: str | None | NotSpecified = NotSpecified.VALUE,
        freeze: bool = True,
        accessor: str | Callable | None = None,
    ) -> Tokenizer:
        """Convert a transformer tokenizer to a formed Tokenizer.

        This step converts a Hugging Face tokenizer into a formed Tokenizer
        with specified special tokens.

        Args:
            tokenizer: Tokenizer identifier, path, or instance.
            pad_token: Padding token (uses tokenizer default if not specified).
            unk_token: Unknown token (uses tokenizer default if not specified).
            bos_token: Beginning-of-sequence token (uses tokenizer default if not specified).
            eos_token: End-of-sequence token (uses tokenizer default if not specified).
            freeze: Whether to freeze the vocabulary.
            accessor: Optional accessor for token extraction.

        Returns:
            Converted formed Tokenizer.

        Raises:
            AssertionError: If pad_token is not specified and not available in the tokenizer.
        """
        given_tokenizer = tokenizer

        if isinstance(tokenizer, (str, PathLike)):
            tokenizer = load_pretrained_tokenizer(tokenizer)

        def get_token(given: str | None | NotSpecified, default: Any) -> str | None:
            if not isinstance(given, NotSpecified):
                return given
            if isinstance(default, str):
                return default
            return None

        vocab = tokenizer.get_vocab().copy()
        pad_token = get_token(pad_token, tokenizer.pad_token)
        unk_token = get_token(unk_token, tokenizer.unk_token)
        bos_token = get_token(bos_token, tokenizer.bos_token)
        eos_token = get_token(eos_token, tokenizer.eos_token)

        assert isinstance(pad_token, str), "pad_token must be specified or available in the tokenizer"

        surface_indexer = TokenSequenceIndexer(
            vocab=vocab,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            freeze=freeze,
        )
        analyzer = PretrainedTransformerAnalyzer(given_tokenizer)
        return Tokenizer(
            surfaces=surface_indexer,
            analyzer=Param.cast(analyzer),
            accessor=accessor,
        )
