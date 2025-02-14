from collections.abc import Callable, Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeVar, Union, cast

import datasets
import minato
import torch
import transformers
from colt import Lazy
from transformers import DataCollator, PreTrainedModel, TrainerCallback, TrainingArguments
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction

from formed.integrations.datasets.workflow import DatasetFormat
from formed.workflow import Format, step, use_step_workdir

PretrainedModelT = TypeVar("PretrainedModelT", bound=PreTrainedModel)


@Format.register("transformers::pretrained_model")
class TransformersPretrainedModelFormat(Generic[PretrainedModelT], Format[PretrainedModelT]):
    def write(self, artifact: PretrainedModelT, directory: Path) -> None:
        artifact.save_pretrained(str(directory / "model"))

    def read(self, directory: Path) -> PretrainedModelT:
        return cast(PretrainedModelT, transformers.AutoModel.from_pretrained(str(directory / "model")))


@step(
    "transformers::tokenize_dataset",
    format=DatasetFormat(),
)
def tokenize_dataset(
    dataset: Union[datasets.Dataset, datasets.DatasetDict],
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    padding: Union[bool, Literal["max_length", "longest", "do_not_pad"]] = False,
    truncation: Union[bool, Literal["only_first", "only_second", "longest_first", "do_not_truncate"]] = False,
    return_special_tokens_mask: bool = False,
    max_length: Optional[int] = None,
) -> Union[datasets.Dataset, datasets.DatasetDict]:
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


@step("transformers::load_pretrained_model", cacheable=False)
def load_pretrained_model(
    model_name_or_path: Union[str, PathLike],
    auto_class: Union[str, type[_BaseAutoModelClass]] = transformers.AutoModel,
    submodule: Optional[str] = None,
    **kwargs: Any,
) -> transformers.PreTrainedModel:
    if isinstance(auto_class, str):
        auto_class = getattr(transformers, auto_class)
    assert isinstance(auto_class, type) and issubclass(auto_class, _BaseAutoModelClass)
    with suppress(FileNotFoundError):
        model_name_or_path = minato.cached_path(model_name_or_path)
    model = auto_class.from_pretrained(model_name_or_path, **kwargs)
    if submodule:
        model = getattr(model, submodule)
    return model


@step("transformers::load_pretrained_tokenizer", cacheable=False)
def load_pretrained_tokenizer(
    pretrained_model_name_or_path: Union[str, PathLike],
    submodule: Optional[str] = None,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    with suppress(FileNotFoundError):
        pretrained_model_name_or_path = minato.cached_path(pretrained_model_name_or_path)
    return transformers.AutoTokenizer.from_pretrained(str(pretrained_model_name_or_path), **kwargs)


@step(
    "transformers::finetune_model",
    format=TransformersPretrainedModelFormat(),
)
def finetune_model(
    model: PreTrainedModel,
    args: Lazy[TrainingArguments],
    data_collator: Optional[DataCollator] = None,
    dataset: Optional[
        Union[
            datasets.Dataset,
            datasets.DatasetDict,
            Mapping[
                str,
                Union[datasets.Dataset, datasets.DatasetDict],
            ],
        ]
    ] = None,
    processing_class: Optional[
        Union[
            PreTrainedTokenizerBase,
            BaseImageProcessor,
            FeatureExtractionMixin,
            ProcessorMixin,
        ]
    ] = None,
    model_init: Optional[Callable[[], PreTrainedModel]] = None,
    compute_loss_func: Optional[Callable] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
    callbacks: Optional[list[TrainerCallback]] = None,
    optimizers: tuple[
        Optional[Lazy[torch.optim.Optimizer]],
        Optional[Lazy[torch.optim.lr_scheduler.LambdaLR]],
    ] = (None, None),
    optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
    preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    train_dataset_key: str = "train",
    eval_dataset_key: str = "validation",
) -> PreTrainedModel:
    workdir = use_step_workdir()

    args_ = args.construct(output_dir=str(workdir))

    train_dataset: Optional[Union[datasets.Dataset, datasets.DatasetDict]] = None
    eval_dataset: Optional[Union[datasets.Dataset, datasets.DatasetDict]] = None
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
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        model_init=model_init,
        compute_loss_func=compute_loss_func,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizer_cls_and_kwargs or (optimizer, lr_scheduler),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    return model
