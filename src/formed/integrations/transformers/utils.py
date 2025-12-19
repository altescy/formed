import importlib
from contextlib import suppress
from functools import lru_cache
from os import PathLike

import minato
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from formed.common.attributeutils import xgetattr


@lru_cache(maxsize=8)
def load_pretrained_transformer(
    model_name_or_path: str | PathLike,
    auto_class: str | type[_BaseAutoModelClass] | None = None,
    submodule: str | None = None,
    **kwargs,
) -> PreTrainedModel:
    if auto_class is None:
        auto_class = AutoModel
    elif isinstance(auto_class, str):
        assert ":" in auto_class, "auto_class string must be in 'module:ClassName' format"
        module_name, class_name = auto_class.rsplit(":", 1)
        module = importlib.import_module(module_name)
        auto_class = getattr(module, class_name)

    assert isinstance(auto_class, type) and issubclass(auto_class, _BaseAutoModelClass), (
        "auto_class must be a subclass of transformers._BaseAutoModelClass"
    )

    with suppress(FileNotFoundError):
        model_name_or_path = minato.cached_path(model_name_or_path)
    model = auto_class.from_pretrained(str(model_name_or_path), **kwargs)
    if submodule:
        model = xgetattr(model, submodule)
    return model


@lru_cache(maxsize=8)
def load_pretrained_tokenizer(
    model_name_or_path: str | PathLike,
    **kwargs,
) -> PreTrainedTokenizerBase:
    with suppress(FileNotFoundError):
        model_name_or_path = minato.cached_path(model_name_or_path)
    return AutoTokenizer.from_pretrained(str(model_name_or_path), **kwargs)
