from typing import Any, Dict, Generic, Optional, Set, Tuple, TypeVar

import colt
import flax
import jax
from flax import nnx

from .types import ModelInputT, ModelOutputT, ModelParamsT


class FlaxModel(
    nnx.Module,
    colt.Registrable,
    Generic[ModelInputT, ModelOutputT, ModelParamsT],
):
    Input: type[ModelInputT]
    Output: type[ModelOutputT]
    Params: type[ModelParamsT]

    def __call__(
        self,
        inputs: ModelInputT,
        params: Optional[ModelParamsT] = None,
        *,
        train: bool = False,
    ) -> ModelOutputT:
        raise NotImplementedError
