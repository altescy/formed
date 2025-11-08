from typing import Generic, Optional

from colt import Registrable
from flax import nnx

from .types import ModelInputT, ModelOutputT, ModelParamsT


class BaseFlaxModel(
    nnx.Module,
    Registrable,
    Generic[ModelInputT, ModelOutputT, ModelParamsT],
):
    def __call__(self, inputs: ModelInputT, params: Optional[ModelParamsT] = None) -> ModelOutputT:
        raise NotImplementedError()
