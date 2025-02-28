from typing import TYPE_CHECKING, Generic, Optional, Sequence

import colt
from flax import nnx

from formed.integrations.ml import DataModule, extract_fields

from .types import ModelInputT, ModelOutputT, ModelParamsT

if TYPE_CHECKING:
    from .training import FlaxTrainingModule, TrainerCallback


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

    @classmethod
    def default_data_module(cls) -> Optional[DataModule]:
        fields = extract_fields(cls.Input)
        if fields is not None:
            return DataModule(fields)
        return None

    @classmethod
    def default_training_module(cls) -> Optional["FlaxTrainingModule"]:
        return None

    def trainer_callbacks(self) -> Sequence["TrainerCallback"]:
        return []
