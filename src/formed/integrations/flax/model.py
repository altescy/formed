"""Base model abstraction for Flax NNX models.

This module provides the base class for all Flax models in the framework,
integrating Flax NNX with the registrable pattern for configuration-based
model instantiation.

Key Features:
    - Integration with Flax NNX Module system
    - Registrable pattern for configuration-based instantiation
    - Generic type support for inputs, outputs, and parameters
    - Compatible with FlaxTrainer for end-to-end training

Examples:
    >>> from formed.integrations.flax import BaseFlaxModel
    >>> from flax import nnx
    >>> import jax
    >>>
    >>> @BaseFlaxModel.register("my_model")
    ... class MyModel(BaseFlaxModel[dict, jax.Array, None]):
    ...     def __init__(self, rngs: nnx.Rngs, hidden_dim: int):
    ...         self.linear = nnx.Linear(10, hidden_dim, rngs=rngs)
    ...
    ...     def __call__(self, inputs: dict, params: None = None) -> jax.Array:
    ...         return self.linear(inputs["features"])

"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic

from colt import Registrable
from flax import nnx

from .types import ModelInputT, ModelOutputT, ModelParamsT


class BaseFlaxModel(
    nnx.Module,
    Registrable,
    Generic[ModelInputT, ModelOutputT, ModelParamsT],
):
    """Base class for all Flax NNX models in the framework.

    This class combines Flax's NNX Module with the registrable pattern,
    allowing models to be instantiated from configuration files and
    seamlessly integrated with the training infrastructure.

    Type Parameters:
        ModelInputT: Type of input data to the model.
        ModelOutputT: Type of model output.
        ModelParamsT: Type of additional parameters (typically None or a dataclass).

    Note:
        Subclasses should implement __call__ to define the forward pass.
        Models are automatically compatible with FlaxTrainer when registered.

    """

    if TYPE_CHECKING:
        __model_config__: Mapping[str, Any]

    def __call__(self, inputs: ModelInputT, params: ModelParamsT | None = None) -> ModelOutputT:
        """Forward pass of the model.

        Args:
            inputs: Input data to the model.
            params: Optional additional parameters for the forward pass.

        Returns:
            Model output.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError()
