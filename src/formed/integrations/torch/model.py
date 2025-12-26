"""Base model abstraction for PyTorch models.

This module provides the base class for all PyTorch models in the framework,
integrating torch.nn.Module with the registrable pattern for configuration-based
model instantiation.

Key Features:
    - Integration with PyTorch Module system
    - Registrable pattern for configuration-based instantiation
    - Generic type support for inputs, outputs, and parameters
    - Compatible with TorchTrainer for end-to-end training

Examples:
    >>> from formed.integrations.torch import BaseTorchModel
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> @BaseTorchModel.register("my_model")
    ... class MyModel(BaseTorchModel[dict, torch.Tensor, None]):
    ...     def __init__(self, hidden_dim: int):
    ...         super().__init__()
    ...         self.linear = nn.Linear(10, hidden_dim)
    ...
    ...     def forward(self, inputs: dict, params: None = None) -> torch.Tensor:
    ...         return self.linear(inputs["features"])

"""

from collections.abc import Mapping
from typing import Any, Generic

import torch.nn as nn
from colt import Registrable

from .types import ModelInputT, ModelOutputT, ModelParamsT


class BaseTorchModel(
    nn.Module,
    Registrable,
    Generic[ModelInputT, ModelOutputT, ModelParamsT],
):
    """Base class for all PyTorch models in the framework.

    This class combines PyTorch's nn.Module with the registrable pattern,
    allowing models to be instantiated from configuration files and
    seamlessly integrated with the training infrastructure.

    Type Parameters:
        ModelInputT: Type of input data to the model.
        ModelOutputT: Type of model output.
        ModelParamsT: Type of additional parameters (typically None or a dataclass).

    Note:
        Subclasses should implement forward() to define the forward pass.
        Models are automatically compatible with TorchTrainer when registered.

    """

    __model_config__: Mapping[str, Any] | None = None

    def forward(self, inputs: ModelInputT, params: ModelParamsT | None = None) -> ModelOutputT:
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

    def __call___(self, inputs: ModelInputT, params: ModelParamsT | None = None) -> ModelOutputT:
        """Invoke the model on the given inputs.

        This method wraps the forward method and can include additional
        functionality such as hooks or pre/post-processing.

        Args:
            inputs: Input data to the model.
            params: Optional additional parameters for the forward pass.

        Returns:
            Model output.
        """
        return super().__call__(inputs, params)
