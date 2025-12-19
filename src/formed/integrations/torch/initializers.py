from collections.abc import Sequence

import torch
from colt import Registrable


class BaseTensorInitializer(Registrable):
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError


@BaseTensorInitializer.register("uniform")
class UniformTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int], low: float = 0.0, high: float = 1.0):
        self._shape = shape
        self._low = low
        self._high = high

    def __call__(self) -> torch.Tensor:
        return torch.empty(self._shape).uniform_(self._low, self._high)


@BaseTensorInitializer.register("normal")
class NormalTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int], mean: float = 0.0, std: float = 1.0):
        self._shape = shape
        self._mean = mean
        self._std = std

    def __call__(self) -> torch.Tensor:
        return torch.empty(self._shape).normal_(self._mean, self._std)


@BaseTensorInitializer.register("xavier_uniform")
class XavierUniformTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int], gain: float = 1.0):
        self._shape = shape
        self._gain = gain

    def __call__(self) -> torch.Tensor:
        tensor = torch.empty(self._shape)
        torch.nn.init.xavier_uniform_(tensor, gain=self._gain)
        return tensor


@BaseTensorInitializer.register("xavier_normal")
class XavierNormalTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int], gain: float = 1.0):
        self._shape = shape
        self._gain = gain

    def __call__(self) -> torch.Tensor:
        tensor = torch.empty(self._shape)
        torch.nn.init.xavier_normal_(tensor, gain=self._gain)
        return tensor


@BaseTensorInitializer.register("kaiming_uniform")
class KaimingUniformTensorInitializer(BaseTensorInitializer):
    def __init__(
        self,
        shape: Sequence[int],
        a: float = 0,
        mode: torch.nn.init._FanMode = "fan_in",
        nonlinearity: torch.nn.init._NonlinearityType = "leaky_relu",
    ):
        self._shape = shape
        self._a = a
        self._mode: torch.nn.init._FanMode = mode
        self._nonlinearity: torch.nn.init._NonlinearityType = nonlinearity

    def __call__(self) -> torch.Tensor:
        tensor = torch.empty(self._shape)
        torch.nn.init.kaiming_uniform_(tensor, a=self._a, mode=self._mode, nonlinearity=self._nonlinearity)
        return tensor


@BaseTensorInitializer.register("kaiming_normal")
class KaimingNormalTensorInitializer(BaseTensorInitializer):
    def __init__(
        self,
        shape: Sequence[int],
        a: float = 0,
        mode: torch.nn.init._FanMode = "fan_in",
        nonlinearity: torch.nn.init._NonlinearityType = "leaky_relu",
    ):
        self._shape = shape
        self._a = a
        self._mode: torch.nn.init._FanMode = mode
        self._nonlinearity: torch.nn.init._NonlinearityType = nonlinearity

    def __call__(self) -> torch.Tensor:
        tensor = torch.empty(self._shape)
        torch.nn.init.kaiming_normal_(tensor, a=self._a, mode=self._mode, nonlinearity=self._nonlinearity)
        return tensor


@BaseTensorInitializer.register("orthogonal")
class OrthogonalTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int], gain: float = 1.0):
        self._shape = shape
        self._gain = gain

    def __call__(self) -> torch.Tensor:
        tensor = torch.empty(self._shape)
        torch.nn.init.orthogonal_(tensor, gain=self._gain)
        return tensor


@BaseTensorInitializer.register("sparse")
class SparseTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int], sparsity: float = 0.1, std: float = 0.01):
        self._shape = shape
        self._sparsity = sparsity
        self._std = std

    def __call__(self) -> torch.Tensor:
        tensor = torch.empty(self._shape)
        torch.nn.init.sparse_(tensor, sparsity=self._sparsity, std=self._std)
        return tensor


@BaseTensorInitializer.register("zeros")
class ZerosTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int]):
        self._shape = shape

    def __call__(self) -> torch.Tensor:
        return torch.zeros(self._shape)


@BaseTensorInitializer.register("ones")
class OnesTensorInitializer(BaseTensorInitializer):
    def __init__(self, shape: Sequence[int]):
        self._shape = shape

    def __call__(self) -> torch.Tensor:
        return torch.ones(self._shape)
