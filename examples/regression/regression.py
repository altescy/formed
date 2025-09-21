import dataclasses
from collections.abc import Mapping
from typing import Optional, Union

import jax
import numpy
from flax import nnx, struct

from formed.integrations.flax import FlaxModel
from formed.workflow import step


@dataclasses.dataclass
class RegressionExample:
    x: numpy.ndarray
    y: float


@struct.dataclass
class RegressorInput:
    x: jax.Array
    y: Optional[jax.Array] = None


@struct.dataclass
class RegressorOutput:
    y: jax.Array
    loss: Optional[jax.Array] = None
    metrics: Optional[Mapping[str, jax.Array]] = None


@FlaxModel.register("tiny-regressor")
class Regressor(FlaxModel[RegressorInput, RegressorOutput, None]):
    Input = RegressorInput
    Output = RegressorOutput

    def __init__(self, rngs: Union[int, nnx.Rngs] = 0, hidden_dim: int = 32) -> None:
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self._linear1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self._linear2 = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self._dropout = nnx.Dropout(rate=0.1, rngs=rngs)

    def __call__(
        self,
        inputs: RegressorInput,
        params: None = None,
        *,
        train: bool = False,
    ) -> RegressorOutput:
        del params

        h = jax.nn.relu(self._linear1(inputs.x))
        h = self._dropout(h, deterministic=not train)
        y = self._linear2(h)

        loss: Optional[jax.Array]
        if inputs.y is not None:
            assert inputs.y is not None
            loss = jax.numpy.mean((inputs.y - y) ** 2)
        else:
            loss = None

        return Regressor.Output(y=y, loss=loss)


@step("load_dataset", format="pickle")
def load_dataset(size: int = 200) -> list[RegressionExample]:
    X = numpy.linspace(0, 1, size)
    Y = 0.8 * X**2 + 0.1 + numpy.random.normal(0, 0.1, size=X.shape)  # type: ignore
    return [RegressionExample(x, float(y)) for x, y in zip(X[:, None], Y)]  # type: ignore
