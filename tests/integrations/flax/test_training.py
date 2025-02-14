from collections.abc import Mapping, Sequence
from typing import Optional, Union

import flax
import jax
import numpy
import pytest
from flax import nnx

from formed.integrations.flax import DataLoader, FlaxModel, FlaxTrainer


class Regressor(FlaxModel["Regressor.Input", "Regressor.Output", "Regressor.Params"]):
    @flax.struct.dataclass
    class Input:
        x: jax.Array
        y: Optional[jax.Array] = None

    @flax.struct.dataclass
    class Output:
        y: jax.Array
        loss: Optional[jax.Array] = None
        metrics: Optional[Mapping[str, jax.Array]] = None

    Params = type[None]

    def __init__(
        self,
        rngs: Union[int, nnx.Rngs] = 0,
        hidden_dim: int = 32,
    ) -> None:
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self._linear1 = nnx.Linear(1, hidden_dim, rngs=rngs)
        self._linear2 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        inputs: Input,
        params: Optional[Params] = None,
        *,
        train: bool = False,
    ) -> Output:
        h = jax.nn.relu(self._linear1(inputs.x))
        y = self._linear2(h)

        loss: Optional[jax.Array]
        if inputs.y is not None:
            loss = jax.numpy.mean((inputs.y - y) ** 2)
        else:
            loss = None

        return Regressor.Output(y=y, loss=loss)


def collate(data: Sequence[tuple[float, float]]) -> Regressor.Input:
    x, y = zip(*data)
    return Regressor.Input(x=jax.numpy.array(x)[:, None], y=jax.numpy.array(y))


@pytest.fixture
def regression_dataset() -> Sequence[tuple[float, float]]:
    X = numpy.linspace(0, 1, 200)
    Y = 0.8 * X**2 + 0.1 + numpy.random.normal(0, 0.1, size=X.shape)  # type: ignore
    return [(float(x), float(y)) for x, y in zip(X, Y)]  # type: ignore


def test_training(
    regression_dataset: Sequence[tuple[float, float]],
) -> None:
    rngs = nnx.Rngs(0)

    train_dataset = regression_dataset[:150]
    val_dataset = regression_dataset[150:]

    model = Regressor()

    trainer = FlaxTrainer[
        tuple[float, float],
        Regressor.Input,
        Regressor.Output,
        Regressor.Params,
    ](
        train_dataloader=DataLoader(collate, batch_size=32, shuffle=True),
    )
    trainer.train(rngs, model, train_dataset, val_dataset)
