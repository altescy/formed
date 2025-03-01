import jax

from formed.integrations.flax.modules import FeedForward


class TestFeedForward:
    def test_forward(self) -> None:
        model = FeedForward(features=4, num_layers=2)
        x = jax.numpy.ones((8, 4))
        y = model(x)
        assert y.shape == (8, 4)
