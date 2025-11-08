from flax import nnx
from flax.training import train_state


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef
    additional_states: tuple[nnx.State, ...] = ()
