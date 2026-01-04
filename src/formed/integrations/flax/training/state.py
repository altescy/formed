"""Training state management for Flax NNX models.

This module extends Flax's TrainState to support NNX models by storing
the graph definition and additional states (BatchStat, RngState) separately
from trainable parameters.

"""

from typing import Generic, TypeVar

from flax import nnx
from flax.training import train_state

Node = TypeVar("Node")


class TrainState(train_state.TrainState, Generic[Node]):
    """Extended training state for Flax NNX models.

    This class extends flax.training.train_state.TrainState to work with
    Flax NNX models, storing the model's graph definition and additional
    states (like batch statistics and RNG states) separately from parameters.

    Attributes:
        graphdef: NNX graph definition describing the model structure.
        additional_states: Tuple of additional states (BatchStat, RngState, etc.).
        params: Trainable parameters (inherited from TrainState).
        opt_state: Optimizer state (inherited from TrainState).
        step: Training step counter (inherited from TrainState).
        tx: Optimizer transformation (inherited from TrainState).

    Examples:
        >>> # Create state from model
        >>> graphdef, params, *states = nnx.split(model, nnx.Param, nnx.BatchStat)
        >>> state = TrainState.create(
        ...     apply_fn=None,
        ...     graphdef=graphdef,
        ...     additional_states=tuple(states),
        ...     params=params,
        ...     tx=optimizer
        ... )
        >>>
        >>> # Reconstruct model
        >>> model = nnx.merge(state.graphdef, state.params, *state.additional_states)

    """

    graphdef: nnx.GraphDef[Node]
    additional_states: tuple[nnx.State, ...] = ()
