"""Jsonnet configuration loading and processing utilities.

This module provides utilities for loading Jsonnet configuration files with support
for external variables and configuration overrides. It enables flexible configuration
management through Jsonnet's template language.

Key Features:
    - Load Jsonnet files with external variable substitution
    - Apply runtime configuration overrides
    - Automatic environment variable access
    - FromJsonnet mixin for easy Jsonnet-based object construction

Example:
    >>> # Load a Jsonnet configuration file
    >>> config = load_jsonnet(
    ...     "workflow.jsonnet",
    ...     ext_vars={"dataset": "train"},
    ...     overrides="steps.preprocess.batch_size=64"
    ... )
    >>>
    >>> # Use FromJsonnet mixin in your class
    >>> class MyWorkflow(FromJsonnet):
    ...     pass
    >>>
    >>> workflow = MyWorkflow.from_jsonnet("config.jsonnet")

"""

import copy
import itertools
import json
import os
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any, ClassVar, Optional, TypeVar, Union, cast

from colt.builder import ColtBuilder
from rjsonnet import evaluate_file, evaluate_snippet

T = TypeVar("T", dict, list)


def _is_encodable(value: str) -> bool:
    return (value == "") or (value.encode("utf-8", "ignore") != b"")


def _environment_variables() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if _is_encodable(value)}


def _parse_overrides(serialized_overrides: str, ext_vars: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if serialized_overrides:
        ext_vars = {**_environment_variables(), **(ext_vars or {})}
        output = json.loads(evaluate_snippet("", serialized_overrides, ext_vars=ext_vars))
        assert isinstance(output, dict), "Overrides must be a JSON object."
        return output
    return {}


def _with_overrides(original: T, overrides_dict: dict[str, Any], prefix: str = "") -> T:
    merged: T
    keys: Union[Iterable[str], Iterable[int]]
    if isinstance(original, list):
        merged = [None] * len(original)
        keys = cast(Iterable[int], range(len(original)))
    elif isinstance(original, dict):
        merged = cast(T, {})
        keys = cast(
            Iterable[str],
            itertools.chain(
                original.keys(),
                (k for k in overrides_dict if "." not in k and k not in original),
            ),
        )
    else:
        if prefix:
            raise ValueError(
                f"overrides for '{prefix[:-1]}.*' expected list or dict in original, found {type(original)} instead"
            )
        else:
            raise ValueError(f"expected list or dict, found {type(original)} instead")

    used_override_keys: set[str] = set()
    for key in keys:
        if str(key) in overrides_dict:
            merged[key] = copy.deepcopy(overrides_dict[str(key)])  # pyright: ignore[reportArgumentType, reportCallIssue]
            used_override_keys.add(str(key))
        else:
            overrides_subdict = {}
            for o_key in overrides_dict:
                if o_key.startswith(f"{key}."):
                    overrides_subdict[o_key[len(f"{key}.") :]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                merged[key] = _with_overrides(original[key], overrides_subdict, prefix=prefix + f"{key}.")  # pyright: ignore[reportArgumentType, reportCallIssue]
            else:
                merged[key] = copy.deepcopy(original[key])  # pyright: ignore[reportArgumentType, reportCallIssue]

    unused_override_keys = [prefix + key for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f"overrides dict contains unused keys: {unused_override_keys}")

    return merged


def load_jsonnet(
    filename: Union[str, PathLike],
    ext_vars: Optional[Mapping[str, Any]] = None,
    overrides: Optional[str] = None,
) -> Any:
    """Load and evaluate a Jsonnet configuration file.

    This function loads a Jsonnet file, evaluates it with optional external variables,
    and applies runtime configuration overrides. Environment variables are automatically
    made available to the Jsonnet template.

    Args:
        filename: Path to the Jsonnet configuration file.
        ext_vars: External variables to pass to Jsonnet. These are accessible
            in the template as `std.extVar("key")`.
        overrides: Jsonnet expression for runtime overrides. Must evaluate to
            an object with keys like "path.to.field=value".

    Returns:
        The evaluated configuration (typically a dict or list).

    Example:
        >>> # Basic usage
        >>> config = load_jsonnet("config.jsonnet")
        >>>
        >>> # With external variables
        >>> config = load_jsonnet(
        ...     "workflow.jsonnet",
        ...     ext_vars={"mode": "train", "epochs": 10}
        ... )
        >>>
        >>> # With overrides
        >>> config = load_jsonnet(
        ...     "workflow.jsonnet",
        ...     overrides="{\"steps.train.epochs\": 20}"
        ... )

    Note:
        - Environment variables are automatically available in ext_vars
        - Overrides are applied after Jsonnet evaluation
        - Overrides use dot-separated paths (e.g., "steps.train.batch_size")

    """
    ext_vars = {**_environment_variables(), **(ext_vars or {})}
    output = json.loads(evaluate_file(str(filename), ext_vars=ext_vars))
    if overrides:
        output = _with_overrides(output, _parse_overrides(overrides, ext_vars=ext_vars))
    return output


_T_FromJsonnet = TypeVar("_T_FromJsonnet", bound="FromJsonnet")


class FromJsonnet:
    """Mixin class for loading objects from Jsonnet configuration files.

    FromJsonnet provides a class method `from_jsonnet()` that loads a Jsonnet
    configuration file and constructs an instance using the colt builder pattern.
    Classes that inherit from this mixin can be instantiated from Jsonnet configs.

    Class Attributes:
        __COLT_BUILDER__: The colt builder instance used for object construction.
            Defaults to ColtBuilder with "type" as the type key.

    Example:
        >>> from formed.common.jsonnet import FromJsonnet
        >>> from colt import Registrable
        >>>
        >>> class MyModel(FromJsonnet, Registrable):
        ...     def __init__(self, learning_rate: float):
        ...         self.learning_rate = learning_rate
        >>>
        >>> # Load from Jsonnet file
        >>> model = MyModel.from_jsonnet("config.jsonnet")
        >>>
        >>> # Access the original config
        >>> print(model.to_dict())

    Note:
        - Inheriting classes should be compatible with colt's builder pattern
        - The loaded config is stored in the instance as `__config__` attribute
        - Override `__pre_init__()` to modify config before object construction

    """

    __COLT_BUILDER__: ClassVar = ColtBuilder(typekey="type")

    @classmethod
    def from_jsonnet(
        cls: type[_T_FromJsonnet],
        filename: Union[str, PathLike],
        ext_vars: Optional[Mapping[str, Any]] = None,
        overrides: Optional[str] = None,
    ) -> _T_FromJsonnet:
        """Load an instance from a Jsonnet configuration file.

        Args:
            filename: Path to the Jsonnet configuration file.
            ext_vars: External variables to pass to Jsonnet.
            overrides: Runtime configuration overrides.

        Returns:
            An instance of the class constructed from the Jsonnet config.

        Example:
            >>> class Workflow(FromJsonnet):
            ...     def __init__(self, steps: dict):
            ...         self.steps = steps
            >>>
            >>> workflow = Workflow.from_jsonnet("workflow.jsonnet")

        """
        config = load_jsonnet(filename, ext_vars=ext_vars, overrides=overrides)
        config = cls.__pre_init__(config)
        obj: _T_FromJsonnet = cls.__COLT_BUILDER__(config, cls)
        setattr(obj, "__config__", config)
        return obj

    @classmethod
    def __pre_init__(cls, config: Any) -> Any:
        return config

    def to_dict(self) -> Any:
        return getattr(self, "__config__")
