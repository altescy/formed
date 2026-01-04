"""Jsonnet configuration loading and processing utilities.

This module provides utilities for loading Jsonnet configuration files with support
for external variables and configuration overrides. It enables flexible configuration
management through Jsonnet's template language.

Key Features:
    - Load Jsonnet files with external variable substitution
    - Apply runtime configuration overrides
    - Automatic environment variable access
    - FromJsonnet mixin for easy Jsonnet-based object construction

Examples:
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
from typing_extensions import Self

from formed.types import JsonValue

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
    else:
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
            merged[key] = copy.deepcopy(overrides_dict[str(key)])  # pyright: ignore[reportCallIssue, reportArgumentType]
            used_override_keys.add(str(key))
        else:
            overrides_subdict = {}
            for o_key in overrides_dict:
                if o_key.startswith(f"{key}."):
                    overrides_subdict[o_key[len(f"{key}.") :]] = overrides_dict[o_key]
                    used_override_keys.add(o_key)
            if overrides_subdict:
                merged[key] = _with_overrides(original[key], overrides_subdict, prefix=prefix + f"{key}.")  # pyright:ignore[reportCallIssue, reportArgumentType]
            else:
                merged[key] = copy.deepcopy(original[key])  # pyright: ignore[reportCallIssue, reportArgumentType]

    unused_override_keys = [prefix + key for key in set(overrides_dict.keys()) - used_override_keys]
    if unused_override_keys:
        raise ValueError(f"overrides dict contains unused keys: {unused_override_keys}")

    return merged


def load_jsonnet(
    filename: Union[str, PathLike],
    ext_vars: Optional[Mapping[str, Any]] = None,
    overrides: Optional[str] = None,
) -> Any:
    ext_vars = {**_environment_variables(), **(ext_vars or {})}
    output = json.loads(evaluate_file(str(filename), ext_vars=ext_vars))
    if overrides:
        output = _with_overrides(output, _parse_overrides(overrides, ext_vars=ext_vars))
    return output


class FromJsonnet:
    __COLT_BUILDER__: ClassVar = ColtBuilder(typekey="type")
    __json_config__: JsonValue

    @classmethod
    def from_jsonnet(
        cls,
        filename: Union[str, PathLike],
        ext_vars: Optional[Mapping[str, Any]] = None,
        overrides: Optional[str] = None,
    ) -> Self:
        json_config = load_jsonnet(filename, ext_vars=ext_vars, overrides=overrides)
        return cls.from_json(json_config)

    @classmethod
    def __pre_init__(cls, config: Any) -> Any:
        return config

    def json(self) -> JsonValue:
        if not hasattr(self, "__json_config__"):
            raise RuntimeError(f"{self.__class__.__name__} instance has no JSON config")
        return self.__json_config__

    @classmethod
    def from_json(cls, o: JsonValue, /) -> Self:
        obj = cls.__COLT_BUILDER__(cls.__pre_init__(o), cls)
        obj.__json_config__ = o
        return obj
