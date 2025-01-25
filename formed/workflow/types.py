from typing import Union

from formed.types import JsonValue

StepConfig = dict[str, JsonValue]
StrictParamPath = tuple[Union[int, str], ...]
