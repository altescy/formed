from collections.abc import Mapping
from typing import Any, Final, Self


class BaseSingleton:
    VALUE: Final[Self]  # pyright:ignore[reportGeneralTypeIssues]

    def __init_subclass__(
        cls,
        singleton_init: Mapping[str, Any] = {},
    ) -> None:
        super().__init_subclass__()

        value = cls.__new__(cls)
        value.__init__(**singleton_init)
        setattr(cls, "VALUE", value)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
