from importlib.metadata import version

from colt import ConfigurationError, Lazy, Registrable

__version__ = version("formed")
__all__ = [
    # colt
    "ConfigurationError",
    "Lazy",
    "Registrable",
]
