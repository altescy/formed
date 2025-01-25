import dataclasses
import importlib


@dataclasses.dataclass(frozen=True)
class PackageInfo:
    name: str
    version: str


def get_installed_packages() -> list[PackageInfo]:
    distributions = importlib.metadata.distributions()
    return sorted([PackageInfo(d.metadata["Name"], d.version) for d in distributions], key=lambda p: p.name)
