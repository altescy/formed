import dataclasses
import os
import subprocess
from os import PathLike
from pathlib import Path
from typing import Optional, Union


@dataclasses.dataclass(frozen=True)
class GitInfo:
    commit: str
    branch: str


class GitClient:
    @staticmethod
    def is_available() -> bool:
        return (
            subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        )

    def __init__(self, directory: Union[str, PathLike]) -> None:
        if not self.is_available():
            raise RuntimeError("Git is not available")

        self._directory = Path(directory)

    def root(self) -> Path:
        return Path(os.popen(f"git -C {self._directory} rev-parse --show-toplevel").read().strip())

    def is_initialized(self) -> bool:
        return (
            subprocess.run(
                ["git", "status"], cwd=self._directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode
            == 0
        )

    def diff(self) -> str:
        return os.popen(f"git -C {self._directory} diff").read()

    def show(
        self,
        commit: Optional[str] = None,
        format: Optional[str] = None,
        no_patch: bool = False,
    ) -> str:
        command = f"git -C {self._directory} show"
        if commit:
            command += f" {commit}"
        if format:
            command += f" --format='{format}'"
        if no_patch:
            command += " --no-patch"
        return os.popen(command).read()

    def get_commit(self) -> str:
        return self.show(no_patch=True, format="%H").strip()

    def get_branch(self) -> str:
        return os.popen(f"git -C {self._directory} rev-parse --abbrev-ref HEAD").read().strip()

    def get_info(self) -> GitInfo:
        commit = self.get_commit()
        branch = self.get_branch()
        return GitInfo(commit, branch)


def get_git_info(directory: Optional[Union[str, PathLike]] = None) -> Optional[GitInfo]:
    if not GitClient.is_available():
        return None
    directory = directory or Path.cwd()
    client = GitClient(directory)
    if not client.is_available() or not client.is_initialized():
        return None
    return client.get_info()
