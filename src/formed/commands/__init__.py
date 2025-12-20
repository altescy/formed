import argparse

from formed import __version__
from formed.commands import workflow
from formed.commands.subcommand import Subcommand
from formed.settings import load_formed_settings

__all__ = [
    "create_subcommand",
    "main",
    "workflow",
]


def create_subcommand(prog: str | None = None) -> Subcommand:
    parser = argparse.ArgumentParser(usage="%(prog)s", prog=prog)
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return Subcommand(parser)


def main(prog: str | None = None) -> None:
    load_formed_settings(None)
    app = create_subcommand(prog)
    app()
