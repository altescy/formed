import argparse

from formed import __version__
from formed.commands import workflow  # noqa: F401
from formed.commands.subcommand import Subcommand


def create_subcommand(prog: str | None = None) -> Subcommand:
    parser = argparse.ArgumentParser(usage="%(prog)s", prog=prog)
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return Subcommand(parser)


def main(prog: str | None = None) -> None:
    app = create_subcommand(prog)
    app()
