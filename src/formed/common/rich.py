from collections.abc import Callable, Iterable, Iterator, Sized
from contextlib import contextmanager
from functools import partial
from typing import Final, TypeVar

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.spinner import Spinner

T = TypeVar("T")


STDOUT_CONSOLE: Final[Console] = Console()
STDERR_CONSOLE: Final[Console] = Console(stderr=True)


@contextmanager
def progress(
    iterable: Iterable[T],
    desc: str | None = None,
    console: Console = STDERR_CONSOLE,
) -> Iterator[Iterator[T]]:
    desc = desc or "Processing...."

    def _iterator(callback: Callable[[], None] | None = None) -> Iterator[T]:
        for item in iterable:
            yield item
            if callback:
                callback()

    if isinstance(iterable, Sized):
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(desc, total=len(iterable))
            yield _iterator(partial(progress.update, task, advance=1))
    else:
        spinner = Spinner("dots", text=desc)
        with Live(
            spinner,
            refresh_per_second=10,
            console=console,
        ):
            yield _iterator()
