import argparse
import math
import pathlib
from collections.abc import Callable, Iterator
from logging import getLogger
from typing import cast

from formed.commands.subcommand import Subcommand

logger = getLogger(__name__)


@Subcommand.register("torch")
class TorchCommand(Subcommand):
    """Base command for Torch-related subcommands."""

    pass


@TorchCommand.register("describe")
class TorchDescribeCommand(TorchCommand):
    """Describe Torch integration details."""


@TorchDescribeCommand.register("trainer")
class TorchDescribeTrainerCommand(TorchDescribeCommand):
    """Describe Torch trainer details."""

    def setup(self) -> None:
        self.parser.add_argument(
            "config",
            type=str,
            help="config file path",
        )
        self.parser.add_argument(
            "--trainer",
            type=str,
            default=None,
            help="trainer path to describe",
        )
        self.parser.add_argument(
            "--train-data-size",
            type=int,
            default=1000,
            help="size of training data",
        )
        self.parser.add_argument(
            "--train-batch-size",
            type=int,
            default=None,
            help="training batch size (overrides config)",
        )

    def run(self, args: argparse.Namespace) -> None:
        import torch
        from colt import Lazy
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        from formed.common.attributeutils import xgetattr
        from formed.common.jsonnet import load_jsonnet
        from formed.common.rich import BraillePlot
        from formed.workflow.colt import COLT_BUILDER

        from .distributors import BaseDistributor
        from .training.engine import get_default_lr_scheduler_factory, get_default_optimizer_factory
        from .training.trainer import get_default_distributor, get_default_max_epochs
        from .types import LRSchedulerFactory, OptimizerFactory

        def find_batch_size(config: dict) -> int | None:
            batch_size = config.get("batch_size")
            if isinstance(batch_size, int):
                return batch_size
            for value in config.values():
                if isinstance(value, dict):
                    if bs := find_batch_size(value):
                        return bs
            return None

        config: dict = load_jsonnet(pathlib.Path(args.config))
        if args.trainer:
            config = xgetattr(config, args.trainer)

        max_epochs: int = config.get("max_epochs", get_default_max_epochs())
        train_batch_size: int = args.train_batch_size or find_batch_size(config) or 32
        max_train_steps: int | None = None
        if train_batch_size is not None:
            max_train_steps = math.ceil(args.train_data_size / train_batch_size * max_epochs)

        optimizer_factory: OptimizerFactory = get_default_optimizer_factory()
        if "optimizer" in config.get("engine", {}):
            optimizer_factory = COLT_BUILDER(
                config["engine"]["optimizer"],
                cast(Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer], OptimizerFactory),
            )

        lr_scheduler_factory: LRSchedulerFactory | None = get_default_lr_scheduler_factory()
        if "lr_scheduler" in config.get("engine", {}):
            lr_scheduler_factory = COLT_BUILDER(
                config["engine"]["lr_scheduler"],
                cast(Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler], LRSchedulerFactory),
            )

        distributor: BaseDistributor = get_default_distributor()
        if "distributor" in config:
            distributor = COLT_BUILDER(config["distributor"], BaseDistributor)

        console = Console()

        # Display title with decorative header
        console.print()
        title_text = Text()
        title_text.append("TorchTrainer Configuration Summary", style="bold cyan")
        console.print(Panel.fit(title_text, border_style="bright_cyan", padding=(0, 2)))
        console.print()

        # Create training configuration table with better layout
        config_table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        config_table.add_column("Property", style="bold bright_blue", no_wrap=True, width=20)
        config_table.add_column("Value", style="bright_white")

        config_table.add_row("Dataset Size", f"[cyan]{args.train_data_size:,}[/cyan] samples")
        config_table.add_row("Batch Size", f"[cyan]{train_batch_size}[/cyan]")
        config_table.add_row("Epochs", f"[cyan]{max_epochs}[/cyan]")
        if max_train_steps:
            steps_per_epoch = max_train_steps // max_epochs
            config_table.add_row(
                "Training Steps", f"[cyan]{max_train_steps:,}[/cyan] total ([dim]{steps_per_epoch}[/dim] per epoch)"
            )

        console.print(
            Panel(
                config_table,
                title="[bold bright_blue]Training Configuration[/bold bright_blue]",
                border_style="blue",
            )
        )
        console.print()

        # Display optimizer configuration with cleaner formatting
        if isinstance(optimizer_factory, Lazy):
            optimizer_class = optimizer_factory._cls.__name__ if optimizer_factory._cls else "Unknown"
            optimizer_config = optimizer_factory._config

            opt_table = Table(show_header=False, box=None, padding=(0, 1))
            opt_table.add_column("Key", style="bold green")
            opt_table.add_column("Value", style="bright_white")

            opt_table.add_row("Type", optimizer_class)
            if optimizer_config:
                for k, v in optimizer_config.items():
                    if k != "type":
                        opt_table.add_row(k, str(v))

            console.print(Panel(opt_table, title="[bold green]Optimizer[/bold green]", border_style="green"))
        else:
            optimizer_info = str(optimizer_factory)
            console.print(
                Panel(
                    Text(optimizer_info, style="green"),
                    title="[bold green]Optimizer[/bold green]",
                    border_style="green",
                )
            )

        console.print()

        # Display distributor configuration with icon
        distributor_class_name = distributor.__class__.__name__
        distributor_info = Text()
        distributor_info.append(distributor_class_name, style="bold magenta")

        if hasattr(distributor, "__dict__"):
            params = {k: v for k, v in distributor.__dict__.items() if not k.startswith("_") and v is not None}
            if params:
                distributor_info.append("\n")
                for k, v in params.items():
                    distributor_info.append(f"{k}: ", style="dim")
                    distributor_info.append(str(v), style="bright_white")
                    distributor_info.append("\n")

        console.print(Panel(distributor_info, title="[bold magenta]Distributor[/bold magenta]", border_style="magenta"))
        console.print()

        # Display learning rate schedule
        if lr_scheduler_factory and max_train_steps:
            from rich.console import Group

            # Create a dummy model and construct optimizer using the actual factory
            dummy_model = torch.nn.Linear(1, 1)
            if isinstance(optimizer_factory, Lazy):
                dummy_optimizer = optimizer_factory.construct(params=dummy_model.parameters())
            elif callable(optimizer_factory):
                dummy_optimizer = optimizer_factory(dummy_model.parameters())
            else:
                dummy_optimizer = optimizer_factory

            # Instantiate scheduler from factory
            if isinstance(lr_scheduler_factory, Lazy):
                scheduler = lr_scheduler_factory.construct(optimizer=dummy_optimizer)
            elif callable(lr_scheduler_factory):
                scheduler = lr_scheduler_factory(dummy_optimizer)
            else:
                scheduler = lr_scheduler_factory

            # Get scheduler info - create configuration table
            scheduler_class = scheduler.__class__.__name__
            config_items = []
            if hasattr(scheduler, "__dict__"):
                params = {
                    k: v
                    for k, v in scheduler.__dict__.items()
                    if not k.startswith("_") and k not in ["optimizer", "base_lrs", "warmup_steps"]
                }
                config_items = list(params.items())

            # Simulate learning rate changes
            num_plot_points = min(max_train_steps, 200)  # Limit plot points
            step_interval = max(1, max_train_steps // num_plot_points)

            steps = []
            lrs = []

            # Cast to actual PyTorch scheduler for accessing internal methods
            torch_scheduler = cast(torch.optim.lr_scheduler._LRScheduler, scheduler)

            for step in range(0, max_train_steps, step_interval):
                # Manually set last_epoch to simulate the step
                torch_scheduler.last_epoch = step
                current_lrs = torch_scheduler.get_lr()
                steps.append(step)
                lrs.append(current_lrs[0])  # Use first param group

            # Create compact configuration section
            config_text = Text()
            config_text.append(f"{scheduler_class}", style="bold yellow")
            if config_items:
                config_text.append(" (")
                for i, (k, v) in enumerate(config_items[:4]):  # Show first 4 params
                    if i > 0:
                        config_text.append(", ")
                    config_text.append(f"{k}=", style="dim")
                    config_text.append(str(v), style="bright_white")
                if len(config_items) > 4:
                    config_text.append(", ...", style="dim")
                config_text.append(")")

            # Create plot with adaptive width
            # Calculate available width: console width - panel borders (2) - padding (4) - margin (4)
            available_width = max(40, console.width - 20)  # Minimum 40 chars
            plot = BraillePlot(available_width, 18)
            plot.plot_line(steps, lrs)

            # Create statistics in a horizontal layout
            lr_stats_table = Table(show_header=True, box=None, padding=(0, 2), expand=True)
            lr_stats_table.add_column("Initial", style="bright_yellow", justify="center")
            lr_stats_table.add_column("Peak", style="bright_green", justify="center")
            lr_stats_table.add_column("Min", style="bright_blue", justify="center")
            lr_stats_table.add_column("Final", style="bright_magenta", justify="center")

            lr_stats_table.add_row(f"{lrs[0]:.2e}", f"{max(lrs):.2e}", f"{min(lrs):.2e}", f"{lrs[-1]:.2e}")

            # Group all scheduler information together
            scheduler_group = Group(
                config_text,
                Text(""),
                plot.render_rich(style="yellow"),
                Text(""),
                lr_stats_table,
            )

            console.print(
                Panel(
                    scheduler_group,
                    title="[bold yellow]Learning Rate Scheduler[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )
            console.print()

        elif lr_scheduler_factory:
            # Display scheduler info without plot if max_train_steps is not available
            scheduler_info = str(lr_scheduler_factory)
            console.print(
                Panel(
                    Text(scheduler_info, style="yellow"),
                    title="[bold yellow]LR Scheduler[/bold yellow]",
                    border_style="yellow",
                )
            )
            console.print()

        # Display callbacks if present with better formatting
        if "callbacks" in config and config["callbacks"]:
            callbacks_text = Text()
            for i, cb_config in enumerate(config["callbacks"]):
                if i > 0:
                    callbacks_text.append("\n")

                if isinstance(cb_config, dict) and "type" in cb_config:
                    cb_name = cb_config["type"]
                else:
                    cb_name = str(cb_config)

                callbacks_text.append("• ", style="bright_blue")
                callbacks_text.append(cb_name, style="bright_white")
            console.print(
                Panel(
                    callbacks_text,
                    title="[bold bright_blue] Callbacks[/bold bright_blue]",
                    border_style="bright_blue",
                )
            )
            console.print()
