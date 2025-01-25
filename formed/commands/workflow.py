import argparse
import pathlib
import sys
from logging import getLogger

from formed.settings import load_formed_settings
from formed.workflow import WorkflowExecutionID, WorkflowExecutionInfo, WorkflowGraph

from .subcommand import Subcommand

logger = getLogger(__name__)


@Subcommand.register("workflow")
class WorkflowCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument(
            "--settings",
            type=str,
            default=None,
            help="workflow environment settings file path",
        )


@WorkflowCommand.register("run")
class WorkflowRunCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument(
            "config",
            type=str,
            help="config file path",
        )
        self.parser.add_argument(
            "--execution-id",
            type=WorkflowExecutionID,
            default=None,
            help="execution id",
        )
        self.parser.add_argument(
            "--step",
            type=str,
            default=None,
            help="step name to execute",
        )
        self.parser.add_argument(
            "--overrides",
            type=str,
            default=None,
            help="overrides jsonnet file path",
        )

    def run(self, args: argparse.Namespace) -> None:
        formed_settings = load_formed_settings(args.settings)
        settings = formed_settings.workflow
        config_path = pathlib.Path(args.config)

        logger.info(f"Load workflow from {config_path}")
        graph = WorkflowGraph.from_jsonnet(config_path, overrides=args.overrides)
        if args.step is not None:
            graph = graph.get_subgraph(args.step)

        execution = WorkflowExecutionInfo(graph, id=args.execution_id)

        organizer = settings.organizer
        executor = settings.executor

        if execution.id is not None and organizer.exists(execution.id):
            logger.error(f"Execution {execution.id} already exists")
            sys.exit(1)

        organizer.run(executor, execution)


@WorkflowCommand.register("remove")
class WorkflowRemoveCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument(
            "execution_id",
            type=WorkflowExecutionID,
            help="execution id",
        )

    def run(self, args: argparse.Namespace) -> None:
        formed_settings = load_formed_settings(args.settings)
        settings = formed_settings.workflow
        organizer = settings.organizer

        if not organizer.exists(args.execution_id):
            logger.error(f"Execution {args.execution_id} does not exist")
            sys.exit(1)

        organizer.remove(args.execution_id)
