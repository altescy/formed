# Quick Start

## Installation

```shell
pip install formed
```

If you want to use integrations, install the corresponding extra packages:

```shell
pip install formed[mlflow]
pip install formed[torch]
pip install formed[flax]
pip install formed[transformers]
pip install formed[sentence-transformers]
pip install formed[all]  # install all integrations
```

## Basic Usage

This section walks you through building a simple workflow to explore properties
of the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture).
You'll learn the fundamental workflow of formed:

1. Define steps as reusable computation units
2. Define a workflow configuration that connects steps
3. Configure formed settings
4. Run the workflow via CLI
5. Inspect the results programmatically

### 1. Define Steps

Steps are the building blocks of workflows. Each step is a Python function
decorated with `@workflow.step`:

```python
# collatz.py

from collections.abc import Iterator
from formed import workflow


@workflow.step
def generate_collatz(init: int) -> Iterator[int]:
    n = init
    while n != 1:
        yield n
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
    yield 1


@workflow.step
def count_steps(sequence: Iterator[int]) -> int:
    return sum(1 for _ in sequence) - 1


@workflow.step
def max_value(sequence: Iterator[int]) -> int:
    return max(sequence)


@workflow.step
def summarize(**kwargs: int) -> dict[str, int]:
    logger = workflow.use_step_logger(__name__)
    logger.info(f"Summary: {kwargs}")
    return kwargs
```

Each step is automatically cached based on its inputs and source code, so
re-running the workflow will skip unchanged steps.

### 2. Define Workflow

Workflows are defined using Jsonnet (or JSON) configuration files. The
configuration specifies which steps to run and how to connect their inputs and outputs:

```jsonnet
// config.jsonnet

{
    steps: {
        collatz_sequence: {
            type: "generate_collatz",
            init: 7,
        },
        step_count: {
            type: "count_steps",
            sequence: {type: "ref", ref: "collatz_sequence"},
        },
        summary: {
            type: "summarize",
            step_count: {type: "ref", ref: "step_count"},
        },
    }
}
```

The `type` field specifies which step function to use, and other fields are
passed as arguments. Use `{type: "ref", ref: "step_name"}` to reference outputs
from other steps.

### 3. Configure Formed

```yaml
# formed.yml

workflow:
  organizer:
    type: filesystem

required_modules:
  - collatz
```

The `formed.yml` file configures how formed manages workflows:

- **`workflow.organizer`**: Specifies how to store and manage workflow execution results. The `filesystem` organizer saves results to local files in the `.formed` directory.
- **`required_modules`**: Lists Python modules containing step definitions. Formed automatically loads these modules and all their submodules, making steps available for use in workflows.

### 4. Run Workflow

Execute the workflow from the command line in the same directory as `formed.yml`:

```bash
formed workflow run config.jsonnet --execution-id ex1
```

The `--execution-id` flag assigns a unique identifier to this execution,
allowing you to retrieve results later. Formed will execute the workflow DAG,
caching each step's results.

### 5. Check Results

After execution completes, you can programmatically access the results:

```python
from formed import workflow
from formed.settings import load_formed_settings

settings = load_formed_settings("formed.yml")
organizer = settings.organizer

context = organizer.get(workflow.WorkflowExecutionID("ex1"))
assert context is not None

summary = context.cache[context.info.graph["summary"]]
```

The `organizer.get()` method retrieves the execution context by ID, and `context.cache` provides access to cached step results. You can use step names (like `"summary"`) to look up results in the workflow graph.

For more details on workflow concepts and advanced features, see the [Workflow Guide](./guides/workflow.md).

## Python API

You can also define and run workflows entirely in Python without configuration files:

```python
import logging
from formed import workflow

logging.basicConfig(level=logging.INFO)
organizer = workflow.MemoryWorkflowOrganizer()
executor = workflow.DefaultWorkflowExecutor()
graph = workflow.WorkflowGraph.from_config(
    {
        "steps": {
            "collatz_sequence": {
                "type": "generate_collatz",
                "init": 7,
            },
            "step_count": {
                "type": "count_steps",
                "sequence": {"type": "ref", "ref": "collatz_sequence"},
            },
            "summary": {
                "type": "summarize",
                "step_count": {"type": "ref", "ref": "step_count"},
            },
        }
    }
)
context = organizer.run(executor, graph)
print(f"{context.cache[context.info.graph['summary']]=}")
```

This approach uses `MemoryWorkflowOrganizer` to store results in memory instead
of on disk, which is useful for quick experiments and testing.

## Working with Integrations

Formed integrates with popular tools like MLflow, PyTorch, and
🤗 Transformers. This example demonstrates using MLflow to track experiments.

First, install the MLflow integration:

```bash
pip install formed[mlflow]
```

Then update your `formed.yml` to use the MLflow organizer:

```diff
# formed.yml

workflow:
  organizer:
-   type: filesystem
+   type: mlflow
+   experiment_name: collatz

required_modules:
  - collatz
+ - formed.integrations.mlflow
```

The MLflow organizer automatically logs workflow executions as MLflow runs, including parameters, metrics, and artifacts.

Run the workflow as before:

```bash
formed workflow run config.jsonnet --execution-id ex1
```

View the results in the MLflow UI:

```bash
mlflow ui
```

This integration allows you to track experiments, compare results, and manage
model versions using MLflow's interface. For more integration examples, see the
[Tutorials](./tutorials/index.md).

## Further Reading

- **[Tutorials](./tutorials/index.md)**: Practical examples including text classification and language model fine-tuning
- **[Guides](./guides/index.md)**: In-depth explanations of workflow concepts, caching, and advanced features
- **[API Reference](./reference/index.md)**: Complete API documentation for all modules and integrations
