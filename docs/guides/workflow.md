# Workflow Guide

## Overview

Formed's workflow system provides a flexible framework for organizing computational pipelines with automatic caching, dependency tracking, and reproducible execution. This guide explains the core concepts and how to work with workflows using Python and Jsonnet/JSON configuration files.

## Core Concepts

### Architecture Components

Formed's workflow system is built around the following core components:

- **`WorkflowStep`**
  - The fundamental unit of computation, defined using the `@workflow.step` decorator
  - Each step represents a reusable, cacheable processing unit
- **`WorkflowGraph`**
  - A Directed Acyclic Graph (DAG) that defines dependencies between steps
  - Built from Jsonnet/JSON configuration files
- **`WorkflowExecutor`**
  - Executes the DAG based on dependency relationships
  - Provides sequential execution with automatic dependency resolution
- **`WorkflowCallback`**
  - Provides hooks for additional processing at step start/end and execution start/end
  - Useful for experiment tracking, monitoring execution results, and integration with experiment management tools
- **`WorkflowCache`**
  - Loads and restores step execution results based on content-based hashes
  - Enables automatic memoization and reproducible experiments
- **`Format`**
  - Handles serialization and deserialization of execution results
  - Can be configured per-step (e.g., `pickle`, `json`, custom formats)
- **`WorkflowOrganizer`**
  - Manages workflow execution and results
  - Multiple organizer types available: `memory`, `filesystem`, `mlflow`
  - Configured via `formed.yml`

### How Workflows Execute

Workflows are executed as Directed Acyclic Graphs (DAGs) where:

- **Dependency resolution**: The executor determines execution order based on step dependencies
- **Content-based caching**: Each step has a fingerprint computed from its source code and parameters
  - Steps are only re-executed when their fingerprint changes
  - Code changes are detected via AST structure, so comments and whitespace changes are ignored
- **Automatic memoization**: Results are cached and restored based on fingerprints
  - Enables reproducible experiments and efficient re-execution

## Working with Steps

### Defining Steps

The most basic way to define a step is to decorate a function with `@workflow.step`:

```python
from formed import workflow

@workflow.step
def your_awesome_step(x: int, y: int) -> int:
    return x + y
```

Default behavior:

- **Caching**: Results are automatically cached based on content-addressed fingerprints
- **Format auto-selection**: Storage format is automatically chosen based on data type
  - JSON-compatible values and dataclass objects are saved as JSON
  - Other objects are serialized using cloudpickle
- **Change detection**: Steps are re-executed when argument values or function code changes
  - Code changes are detected via AST structure, so comments and whitespace changes are ignored
- **Jsonnet reference**: Steps are referenced in Jsonnet using `type: '<function_name>'`
  - Example: `{ steps: { result: { type: 'your_awesome_step', x: 123, y: 456 } } }`

### Step Behavior and Customization

You can customize step behavior using decorator parameters and type annotations:

```python
from typing import Annotated
from formed.workflow import WorkflowStepArgFlag

@workflow.step(
    name="my::awesome_step",   # Custom name for the step
    version="001",             # Manual version control
    deterministic=True,        # Declare step determinism
    cacheable=True,            # Enable/disable caching
    format="json",             # Specify cache format
)
def your_awesome_step(
    x: int,
    y: Annotated[int, WorkflowStepArgFlag.IGNORE],  # Exclude from fingerprint
) -> int:
    return x + y
```

**Decorator parameters:**

- **`name`**: Assign a custom name instead of using the function name
  - Useful for namespacing or when refactoring function names
- **`version`**: Manual version string for the step
  - When specified, AST-based change detection is disabled
  - Increment this when you want to force cache invalidation
- **`deterministic`**: Flag indicating whether the step is deterministic
  - Set to `False` to prevent caching (e.g., for steps with random behavior or side effects)
- **`cacheable`**: Explicitly enable/disable caching (default: `True`)
  - Use `False` for steps that should always re-execute
- **`format`**: Specify the serialization format for results
  - Can be a string (e.g., `"json"`, `"pickle"`) or a `Format` class instance

**Argument annotations:**

- **`WorkflowStepArgFlag.IGNORE`**: Exclude specific arguments from the step's fingerprint
  - Changes to these arguments won't trigger re-execution
  - Useful for runtime configuration that doesn't affect results (e.g., device selection, logging verbosity)

### Step Runtime Context

Steps can access runtime context and utilities during execution:

```python
from formed.workflow import use_step_context, use_step_logger, use_step_workdir

@workflow.step
def my_step(x: int) -> int:
    # Access step context (info, state, fingerprint, etc.)
    context = use_step_context()

    # Access step-specific logger
    logger = use_step_logger()
    logger.info(f"Processing {x}")

    # Access step-specific working directory
    workdir = use_step_workdir()
    # Save temporary files to workdir

    return x * 2
```

These context managers provide:
- **`use_step_context()`**: Access to step metadata and execution state
- **`use_step_logger()`**: Step-specific logger instance
- **`use_step_workdir()`**: Dedicated working directory for the step

## Working with Configurations

### Configuration Structure

Workflows are defined using Jsonnet or JSON configuration files with the following structure:

```jsonnet
{
  steps: {                                      // Required root key
    dataset: {                                  // Arbitrary name for the step result
      type: 'generate_dataset',                 // Name of the step defined in Python
      size: 100                                 // Step arguments
    },
    model: {
      type: 'train_model',
      dataset: { type: 'ref', ref: 'dataset' }  // Reference another step's result
    }
  }
}
```

**Key concepts:**

- **`steps`**: Required root object containing all workflow steps
- **Step names**: Arbitrary identifiers for step results (e.g., `dataset`, `model`)
- **`type`**: The registered name of the step (function or class)
- **Step references**: Use `{ type: 'ref', ref: 'step_name' }` to pass one step's output as another step's input
- **Arguments**: All other keys in the step object are passed as arguments to the step

### Basic Workflow Example

**1. Define steps in Python:**

```python
# my_project.py
from formed import workflow

@workflow.step
def add_two_integers(a: int, b: int) -> int:
    return a + b
```

**2. Write Jsonnet configuration:**

Specify the step using `type` and provide corresponding arguments:

```jsonnet
// config.jsonnet
{
  steps: {
    result: {
      type: 'add_two_integers',
      a: 1,
      b: 2
    }
  }
}
```

**3. Configure required modules:**

In `formed.yml`, specify the Python modules containing your steps in `required_modules`.
All submodules under the specified modules will also be loaded:

```yaml
# formed.yml
required_modules:
  - my_project
```

**4. Execute from command line:**

```bash
$ ls
config.jsonnet  formed.yml  my_project.py

$ formed workflow run config.jsonnet
```

### Advanced Configuration Patterns

#### Object Construction

[altescy/colt](https://github.com/altescy/colt) automatically maps configuration to Python objects based on type hints and function signatures:

```python
class AwesomeProcessor:
    def __init__(self, name: str):
        ...

@workflow.step
def do_experiment(processor: AwesomeProcessor):
    ...
```

The following configuration automatically constructs an `AwesomeProcessor` instance and passes it to `do_experiment`:

```jsonnet
{
  steps: {
    experiment: {
      type: 'do_experiment',
      processor: {
        name: "Alice"  // Maps to AwesomeProcessor.__init__ parameters
      }
    }
  }
}
```

#### Using Registrable

Use `colt.Registrable` to register classes with a common interface, making it easy to inject and swap logic:

```python
import colt

class BaseCallback(colt.Registrable):
    ...

@BaseCallback.register("log")
class LoggingCallback(BaseCallback):
    ...

@BaseCallback.register("notify")
class NotificationCallback(BaseCallback):
    ...

@workflow.step
def train_model(..., callbacks: list[BaseCallback]):
    ...
```

In the configuration, registered types can be referenced by their registration names:

```jsonnet
{
  steps: {
    model: {
      type: 'train_model',
      ...,
      callbacks: [
        { type: 'log', ... },      // LoggingCallback
        { type: 'notify', ... }    // NotificationCallback
      ]
    }
  }
}
```

#### Direct Module References

Use the `type: 'path.to:ClassName'` notation to reference any class or function from any module:

```python
@workflow.step
def train_sklearn(model, X, y):
    ...
```

```jsonnet
{
  steps: {
    model: {
      type: 'train_sklearn',
      model: {
        type: 'sklearn.ensemble:RandomForestClassifier',
        n_estimators: 100,
      },
      ...
    }
  }
}
```

**Syntax:** `'module.path:ClassName'` or `'module.path:function_name'`

This allows you to use any Python object without pre-registration.

### JSON Schema Support

The `formed workflow schema` command generates a JSON Schema for your workflow configuration:

```bash
formed workflow schema --output schema.json
```

Use this schema with JSON Schema-compatible LSP (Language Server Protocol) implementations for:
- Auto-completion
- Type validation
- Documentation on hover

Reference the schema in your configuration file:

```json
{
  "$schema": "./schema.json",
  "steps": {
    ...
  }
}
```

This enables IDE support for efficient configuration writing with validation and auto-completion.
