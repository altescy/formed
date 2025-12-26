# 🧬 Formed

[![CI](https://github.com/altescy/formed/actions/workflows/ci.yml/badge.svg)](https://github.com/altescy/formed/actions/workflows/ci.yml)
[![Python version](https://img.shields.io/pypi/pyversions/formed)](https://github.com/altescy/formed)
[![License](https://img.shields.io/github/license/altescy/formed)](https://github.com/altescy/formed/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/formed)](https://pypi.org/project/formed/)

Formed is a framework for managing data/experiments/workflows in both research
and production environments. It is designed to be flexible and extensible, and
to provide a simple and consistent interface for managing data and workflows.

## Features

- **DAG-based workflow system**: Define complex workflows as directed acyclic
  graphs (DAGs) to manage dependencies and execution order.
- **Experiment tracking**: Keep track of experiments, parameters, and results
  in a structured manner.
- **Built-in integration with popular data science libraries**: Seamlessly work
  with libraries like PyTorch, 🤗 Transformers, MLflow, and more.
- **Extensible architecture**: Easily extend the framework with custom components
  and plugins.

## Basic Usage

Define steps in a workflow using the `@workflow.step` decorator with type hints:

```python
# mysteps.py

from collections.abc import Iterator
from formed import workflow

@workflow.step
def load_dataset(size: int) -> Iterator[int]:
    for i in range(size):
        yield i

@workflow.step
def square(dataset: Iterator[int]) -> Iterator[int]:
    for i in dataset:
        yield i * i
```

Create a workflow definition using Jsonnet:

```jsonnet
# workflow.jsonnet

{
  steps: {
    dataset: {
      type: 'load_dataset',
      size: 10
    },
    results: {
      type: 'square',
      dataset: { type: 'ref', ref: 'dataset' } // reference to dataset
    }
  }
}
```

Setup the project configuration in `formed.yml`:

```yaml
# formed.yml

workflow:
  organizer:
    type: filesystem  # store execution results in the filesystem

required_modules:
  - mysteps           # include the custom steps module
```

Run the workflow via the command line interface:

```shell
formed run workflow.jsonnet --execution-id my-experiment
```

## Documentation Guide

- [Quick Start](quick_start.md)
- [API Reference](api_reference/index.md)
