# 🧬 Formed

[![CI](https://github.com/altescy/formed/actions/workflows/ci.yml/badge.svg)](https://github.com/altescy/formed/actions/workflows/ci.yml)
[![Docs](https://github.com/altescy/formed/actions/workflows/docs.yml/badge.svg)](https://altescy.jp/formed/)
[![Python version](https://img.shields.io/pypi/pyversions/formed)](https://github.com/altescy/formed)
[![License](https://img.shields.io/github/license/altescy/formed)](https://github.com/altescy/formed/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/formed)](https://pypi.org/project/formed/)

Formed is a flexible framework for managing data, experiments, and workflows in both research and production environments. It provides a simple yet powerful DAG-based workflow system with automatic caching, dependency tracking, and seamless integration with popular ML tools.

## Key Features

- **📊 DAG-based workflows**: Define complex workflows as directed acyclic graphs with automatic dependency resolution
- **💾 Smart caching**: Content-based automatic caching that detects code changes via AST analysis
- **🔧 Flexible configuration**: Use Jsonnet/JSON for declarative workflow definitions with type safety
- **🔌 Rich integrations**: Built-in support for PyTorch, 🤗 Transformers, MLflow, and more
- **🎯 Type-safe**: Leverage Python type hints for automatic object construction and validation
- **📦 Extensible**: Easy to extend with custom steps, formats, and organizers

## Quick Example

Define reusable computation steps with automatic caching:

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

Connect steps in a workflow configuration:

```jsonnet
// workflow.jsonnet
{
  steps: {
    dataset: {
      type: 'load_dataset',
      size: 10
    },
    results: {
      type: 'square',
      dataset: { type: 'ref', ref: 'dataset' }
    }
  }
}
```

Configure and run:

```yaml
# formed.yml
workflow:
  organizer:
    type: filesystem

required_modules:
  - mysteps
```

```shell
formed workflow run workflow.jsonnet --execution-id my-experiment
```

Results are automatically cached - rerunning only executes changed steps!

## Installation

```shell
pip install formed
```

With integrations:

```shell
pip install formed[mlflow]        # MLflow integration
pip install formed[torch]          # PyTorch integration
pip install formed[transformers]   # 🤗 Transformers integration
pip install formed[all]            # All integrations
```

## Documentation

📖 **Full documentation available [here](https://altescy.jp/formed)**

- [Quick Start](https://altescy.jp/formed/quick_start/) - Get started in minutes
- [Tutorials](https://altescy.jp/formed/tutorials/) - Practical examples and use cases
- [Guides](https://altescy.jp/formed/guides/) - Deep dives into concepts and features
- [API Reference](https://altescy.jp/formed/reference/) - Complete API documentation

## Why Formed?

Formed bridges the gap between experimental notebooks and production pipelines:

- **Reproducible**: Content-based caching ensures consistent results
- **Iterative**: Only re-execute what changed, speeding up development
- **Collaborative**: Declarative configs make workflows easy to share and review
- **Production-ready**: Same code works in research and deployment

Whether you're prototyping in Jupyter or deploying at scale, formed adapts to your workflow.

## License

MIT License - see [LICENSE](LICENSE) file for details.
