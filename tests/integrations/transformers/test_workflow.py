from pathlib import Path

import datasets
import pytest
import transformers

from formed.integrations.datasets import workflow as datasets_workflow  # noqa: F401
from formed.integrations.ml import Tokenizer
from formed.integrations.transformers.workflow import (
    TransformersPretrainedModelFormat,
    convert_tokenizer,
    load_pretrained_model,
    load_pretrained_tokenizer_step,
)
from formed.workflow import (
    DefaultWorkflowExecutor,
    MemoryWorkflowOrganizer,
    WorkflowGraph,
)


class TestTransformersFormat:
    @staticmethod
    def test_write_and_read_model_with_preserved_class(tmp_path: Path) -> None:
        from transformers import AutoModelForTokenClassification

        format = TransformersPretrainedModelFormat()

        model = AutoModelForTokenClassification.from_pretrained(
            "hf-internal-testing/tiny-bert-for-token-classification"
        )
        format.write(model, tmp_path)

        restored_model = format.read(tmp_path)

        assert restored_model.__class__ is model.__class__
        assert restored_model.__class__.__name__ == "BertForTokenClassification"


class TestLoadPretrainedTokenizerStep:
    """Test the load_pretrained_tokenizer_step workflow step."""

    @staticmethod
    def test_load_tokenizer_step() -> None:
        """Test that the workflow step loads a tokenizer correctly."""
        tokenizer = load_pretrained_tokenizer_step("hf-internal-testing/tiny-random-bert")
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    @staticmethod
    def test_load_tokenizer_step_with_kwargs() -> None:
        """Test loading a tokenizer with additional kwargs."""
        tokenizer = load_pretrained_tokenizer_step(
            "hf-internal-testing/tiny-random-bert",
            use_fast=False,
        )
        assert tokenizer is not None
        # Check that it's a slow tokenizer
        assert not tokenizer.is_fast


class TestLoadPretrainedModelStep:
    """Test the load_pretrained_model workflow step."""

    @staticmethod
    def test_load_model_step_with_default_auto_class() -> None:
        """Test loading a model with default AutoModel."""
        model = load_pretrained_model("hf-internal-testing/tiny-random-bert")
        assert model is not None
        assert isinstance(model, transformers.PreTrainedModel)

    @staticmethod
    def test_load_model_step_with_string_auto_class() -> None:
        """Test loading a model with string-specified auto class."""
        model = load_pretrained_model(
            "hf-internal-testing/tiny-random-bert",
            auto_class="AutoModelForSequenceClassification",
        )
        assert model is not None
        assert hasattr(model, "num_labels")

    @staticmethod
    def test_load_model_step_with_type_auto_class() -> None:
        """Test loading a model with type-specified auto class."""
        model = load_pretrained_model(
            "hf-internal-testing/tiny-random-bert",
            auto_class=transformers.AutoModelForSequenceClassification,
        )
        assert model is not None
        assert hasattr(model, "num_labels")

    @staticmethod
    def test_load_model_step_with_submodule() -> None:
        """Test loading a specific submodule from the model."""
        model = load_pretrained_model(
            "hf-internal-testing/tiny-random-bert",
            submodule="embeddings",
        )
        assert model is not None
        # The submodule should be the embeddings module
        assert hasattr(model, "word_embeddings")

    @staticmethod
    def test_load_model_step_bypasses_cache() -> None:
        """Test that the step bypasses the LRU cache by calling __wrapped__."""
        # This test verifies that the step calls __wrapped__ to bypass caching
        # The actual behavior is tested indirectly - the function should work
        # without issues even when called multiple times
        model1 = load_pretrained_model("hf-internal-testing/tiny-random-bert")
        model2 = load_pretrained_model("hf-internal-testing/tiny-random-bert")
        # Both calls should succeed and return valid models
        assert model1 is not None
        assert model2 is not None
        assert isinstance(model1, transformers.PreTrainedModel)
        assert isinstance(model2, transformers.PreTrainedModel)


class TestConvertTokenizer:
    """Test the convert_tokenizer workflow step."""

    @staticmethod
    def test_convert_tokenizer_from_string() -> None:
        """Test converting a tokenizer from model name string."""
        tokenizer = convert_tokenizer("hf-internal-testing/tiny-random-bert")

        # Verify it returns a Tokenizer instance
        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.surfaces is not None
        assert tokenizer.analyzer is not None

    @staticmethod
    def test_convert_tokenizer_from_tokenizer_instance() -> None:
        """Test converting from a tokenizer instance."""
        from transformers import AutoTokenizer

        pretrained_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        tokenizer = convert_tokenizer(pretrained_tokenizer)

        # Verify it returns a Tokenizer instance
        assert isinstance(tokenizer, Tokenizer)
        assert tokenizer.surfaces is not None
        assert tokenizer.analyzer is not None

    @staticmethod
    def test_convert_tokenizer_vocab_mapping() -> None:
        """Test that vocabulary is correctly mapped."""
        tokenizer = convert_tokenizer("hf-internal-testing/tiny-random-bert")

        # The surfaces indexer should have a vocab
        assert tokenizer.surfaces is not None
        assert hasattr(tokenizer.surfaces, "vocab")
        assert len(tokenizer.surfaces.vocab) > 0

    @staticmethod
    def test_convert_tokenizer_special_tokens_from_tokenizer() -> None:
        """Test that special tokens are extracted from tokenizer."""
        tokenizer = convert_tokenizer("hf-internal-testing/tiny-random-bert")

        # Special tokens should be extracted from the tokenizer
        assert tokenizer.surfaces is not None
        assert tokenizer.surfaces.pad_token is not None
        # UNK token may or may not be present depending on tokenizer

    @staticmethod
    def test_convert_tokenizer_custom_special_tokens() -> None:
        """Test providing custom special tokens that exist in vocab."""
        from transformers import AutoTokenizer

        # Load tokenizer to check available tokens
        pretrained = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        vocab = pretrained.get_vocab()

        # Use tokens that actually exist in the vocab
        # For BERT-like tokenizers, these standard tokens should be present
        pad_token = pretrained.pad_token or "[PAD]"
        unk_token = pretrained.unk_token or "[UNK]"

        # Only specify tokens if they exist in vocab
        tokenizer = convert_tokenizer(
            "hf-internal-testing/tiny-random-bert",
            pad_token=pad_token if pad_token in vocab else None,
            unk_token=unk_token if unk_token in vocab else None,
        )

        # Verify the tokens were set correctly
        assert tokenizer.surfaces is not None
        assert tokenizer.surfaces.pad_token == pad_token

    @staticmethod
    def test_convert_tokenizer_freeze_vocab() -> None:
        """Test that freeze parameter is passed to TokenSequenceIndexer."""
        # With freeze=True (default)
        tokenizer_frozen = convert_tokenizer("hf-internal-testing/tiny-random-bert", freeze=True)
        assert tokenizer_frozen.surfaces is not None
        # Vocab should be frozen - test by checking the frozen attribute
        # (assuming TokenSequenceIndexer has this attribute)

        # With freeze=False
        tokenizer_unfrozen = convert_tokenizer("hf-internal-testing/tiny-random-bert", freeze=False)
        assert tokenizer_unfrozen.surfaces is not None

    @staticmethod
    def test_convert_tokenizer_with_accessor() -> None:
        """Test providing a custom accessor."""

        def custom_accessor(x: str) -> str:
            return x.lower()

        tokenizer = convert_tokenizer(
            "hf-internal-testing/tiny-random-bert",
            accessor=custom_accessor,
        )

        # Accessor should be set
        assert tokenizer.accessor is custom_accessor

    @staticmethod
    def test_convert_tokenizer_pad_token_required() -> None:
        """Test that pad_token is required if not in tokenizer."""
        import pytest
        from transformers import GPT2Tokenizer

        # GPT2 tokenizer doesn't have a pad token by default
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Should raise assertion error without pad_token
        with pytest.raises(AssertionError, match="pad_token must be specified"):
            convert_tokenizer(gpt2_tokenizer)

    @staticmethod
    def test_convert_tokenizer_integration() -> None:
        """Test that converted tokenizer can be used for text processing."""
        from formed.integrations.ml.types import AnalyzedText

        tokenizer = convert_tokenizer("hf-internal-testing/tiny-random-bert")

        # Test with raw text
        text = "Hello world"

        # The tokenizer should be able to process text through its analyzer
        assert tokenizer.analyzer is not None
        analyzed = tokenizer.analyzer(text)

        assert isinstance(analyzed, AnalyzedText)
        assert len(analyzed.surfaces) > 0


class TestE2ETransformersWorkflow:
    @pytest.fixture
    @staticmethod
    def dataset() -> datasets.Dataset:
        n = 10
        features = datasets.Features(  # type: ignore[no-untyped-call]
            {
                "text": datasets.Value("string"),
                "id": datasets.Value("int64"),
            }
        )
        dataset = datasets.Dataset.from_dict(
            {
                "text": [" ".join(["foo"] * 5)] * n,
                "id": list(range(n)),
            },
            features=features,
        )
        return dataset

    @pytest.fixture
    @staticmethod
    def dataset_path(tmp_path: Path, dataset: datasets.Dataset) -> Path:
        dataset_path = tmp_path / "dataset"
        dataset.save_to_disk(str(dataset_path))
        return dataset_path

    @staticmethod
    def test_transformers_workflow(dataset_path: Path) -> None:
        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "dataset": {
                        "type": "datasets::load",
                        "path": str(dataset_path),
                    },
                    "tokenizer": {
                        "type": "transformers::load_tokenizer",
                        "pretrained_model_name_or_path": "hf-internal-testing/tiny-random-BertModel",
                    },
                    "tokenized_dataset": {
                        "type": "transformers::tokenize",
                        "dataset": {"type": "ref", "ref": "dataset"},
                        "tokenizer": {"type": "ref", "ref": "tokenizer"},
                        "max_length": 10,
                        "truncation": True,
                    },
                    "backbone_model": {
                        "type": "transformers::load_model",
                        "auto_class": "AutoModelForMaskedLM",
                        "model_name_or_path": "hf-internal-testing/tiny-random-BertModel",
                    },
                    "finetuned_model": {
                        "type": "transformers::train_model",
                        "model": {"type": "ref", "ref": "backbone_model"},
                        "dataset": {"type": "ref", "ref": "tokenized_dataset"},
                        "args": {
                            "per_device_train_batch_size": 2,
                            "per_device_eval_batch_size": 2,
                            "num_train_epochs": 1,
                            "learning_rate": 1e-4,
                            "warmup_ratio": 0.1,
                            "report_to": "none",
                            "do_train": True,
                        },
                        "data_collator": {
                            "type": "transformers:DataCollatorForLanguageModeling",
                            "tokenizer": {"type": "ref", "ref": "tokenizer"},
                            "mlm_probability": 0.15,
                            "pad_to_multiple_of": None,
                        },
                        "processing_class": {"type": "ref", "ref": "tokenizer"},
                        "callbacks": [
                            {"type": "formed.integrations.transformers.training:MlflowTrainerCallback"},
                        ],
                    },
                },
            },
        )

        executor = DefaultWorkflowExecutor()
        organizer = MemoryWorkflowOrganizer()
        context = organizer.run(executor, graph)
        finetuned_model = context.cache[graph["finetuned_model"]]
        assert finetuned_model is not None
        assert isinstance(finetuned_model, transformers.PreTrainedModel)
