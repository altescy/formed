import importlib
from unittest.mock import patch
from contextlib import suppress

import pytest


@pytest.fixture(autouse=True)
def patch_sentencde_transformers():
    with suppress(ImportError):
        from sentence_transformers import SentenceTransformer

        for module_name in ("sentence_transformers.losses.MatryoshkaLoss",):
            module = importlib.import_module(module_name)
            setattr(module, "SentenceTransformer", SentenceTransformer)
