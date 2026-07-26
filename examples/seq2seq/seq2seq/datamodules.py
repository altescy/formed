from typing import Any

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt

from .types import Seq2SeqExample


@ml.DataModule.register("seq2seq::datamodule")
class Seq2SeqDataModule(
    ml.DataModule[
        mlt.DataModuleModeT,
        Seq2SeqExample,
        "Seq2SeqDataModule[mlt.AsInstance]",
        "Seq2SeqDataModule[mlt.AsBatch]",
    ]
):
    source: ml.TextIndexer[Any]
    target: ml.TextIndexer[Any]
    id: ml.Extra[ml.MetadataTransform[Any, str]] = ml.Extra.default()
