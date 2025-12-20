from typing import Any, Generic, TypeVar

import formed.integrations.ml as ml
import formed.integrations.ml.types as mlt

from .types import InputT, TextTransformT_co


@ml.DataModule.register("textclf::text_classification")
class TextClassificationDataModule(
    ml.DataModule[
        mlt.DataModuleModeT,
        InputT,
        "TextClassificationDataModule[mlt.AsInstance, InputT, TextTransformT_co]",
        "TextClassificationDataModule[mlt.AsBatch, InputT, TextTransformT_co]",
    ],
    Generic[mlt.DataModuleModeT, InputT, TextTransformT_co],
):
    id: ml.MetadataTransform[Any, str]
    text: TextTransformT_co
    label: ml.Extra[ml.LabelIndexer] = ml.Extra.default()
