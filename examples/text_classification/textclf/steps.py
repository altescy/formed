import random
from collections.abc import Sequence

from formed import workflow

from .types import ClassificationExample


@workflow.step("textclf::generate_sort_detection_dataset")
def generate_sort_detection_dataset(
    vocab: Sequence[str] = "abcdefghijklmnopqrstuvwxyz",
    num_examples: int = 100,
    max_tokens: int = 10,
    random_seed: int = 42,
) -> list[ClassificationExample]:
    rng = random.Random(random_seed)
    examples = []
    for _ in range(num_examples):
        num_tokens = rng.randint(1, max_tokens)
        label = rng.choice(["sorted", "not_sorted"])
        tokens = rng.choices(vocab, k=num_tokens)
        if label == "sorted":
            tokens.sort()
        examples.append(ClassificationExample(id=str(len(examples)), text=tokens, label=label))
    return examples
