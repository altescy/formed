from dataclasses import dataclass


@dataclass
class Seq2SeqExample:
    id: str
    source: str
    target: str
