from dataclasses import dataclass

from extractor.normalize import transform_text
from extractor.tokenize import AnyReference, tokenize


@dataclass
class Result:
    """Текст и извлечённые токены."""

    text: str
    tokens: list[AnyReference]


def extract(text: str) -> Result:
    """извлекает токены."""
    text = transform_text(text)
    tokens = tokenize(text)
    return Result(text, tokens)
