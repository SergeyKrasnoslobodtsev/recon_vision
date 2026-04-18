from dataclasses import dataclass

from extractor.normalize import transform_text
from extractor.tokenize import AnyReference, tokenize


@dataclass
class Result:
    """Нормализованный текст и извлечённые токены."""

    text: str
    tokens: list[AnyReference]


def extract(text: str) -> Result:
    """Нормализует текст и извлекает токены."""
    normalized = transform_text(text)
    tokens = tokenize(normalized)
    return Result(normalized, tokens)
