from dataclasses import dataclass

from loguru import logger

from extractor.normalize import transform_text
from extractor.tokenize import AnyReference, tokenize


@dataclass
class Result:
    """Текст и извлечённые токены."""

    text: str
    tokens: list[AnyReference]


def extract(text: str) -> Result:
    """извлекает токены."""
    logger.debug(f"Текст до нормализации: {text}")
    text = transform_text(text)
    logger.debug(f"Текст после нормализации: {text}")
    tokens = tokenize(text)
    logger.debug(f"Извлечённые токены: {tokens}")
    return Result(text, tokens)
