"""Настраивает приложение на использование Loguru."""

from __future__ import annotations

import logging
import sys

from loguru import logger

from app.infrastructure.logging.intercept_handler import InterceptHandler


def configure_logging(level: int) -> None:
    """Настраивает интеграцию logging и Loguru.

    Args:
        level: Уровень логирования из модуля logging.
    """
    logging.basicConfig(handlers=[InterceptHandler(level=level)], level=level)
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])
