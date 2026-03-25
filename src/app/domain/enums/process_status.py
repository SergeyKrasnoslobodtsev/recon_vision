"""Определяет статусы обработки документа."""

from __future__ import annotations

from enum import IntEnum


class ProcessStatus(IntEnum):
    """Перечисляет этапы жизненного цикла процесса обработки."""

    RECEIVED = 0
    PROCESSING = 1
    COMPLETED = 2
    FAILED = 3
    FILLED = 4
