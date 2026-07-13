"""Определяет модели данных слоя персистентности."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.entities.process import ProcessState


@dataclass(slots=True)
class ProcessData:
    """Хранит данные процесса обработки документа.

    Attributes:
        process_state: Каноническое состояние процесса в новом приложении.
        pdf_bytes: Legacy-поле исходного PDF в байтах.
        structure: Legacy-поле структурированных данных.
        created_at: Время создания процесса.
    """

    process_state: ProcessState | None = None
    pdf_bytes: bytes | None = None
    structure: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
