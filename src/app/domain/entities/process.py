"""Определяет доменную сущность состояния процесса."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.enums.process_status import ProcessStatus


@dataclass(slots=True)
class ProcessState:
    """Хранит состояние процесса обработки документа.

    Attributes:
        process_id: Идентификатор процесса.
        status: Текущий статус обработки.
        source_pdf: Исходный PDF в байтах.
        document_payload: Каноническое представление документа или его сериализация.
        reconciliation_data: Извлечённые бизнес-данные.
        message: Служебное сообщение состояния.
        metadata: Дополнительные служебные данные.
        created_at: Время создания процесса.
        updated_at: Время последнего изменения.
    """

    process_id: str | None = None
    status: ProcessStatus = ProcessStatus.RECEIVED
    source_pdf: bytes = b""
    document_payload: Any | None = None
    reconciliation_data: ReconciliationData | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def mark_processing(self, message: str = "") -> None:
        """Переводит процесс в статус обработки.

        Args:
            message: Дополнительное сообщение состояния.
        """
        self.status = ProcessStatus.PROCESSING
        self.message = message
        self.updated_at = datetime.now(UTC)

    def mark_completed(
        self,
        reconciliation_data: ReconciliationData,
        message: str = "",
    ) -> None:
        """Переводит процесс в статус завершения.

        Args:
            reconciliation_data: Извлечённые данные акта сверки.
            message: Дополнительное сообщение состояния.
        """
        self.status = ProcessStatus.COMPLETED
        self.reconciliation_data = reconciliation_data
        self.message = message
        self.updated_at = datetime.now(UTC)

    def mark_failed(self, message: str) -> None:
        """Переводит процесс в статус ошибки.

        Args:
            message: Описание ошибки.
        """
        self.status = ProcessStatus.FAILED
        self.message = message
        self.updated_at = datetime.now(UTC)

    def mark_filled(self, message: str = "") -> None:
        """Переводит процесс в статус заполненного документа.

        Args:
            message: Дополнительное сообщение состояния.
        """
        self.status = ProcessStatus.FILLED
        self.message = message
        self.updated_at = datetime.now(UTC)
