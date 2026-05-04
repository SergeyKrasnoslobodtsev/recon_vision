"""Реализует хранение процессов на базе DiskCache."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from typing import Any

from diskcache import Cache
from loguru import logger

from app.domain.entities.ledger_entry import LedgerEntry, RowReference
from app.domain.entities.process import ProcessState
from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.enums.process_status import ProcessStatus
from app.domain.value_objects.period import Period
from app.infrastructure.persistence.models import ProcessData


class DiskCacheProcessRepository:
    """Хранит данные процессов в DiskCache с TTL.

    Args:
        expire: Время жизни записи в секундах.
        cache_dir: Путь до директории кэша.
    """

    def __init__(self, expire: int = 24 * 60 * 60, cache_dir: str = "./.cache"):
        self.cache = Cache(cache_dir)
        self.expire = expire

    async def add(self, process_state: ProcessState) -> str:
        """Сохраняет новый процесс в хранилище.

        Args:
            process_state: Состояние нового процесса.

        Returns:
            str: Идентификатор сохранённого процесса.
        """
        process_id = process_state.process_id or str(uuid.uuid4())
        process_state.process_id = process_id
        payload = self._to_process_data(process_state)
        self.cache.set(process_id, payload, expire=self.expire)
        logger.info(f"Создан процесс {process_id}")
        return process_id

    async def get(self, process_id: str) -> ProcessState | None:
        """Возвращает состояние процесса по идентификатору.

        Args:
            process_id: Идентификатор процесса.

        Returns:
            ProcessState | None: Восстановленное состояние или None.
        """
        data = self.cache.get(process_id)
        if not isinstance(data, ProcessData):
            return None

        return self._to_process_state(process_id, data)

    async def update(self, process_state: ProcessState) -> None:
        """Обновляет состояние сохранённого процесса.

        Args:
            process_state: Новое состояние процесса.

        Raises:
            ValueError: Если у процесса не задан идентификатор.
        """
        if not process_state.process_id:
            raise ValueError("Нельзя обновить процесс без process_id")

        payload = self._to_process_data(process_state)
        self.cache.set(process_state.process_id, payload, expire=self.expire)

    async def delete(self, process_id: str) -> None:
        """Удаляет процесс из хранилища.

        Args:
            process_id: Идентификатор процесса.
        """
        deleted = self.cache.delete(process_id)
        if deleted:
            logger.info(f"Удален процесс {process_id}")

    async def cleanup_expired_cache(self) -> None:
        """Периодически очищает истёкшие записи из кэша."""
        while True:
            await asyncio.sleep(3600)
            logger.info("Очистка устаревших записей кэша...")
            self.cache.expire()
            logger.debug(f"Очистка завершена. Размер кэша: {len(self.cache)}")

    def close(self) -> None:
        """Закрывает хранилище кэша."""
        self.cache.close()

    def _to_process_data(self, process_state: ProcessState) -> ProcessData:
        """Преобразует доменное состояние в модель хранения.

        Args:
            process_state: Состояние процесса.

        Returns:
            ProcessData: Объект для сохранения в кэше.
        """
        structure = None
        if process_state.reconciliation_data is not None:
            structure = self._serialize_reconciliation_data(
                process_state.reconciliation_data,
                process_id=process_state.process_id,
                status=process_state.status,
                message=process_state.message,
            )

        return ProcessData(
            process_state=process_state,
            pdf_bytes=process_state.source_pdf,
            structure=structure,
            created_at=process_state.created_at,
        )

    def _to_process_state(self, process_id: str, data: ProcessData) -> ProcessState:
        """Преобразует запись кэша в доменное состояние.

        Args:
            process_id: Идентификатор процесса.
            data: Сохранённая запись процесса.

        Returns:
            ProcessState: Восстановленное состояние процесса.
        """
        if data.process_state is not None:
            return data.process_state

        reconciliation_data = None
        if data.structure is not None:
            reconciliation_data = self._deserialize_reconciliation_data(data.structure)

        status_value = data.structure.get("status") if data.structure else None
        status = (
            ProcessStatus(status_value)
            if status_value is not None and status_value in ProcessStatus._value2member_map_
            else ProcessStatus.COMPLETED
        )

        return ProcessState(
            process_id=process_id,
            status=status,
            source_pdf=data.pdf_bytes or b"",
            reconciliation_data=reconciliation_data,
            message=(data.structure or {}).get("message", ""),
            created_at=data.created_at,
            updated_at=data.created_at,
        )

    def _serialize_reconciliation_data(
        self,
        reconciliation_data: ReconciliationData,
        process_id: str | None,
        status: ProcessStatus,
        message: str,
    ) -> dict[str, Any]:
        """Сериализует доменные данные акта сверки в словарь.

        Args:
            reconciliation_data: Данные акта сверки.
            process_id: Идентификатор процесса.
            status: Статус процесса.
            message: Сообщение состояния.

        Returns:
            dict[str, Any]: Сериализованный словарь данных.
        """
        payload = asdict(reconciliation_data)
        payload["process_id"] = process_id or ""
        payload["status"] = int(status)
        payload["message"] = message or reconciliation_data.message
        return payload

    def _deserialize_reconciliation_data(
        self,
        structure: dict[str, Any],
    ) -> ReconciliationData:
        """Восстанавливает доменные данные акта сверки из словаря.

        Args:
            structure: Сериализованный словарь данных.

        Returns:
            ReconciliationData: Восстановленная доменная модель.
        """
        period_raw = structure.get("period") or {}
        debit_raw = structure.get("debit") or structure.get("entries") or []
        credit_raw = structure.get("credit") or []

        return ReconciliationData(
            seller=structure.get("seller", ""),
            buyer=structure.get("buyer", ""),
            period=Period(
                start=period_raw.get("start"),
                end=period_raw.get("end"),
            ),
            debit=[self._deserialize_entry(item) for item in debit_raw],
            credit=[self._deserialize_entry(item) for item in credit_raw],
            message=structure.get("message", ""),
        )

    def _deserialize_entry(self, payload: dict[str, Any]) -> LedgerEntry:
        """Восстанавливает доменную запись из словаря.

        Args:
            payload: Сериализованные данные записи.

        Returns:
            LedgerEntry: Восстановленная доменная запись.
        """
        row_reference = None
        row_id = payload.get("row_id") or {}
        if row_id:
            row_reference = RowReference(
                id_table=str(row_id.get("id_table") or row_id.get("num_table") or ""),
                id_row=str(row_id.get("id_row") or row_id.get("num_row") or ""),
                id_col=(
                    int(row_id.get("id_col"))
                    if row_id.get("id_col") is not None
                    else int(row_id.get("num_col"))
                    if row_id.get("num_col") is not None
                    else None
                ),
            )

        value = payload.get("value")
        if value is None:
            value = payload.get("debit") or payload.get("credit") or 0.0

        return LedgerEntry(
            record=payload.get("record") or payload.get("description", ""),
            value=float(value),
            date=payload.get("date") or payload.get("document_date"),
            row_reference=row_reference,
        )
