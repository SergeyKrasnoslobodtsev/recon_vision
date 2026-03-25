"""Определяет DTO сценария чтения статуса процесса."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.entities.process import ProcessState


@dataclass(slots=True, frozen=True)
class GetProcessStatusCommand:
    """Хранит команду получения статуса процесса."""

    process_id: str


@dataclass(slots=True, frozen=True)
class GetProcessStatusResult:
    """Хранит результат чтения состояния процесса."""

    process_state: ProcessState
