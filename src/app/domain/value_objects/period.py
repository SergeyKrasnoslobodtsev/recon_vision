"""Определяет value object периода акта сверки."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Period:
    """Описывает период акта сверки.

    Attributes:
        start: Дата начала периода.
        end: Дата окончания периода.
    """

    start: str | None = None
    end: str | None = None
