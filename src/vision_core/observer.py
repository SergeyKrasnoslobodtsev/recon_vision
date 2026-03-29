"""Протокол наблюдателя pipeline для debug-визуализации."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PipelineObserver(Protocol):
    """Наблюдатель этапов vision pipeline.

    Передаётся в PageAnalyzer опционально.
    Наличие observer -- и есть debug-режим; если observer=None -- ноль overhead.
    """

    def on_side_by_side(
        self,
        original: np.ndarray,
        result: np.ndarray,
        *,
        stage: str,
        prefix: str = "",
        page_number: int,
    ) -> None:
        """Сохраняет сравнение двух изображений side-by-side.

        Args:
            original: Изображение до обработки.
            result: Изображение после обработки.
            stage: Название этапа (preprocessing, table_detection и т.п.).
            prefix: Префикс для имени файла.
            page_number: Номер страницы в документе.
        """
        ...
