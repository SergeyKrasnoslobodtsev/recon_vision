"""Debug-реализация PipelineObserver -- сохраняет side-by-side визуализации."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from vision_core.observer import PipelineObserver
from vision_core.utils.drawer import Drawer


class DebugObserver(PipelineObserver):
    """Сохраняет промежуточные результаты pipeline в виде изображений.

    Каждый этап (stage) сохраняется в свою поддиректорию.
    Файлы именуются по номеру страницы: page_000.png, page_001.png и т.д.

    Attributes:
        output_dir: Корневая директория для debug-изображений.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Инициализирует observer с указанием директории для вывода.

        Args:
            output_dir: Путь к корневой директории для debug-изображений.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_side_by_side(
        self,
        original: np.ndarray,
        result: np.ndarray,
        *,
        stage: str,
        prefix: str = "",
        page_number: int,
    ) -> None:
        """Сохраняет side-by-side сравнение двух изображений.

        Args:
            original: Изображение до обработки.
            result: Изображение после обработки.
            stage: Название этапа -- используется как имя поддиректории.
            prefix: Префикс для имени файла (по умолчанию пустая строка).
            page_number: Номер страницы в документе.
        """
        stage_dir = self.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        drawer = Drawer(original, side_by_side=True)
        drawer.draw_processed(result)

        path = stage_dir / f"{prefix}_{page_number:03d}.png"
        drawer.save(path)
        logger.debug(f"Debug: {stage} side-by-side -> {path}")
