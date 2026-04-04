"""Сохраняет debug-изображения для тестов и ручной диагностики."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from vision_core.entities.page import Page
from vision_core.utils.drawer import Drawer


class DebugImageObserver:
    """Сохраняет debug-изображения для тестов и ручной диагностики.

    Поддерживает два режима:
    - side-by-side сравнение исходного и обработанного изображения;
    - одно изображение с детектируемыми рамками и подписями.

    Каждый этап pipeline пишет изображения в отдельную поддиректорию.
    Файлы именуются по номеру страницы: page_000.png, page_001.png и т.д.

    Attributes:
        output_dir: Корневая директория для debug-изображений.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Инициализирует observer с корневой директорией вывода.

        Args:
            output_dir: Путь к каталогу, куда сохраняются debug-изображения.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_debug_image(
        self,
        source_image: np.ndarray,
        debug_image: np.ndarray,
        *,
        stage: str,
        prefix: str = "",
        page_number: int,
    ) -> None:
        """Сохраняет пару изображений для debug-этапа pipeline.

        Args:
            source_image: Исходное изображение этапа.
            debug_image: Результат этапа, который нужно визуализировать.
            stage: Название этапа pipeline. Используется как имя поддиректории.
            prefix: Префикс для имени файла.
            page_number: Номер страницы в документе.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        Drawer(source_image, side_by_side=True).draw_side_by_side_image(debug_image).save(output_path)
        logger.debug(f"Debug image saved: {stage} -> {output_path}")

    def on_detected_boxes(
        self,
        image: np.ndarray,
        *,
        boxes: list[tuple[int, int, int, int]],
        stage: str,
        prefix: str = "",
        page_number: int,
        color: str | tuple[int, int, int] = "blue",
        width: int = 2,
        fill: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    ) -> None:
        """Сохраняет одно изображение с набором детектируемых рамок.

        Args:
            image: Исходное изображение, на котором рисуются рамки.
            boxes: Список bbox в формате (x1, y1, x2, y2).
            stage: Название debug-этапа.
            prefix: Префикс для имени файла.
            page_number: Номер страницы.
            color: Цвет рамки.
            width: Толщина рамки.
            fill: Цвет заливки. Для полупрозрачной заливки передавайте RGBA.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        Drawer(image).draw_boxes(boxes, color=color, width=width, fill=fill).save(output_path)
        logger.debug(f"Detected boxes saved: {stage} -> {output_path}")

    def on_labeled_boxes(
        self,
        image: np.ndarray,
        *,
        items: list[tuple[tuple[int, int, int, int], str]],
        stage: str,
        prefix: str = "",
        page_number: int,
        color: str | tuple[int, int, int] = "blue",
        width: int = 2,
        fill: tuple[int, int, int] | tuple[int, int, int, int] | None = None,
    ) -> None:
        """Сохраняет одно изображение с рамками и подписями.

        Args:
            image: Исходное изображение, на котором рисуются рамки.
            items: Список пар (bbox, label).
            stage: Название debug-этапа.
            prefix: Префикс для имени файла.
            page_number: Номер страницы.
            color: Цвет рамки и подписи.
            width: Толщина рамки.
            fill: Цвет заливки. Для полупрозрачной заливки передавайте RGBA.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        Drawer(image).draw_labeled_boxes(items, color=color, width=width, fill=fill).save(output_path)
        logger.debug(f"Labeled boxes saved: {stage} -> {output_path}")

    def on_canonical_page(
        self,
        image: np.ndarray,
        *,
        page: Page,
        stage: str,
        prefix: str = "",
        page_number: int | None = None,
    ) -> None:
        """Сохраняет визуализацию канонической страницы документа.

        Отрисовывает только области text blobs из канонической структуры `Page`.
        Blobs параграфов подсвечиваются полупрозрачным оранжевым фоном, а blobs
        ячеек таблиц — полупрозрачным синим. Рамки структур и распознанный текст
        поверх изображения не рисуются.

        Args:
            image: Исходное изображение страницы.
            page: Каноническая страница документа.
            stage: Название debug-этапа.
            prefix: Префикс для имени файла.
            page_number: Явный номер страницы для имени файла. Если не задан,
                используется `page.page_number`.
        """
        resolved_page_number = page.page_number if page_number is None else page_number
        output_path = self._build_output_path(
            stage=stage,
            prefix=prefix,
            page_number=resolved_page_number,
        )
        drawer = Drawer(image, side_by_side=True)

        for paragraph in page.paragraphs:
            if paragraph.is_empty or not paragraph.blobs:
                continue
            drawer.draw_boxes(
                [blob.to_tuple() for blob in paragraph.blobs],
                color="darkorange",
                width=0,
                fill=(255, 165, 0, 96),
                draw_on="left",
            )

        for table in page.tables:
            for cell in table.cells:
                if not cell.blobs:
                    continue
                drawer.draw_boxes(
                    [blob.to_tuple() for blob in cell.blobs],
                    color="blue",
                    width=0,
                    fill=(30, 144, 255, 96),
                    draw_on="left",
                )

        for table in page.tables:
            drawer.draw_table_structure(
                table.bbox.to_tuple(),
                cells=table.cells,
                label=table.id,
                color="black",
                cell_color="black",
                draw_on="right",
            )

        drawer.save(output_path)
        logger.debug(f"Canonical page saved: {stage} -> {output_path}")

    def _build_output_path(self, *, stage: str, prefix: str, page_number: int) -> Path:
        stage_dir = self.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._build_file_name(prefix=prefix, page_number=page_number)
        return stage_dir / file_name

    def _build_file_name(self, *, prefix: str, page_number: int) -> str:
        normalized_prefix = prefix.strip("_")
        if normalized_prefix:
            return f"{normalized_prefix}_{page_number:03d}.png"
        return f"page_{page_number:03d}.png"
