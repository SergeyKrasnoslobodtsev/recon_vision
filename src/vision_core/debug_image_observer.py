"""Сохраняет debug-изображения для тестов и ручной диагностики."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from vision_core.entities.page import Page
from vision_core.entities.paragraph import ParagraphType
from vision_core.entities.table import Table
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
        src_image: np.ndarray,
        stage: str,
        prefix: str = "",
        page_number: int = 0,
    ) -> None:
        """Сохраняет debug-изображение для текущего этапа.

        Args:
            src_image: Исходное изображение, которое сохраняется без изменений.
            stage: Название debug-этапа, например "table_mask" или "detected_boxes".
            prefix: Префикс для имени файла, например "preprocessed".
            page_number: Номер страницы для имени файла.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        Drawer(src_image).save(output_path)
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

    def on_ocr_boxes(
        self,
        image: np.ndarray,
        *,
        boxes: list[tuple[int, int, int, int]],
        stage: str,
        prefix: str = "",
        page_number: int,
    ) -> None:
        """Сохраняет изображение с OCR-блоками, каждый bbox своим цветом из палитры.

        Args:
            image: Исходное изображение страницы.
            boxes: Список bbox в формате (x1, y1, x2, y2).
            stage: Название debug-этапа.
            prefix: Префикс для имени файла.
            page_number: Номер страницы.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        Drawer(image).draw_boxes_cyclic(boxes).save(output_path)
        logger.debug(f"OCR boxes saved: {stage} -> {output_path}")

    def on_table_structures(
        self,
        image: np.ndarray,
        *,
        tables: list[Table],
        stage: str,
        prefix: str = "",
        page_number: int,
    ) -> None:
        """Сохраняет side-by-side изображение со структурой таблиц.

        Слева — исходное изображение страницы, справа — структура таблиц:
        внешние рамки, ячейки и распознанный текст по ячейкам.

        Args:
            image: Изображение страницы (после препроцессора).
            tables: Список таблиц с заполненными ячейками.
            stage: Название debug-этапа.
            prefix: Префикс для имени файла.
            page_number: Номер страницы.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        drawer = Drawer(image, side_by_side=True)
        for table in tables:
            drawer.draw_table_structure(
                table.bbox.to_tuple(),
                cells=table.cells,
                label=table.id,
                color="black",
                cell_color="black",
                draw_on="right",
            )
        drawer.save(output_path)
        logger.debug(f"Table structures saved: {stage} -> {output_path}")

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

    _PARAGRAPH_TYPE_COLORS: dict[ParagraphType, tuple[int, int, int, int]] = {
        ParagraphType.PAGE_HEADER: (255, 165, 0, 60),  # orange
        ParagraphType.PAGE_FOOTER: (255, 165, 0, 60),  # orange
        ParagraphType.SECTION_TITLE: (220, 20, 60, 60),  # crimson
        ParagraphType.TABLE_CAPTION: (148, 0, 211, 60),  # purple
        ParagraphType.BODY_TEXT: (30, 144, 255, 60),  # blue
        ParagraphType.UNKNOWN: (128, 128, 128, 60),  # gray
    }

    def on_document_page(
        self,
        image: np.ndarray,
        *,
        page: Page,
        table_color_map: dict[str, tuple[int, int, int, int]],
        stage: str,
        prefix: str = "",
        page_number: int,
    ) -> None:
        """Сохраняет финальную side-by-side визуализацию страницы документа.

        Слева — выровненное изображение с аннотациями:
          - таблицы раскрашены по цепочке продолжений (один цвет = одна таблица);
          - параграфы раскрашены по типу (HEADER/FOOTER/TEXT/NONE).
        Справа — реконструкция содержимого:
          - структура таблиц с текстом по ячейкам;
          - текст параграфов в соответствии с их bbox.

        Args:
            image: Выровненное изображение страницы (до препроцессора).
            page: Каноническая страница после всех постпроцессоров.
            table_color_map: Словарь {table.id: RGBA} из _build_table_color_map.
            stage: Название debug-этапа.
            prefix: Префикс для имени файла.
            page_number: Номер страницы.
        """
        output_path = self._build_output_path(stage=stage, prefix=prefix, page_number=page_number)
        drawer = Drawer(image, side_by_side=True)

        for table in page.tables:
            color = table_color_map.get(table.id, (100, 100, 255, 80))
            drawer.draw_structure(
                table.bbox.to_tuple(),
                color=color[:3],
                width=2,
                fill=color,
                draw_on="left",
            )

        for paragraph in page.paragraphs:
            color = self._PARAGRAPH_TYPE_COLORS.get(paragraph.type, (180, 180, 180, 60))
            drawer.draw_structure(
                paragraph.bbox.to_tuple(),
                color=color[:3],
                width=1,
                fill=color,
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

        for paragraph in page.paragraphs:
            if paragraph.text.strip():
                color = self._PARAGRAPH_TYPE_COLORS.get(paragraph.type, (180, 180, 180, 60))
                drawer.draw_text_in_bbox(
                    paragraph.bbox.to_tuple(),
                    text=paragraph.text,
                    fill=color[:3],
                    draw_on="right",
                )

        drawer.save(output_path)
        logger.debug(f"Document page saved: {stage} -> {output_path}")

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
