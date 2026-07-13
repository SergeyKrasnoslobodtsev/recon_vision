"""Заполняет ячейки таблиц распознанным текстом."""

from __future__ import annotations

from loguru import logger

from vision_core.entities.bbox import BBox
from vision_core.entities.table import Table
from vision_core.ocr.base import OcrResult


class CellTextFiller:
    """Заполняет ячейки таблиц текстом из OCR-результатов.

    Каждый OCR-блок назначается ячейке с максимальной долей перекрытия
    (intersection / ocr_area). Это корректно обрабатывает широкие блоки
    объединённых ячеек, которые при центроид-методе попадают не туда.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        min_overlap_ratio: float = 0.1,
        overflow_threshold: float = 15.0,
        debug_image=None,
        ocr_engine=None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_overlap_ratio = min_overlap_ratio
        self.overflow_threshold = overflow_threshold
        self._debug = debug_image
        self._ocr_engine = ocr_engine

    def fill_cells(
        self,
        tables: list[Table],
        ocr_results: list[OcrResult],
        image=None,
        page_number: int = 0,
    ) -> None:
        """Заполняет ячейки таблиц текстом in-place.

        Args:
            tables: Список таблиц для заполнения.
            ocr_results: Результаты OCR со всего изображения.
        """
        if not tables:
            logger.warning("Нет таблиц для заполнения, пропускаем этап cell text filling.")
            return

        if self._debug and image is not None:
            self._debug.on_ocr_boxes(
                image, boxes=[r.bbox for r in ocr_results], stage="5_ocr_words", page_number=page_number
            )

        for table in tables:
            logger.debug(f"Заполнение ячеек для таблицы {table.id} с {len(table.cells)} ячейками...")

            cell_texts: dict[int, list[str]] = {id(c): [] for c in table.cells}
            cell_blobs: dict[int, list[BBox]] = {id(c): [] for c in table.cells}

            for ocr_item in ocr_results:
                if ocr_item.confidence < self.confidence_threshold:
                    continue
                ocr_bbox = BBox.from_tuple(ocr_item.bbox)
                ocr_area = ocr_bbox.area
                if ocr_area == 0:
                    continue

                best_cell = None
                best_ratio = self.min_overlap_ratio
                # best_inter = 0.0

                for cell in table.cells:
                    inter = ocr_bbox.intersect(cell.bbox)
                    ratio = inter / ocr_area
                    if ratio > best_ratio:
                        best_ratio = ratio
                        # best_inter = inter
                        best_cell = cell

                if best_cell is None:
                    continue

                cell_texts[id(best_cell)].append(ocr_item.text)
                cell_blobs[id(best_cell)].append(ocr_bbox)
                # logger.debug(
                #     f"OCR '{ocr_item.text[:25]}' "
                #     f"bbox={ocr_bbox.to_tuple()} w={ocr_bbox.width:.0f} "
                #     f"-> R{best_cell.row}:C{best_cell.col} "
                #     f"cell_bbox={best_cell.bbox.to_tuple()} cell_w={best_cell.bbox.width:.0f} "
                #     f"inter={best_inter:.0f} ratio={best_ratio:.2f}"
                # )

            for cell in table.cells:
                cell.value = "\n".join(cell_texts[id(cell)])
                cell.blobs = cell_blobs[id(cell)]
                logger.debug(f"R{cell.row}:C{cell.col} - {cell.value}")

            logger.debug("=" * 40)
            logger.debug(f"Завершено заполнение таблицы {table.id}.")

            if self._ocr_engine is not None and image is not None:
                reocr_cells = self._find_overflow_cells(table, cell_blobs)
                if reocr_cells:
                    logger.debug(f"Переполненных ячеек в {table.id}: {len(reocr_cells)} — запуск re-OCR")
                    self._refill_cells_by_crop(reocr_cells, image)

        if self._debug and image is not None:
            self._debug.on_table_structures(image, tables=tables, stage="6_table_structures", page_number=page_number)

    def _find_overflow_cells(
        self,
        table: Table,
        cell_blobs: dict[int, list[BBox]],
    ) -> list:
        """Находит ячейки где OCR-блок выходит за границу ячейки.

        Если блок выходит влево/вправо значительно (> overflow_threshold),
        значит PaddleOCR объединил текст из нескольких ячеек в один блок.
        Такую ячейку и пустые ячейки в зоне переполнения нужно перезаполнить.
        """
        thr = self.overflow_threshold
        reocr_ids: set[int] = set()

        for cell in table.cells:
            blobs = cell_blobs.get(id(cell), [])
            for blob in blobs:
                overflow_left = cell.bbox.x_min - blob.x_min
                overflow_right = blob.x_max - cell.bbox.x_max
                if overflow_left <= thr and overflow_right <= thr:
                    continue

                reocr_ids.add(id(cell))
                logger.debug(
                    f"Overflow R{cell.row}:C{cell.col}: "
                    f"blob={blob.to_tuple()} cell_bbox={cell.bbox.to_tuple()} "
                    f"left={overflow_left:.0f} right={overflow_right:.0f}"
                )

                # Найти пустые ячейки в зоне переполнения (они потеряли свой текст)
                overflow_bbox = BBox(
                    x_min=min(blob.x_min, cell.bbox.x_min),
                    y_min=blob.y_min,
                    x_max=max(blob.x_max, cell.bbox.x_max),
                    y_max=blob.y_max,
                )
                for other in table.cells:
                    if id(other) == id(cell):
                        continue
                    if other.is_empty and overflow_bbox.intersect(other.bbox) > 0:
                        reocr_ids.add(id(other))
                        logger.debug(f"  -> пустая смежная R{other.row}:C{other.col} добавлена в re-OCR")

        return [cell for cell in table.cells if id(cell) in reocr_ids]

    def _refill_cells_by_crop(self, cells: list, image) -> None:
        """Перезаполняет ячейки отдельным OCR по кропу."""
        for cell in cells:
            crop = cell.bbox.roi(image)
            if crop.size == 0:
                continue
            results = self._ocr_engine.predict([crop])
            cell_results = results[0] if results else []
            texts = [r.text for r in cell_results if r.confidence >= self.confidence_threshold]
            cell.value = "\n".join(texts)
            cell.blobs = []
            logger.debug(f"Re-OCR R{cell.row}:C{cell.col} -> {cell.value!r}")

    def exclude_table_text(
        self,
        ocr_results: list[OcrResult],
        tables: list[Table],
    ) -> list[OcrResult]:
        """Возвращает OCR-результаты, не попадающие ни в одну таблицу.

        Args:
            ocr_results: Все OCR-результаты страницы.
            tables: Список таблиц для исключения.

        Returns:
            Отфильтрованный список OCR-результатов.
        """
        if not tables:
            logger.warning("Нет таблиц для исключения, пропускаем этап exclude table text.")
            return ocr_results

        return [
            item
            for item in ocr_results
            if not any(table.bbox.contains_center(BBox.from_tuple(item.bbox)) for table in tables)
        ]
