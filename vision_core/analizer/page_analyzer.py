import numpy as np
import cv2

from vision_core.entities.bbox import BBox
from vision_core.entities.page import Page
from vision_core.entities.table import Table
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.paragraph_preprocessor import ParagraphPreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.ocr.paddle_ocr import PaddleOcrEngine, OcrResult

from typing import Optional
from loguru import logger


class PageAnalyzer:
    """Анализирует страницу для извлечения таблиц и абзацев.

    Класс объединяет препроцессинг изображения, детекцию структурных элементов
    (таблиц, абзацев) и OCR для формирования полного представления страницы.

    Attributes:
        image_preprocessor: Препроцессор изображений страницы.
        paragraph_preprocessor: Препроцессор абзацев.
        table_detector: Детектор таблиц.
        paragraph_detector: Детектор абзацев.
        ocr_engine: Движок оптического распознавания текста.
        CONFIDENCE_THRESHOLD: Порог уверенности OCR для фильтрации результатов.
    """

    CONFIDENCE_THRESHOLD = 0.7  # TODO: убрать в конфиг

    def __init__(
        self,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        paragraph_preprocessor: Optional[ParagraphPreprocessor] = None,
        table_detector: Optional[TableDetector] = None,
        paragraph_detector: Optional[ParagraphDetector] = None,
        ocr_engine: Optional[PaddleOcrEngine] = None,
    ):
        """Инициализирует анализатор страницы с возможностью внедрения зависимостей.

        Args:
            image_preprocessor: Препроцессор изображения страницы.
                По умолчанию создаётся экземпляр ImagePreprocessor.
            paragraph_preprocessor: Препроцессор абзацев.
                По умолчанию создаётся экземпляр ParagraphPreprocessor.
            table_detector: Детектор таблиц.
                По умолчанию создаётся экземпляр TableDetector.
            paragraph_detector: Детектор абзацев.
                По умолчанию создаётся экземпляр ParagraphDetector.
            ocr_engine: OCR движок.
                По умолчанию создаётся экземпляр PaddleOcrEngine.
        """
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.ocr_engine = ocr_engine or PaddleOcrEngine()
        self.paragraph_preprocessor = paragraph_preprocessor or ParagraphPreprocessor()
        self.table_detector = table_detector or TableDetector()
        self.paragraph_detector = paragraph_detector or ParagraphDetector()

    def analyze_page(self, image: np.ndarray) -> Page:
        """Анализирует изображение страницы и извлекает структурированные данные.

        Последовательность обработки:
        1. Предобработка изображения для улучшения качества распознавания.
        2. Детекция таблиц на изображении.
        3. OCR-распознавание всего текста.
        4. Заполнение ячеек таблиц распознанным текстом.

        Args:
            image: Изображение страницы в формате numpy array (BGR или RGB).

        Returns:
            Объект Page с распознанными таблицами, абзацами и метаданными.

        Note:
            Обработка абзацев пока не реализована (TODO #124).
        """
        processed_image = self._preprocess_image(image)

        tables = self._detect_tables(processed_image)

        ocr_results = self._recognize_text(
            cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        )

        self._fill_table_cells(tables, ocr_results)

        filtered_ocr_results = self._exclude_table_text(ocr_results, tables)
        logger.debug("Детекция абзацев начата")
        paragraphs = self.paragraph_detector.detect_paragraphs(
            filtered_ocr_results, processed_image.shape[:2]
        )
        logger.debug(f"Детекция абзацев завершена, обнаружено: {len(paragraphs)}")
        return Page(
            image=image,
            tables=tables,
            paragraphs=paragraphs,  # TODO(#124): Добавить обработку абзацев
            metadata={},
        )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Выполняет предобработку изображения для улучшения качества распознавания.

        Args:
            image: Исходное изображение страницы.

        Returns:
            Предобработанное изображение.
        """
        return self.image_preprocessor.process(image)

    def _detect_tables(self, image: np.ndarray) -> list[Table]:
        """Детектирует таблицы на предобработанном изображении.

        Args:
            image: Предобработанное изображение страницы.

        Returns:
            Список обнаруженных таблиц с их структурой и границами.
        """
        return self.table_detector.detect_tables(image)

    def _recognize_text(self, image: np.ndarray) -> list[OcrResult]:
        """Распознаёт текст на всём изображении.

        Args:
            image: Изображение для распознавания текста.

        Returns:
            Список результатов OCR. Пустой список, если распознавание не дало результатов.

        Note:
            PaddleOCR.predict возвращает list[list[OcrResult]] (по одному списку на изображение),
            метод извлекает первый элемент для одиночного изображения.
        """
        logger.debug("Начало OCR распознавания текста на изображении")
        results = self.ocr_engine.predict(image)
        if not results:
            return []
        logger.debug(
            f"OCR распознавание завершено, средняя уверенность: {np.mean([r.confidence for r in results[0]])}"
        )
        return results[0]

    def _fill_table_cells(
        self,
        tables: list[Table],
        ocr_results: list[OcrResult],
    ) -> None:
        """Заполняет ячейки таблиц распознанным текстом.

        Алгоритм:
        1. Для каждой ячейки собирает OCR-результаты, чей bbox содержится внутри bbox ячейки.
        2. Фильтрует результаты по порогу уверенности (CONFIDENCE_THRESHOLD).
        3. Объединяет тексты через перенос строки, записывает в cell.value.

        Args:
            tables: Список таблиц для заполнения (модифицируются in-place).
            ocr_results: Результаты OCR со всего изображения.

        Note:
            Использует геометрическое вхождение bbox (cell.bbox.contains).
            Это может давать ложные срабатывания, если таблицы пересекаются или
            OCR-блоки выходят за границы ячеек.
        """
        for table in tables:
            for cell in table.cells:
                texts: list[str] = []

                for ocr_item in ocr_results:
                    if ocr_item.confidence < self.CONFIDENCE_THRESHOLD:
                        continue

                    if not cell.bbox.contains(BBox.from_tuple(ocr_item.bbox)):
                        continue

                    texts.append(ocr_item.text)
                    cell.blobs.append(ocr_item.bbox)

                cell.value = "\n".join(texts)

    def _exclude_table_text(
        self,
        ocr_results: list[OcrResult],
        tables: list[Table],
    ):
        """Отфильтровывает результаты ocr находящиеся в границах таблиц

        Args:
            ocr_results (list[OcrResult]): Результаты ocr
            tables (list[Table]): Список таблиц для исключения текста

        Returns:
            list[OcrResult]: Отфильтрованный список результатов ocr
        """
        if not tables:
            return ocr_results

        filtered: list[OcrResult] = []

        for ocr_item in ocr_results:
            is_excluded = False
            ocr_bbox = BBox.from_tuple(ocr_item.bbox)
            for table in tables:
                if table.bbox.contains_center(ocr_bbox):
                    is_excluded = True
                    break
            if not is_excluded:
                filtered.append(ocr_item)

        return filtered
