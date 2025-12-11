import pytest
import cv2
from pathlib import Path
from loguru import logger
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.entities.bbox import BBox
import numpy as np


# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestTableDetector:
    """Тесты для TableDetector"""

    @pytest.fixture
    def pdf_path(self) -> Path:
        """Путь к папке с тестовыми PDF"""
        return Path("./examples/test")

    @pytest.fixture
    def output_dir(self) -> Path:
        """Путь к папке для сохранения результатов"""
        output_dir = Path("./examples/output/table_detection")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def preprocessor(self) -> TableDetector:
        """Экземпляр TableDetector"""
        return TableDetector()

    @pytest.fixture
    def drawer_bbox_and_label(self):
        """Функция для отрисовки BBox и метки на изображении"""

        def _drawer_bbox_and_label(
            image: np.ndarray, bbox: BBox, label: str, color: tuple = (255, 0, 0)
        ) -> np.ndarray:
            """Отрисовывает BBox и метку на изображении"""
            pt1 = (int(bbox.x_min), int(bbox.y_min))
            pt2 = (int(bbox.x_max), int(bbox.y_max))
            cv2.rectangle(image, pt1, pt2, color, 2)
            # добавить фон для текста
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                2,
            )
            cv2.rectangle(
                image,
                (pt1[0], pt1[1] - text_height - 15),
                (pt1[0] + text_width, pt1[1]),
                (255, 0, 0),
                -1,
            )
            cv2.putText(
                image,
                label,
                (pt1[0], pt1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            return image

        return _drawer_bbox_and_label

    def test_find_tables(
        self,
        pdf_path: Path,
        output_dir: Path,
        preprocessor: TableDetector,
        drawer_bbox_and_label,
    ):
        """Тестирует детекцию таблиц на изображении"""

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files[:1]:
            logger.info(f"Тестирование на файле: {test_file.name}")
            pdf_bytes = test_file.read_bytes()
            with PDFLoader(pdf_bytes) as loader:
                original = loader.get_page_image(0, dpi=300)

            preprocessor_img = ImagePreprocessor()
            processed = preprocessor_img.process(original)
            bboxes = preprocessor.extract_tables(processed)
            debug_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            for i, bbox in enumerate(bboxes):
                debug_image = drawer_bbox_and_label(
                    debug_image, bbox, label=f"Table {i + 1}"
                )
            cv2.imwrite(
                str(output_dir / f"{test_file.stem}_detected_tables.png"), debug_image
            )
