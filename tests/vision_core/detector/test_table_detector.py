import pytest
from pathlib import Path
from loguru import logger
import numpy as np
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.utils.drawer import Drawer

# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestTableDetector:
    """Тесты для TableDetector"""

    def test_find_tables(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        table_detector: TableDetector,
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

            original = pdf_loader_single_page(pdf_bytes)

            processed_img = preprocessor_img.process(original)

            mask_table = preprocessor_table.create_table_mask(processed_img)

            bboxes = table_detector.extract_raw_tables(mask_table)

            debug_image = original.copy()
            drawer = Drawer(debug_image, side_by_side=True)
            for i, bbox in enumerate(bboxes):
                drawer.draw_structure(
                    bbox.to_tuple(),
                    label=f"Table {i + 1}",
                    color="blue",
                    position=0,
                )
            drawer.save(output_dir / f"detected_tables_{test_file.stem}.png")

    def test_extract_tables(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        table_detector: TableDetector,
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

            original = pdf_loader_single_page(pdf_bytes)

            processed_img = preprocessor_img.process(original)

            tables = table_detector.detect_tables(processed_img)

            debug_image = original.copy()

            drawer = Drawer(debug_image, side_by_side=True)

            if not tables:
                logger.warning(f"Таблицы не найдены в файле: {test_file.name}")
                # debug_image.save(output_dir / f"no_tables_{test_file.stem}.png")
            else:
                for i, table in enumerate(tables):
                    drawer.draw_structure(
                        table.bbox.to_tuple(),
                        label=f"Table {i + 1}",
                        color="blue",
                        position=0,
                    )
                    for cell in table.cells:
                        drawer.draw_structure(
                            cell.bbox.to_tuple(),
                            label=f"R{cell.row}C{cell.col}S{cell.colspan}",
                            color="darkgreen",
                            position=0,
                        )
            drawer.save(output_dir / f"detected_tables_struct_{test_file.stem}.png")
