import pytest
from pathlib import Path
from loguru import logger
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.detector.table_detector import TableDetector
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.utils.drawer import Drawer, Position
import numpy as np


# command pytest tests/vision_core/detector/test_table_detector.py -v -s


class TestCellDetector:
    """Тесты для TableCellDetector"""

    def test_extract_cells(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        preprocessor_img: ImagePreprocessor,
        preprocessor_table: TablePreprocessor,
        table_detector: TableDetector,
        cell_detector: TableCellDetector,
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

            processed = preprocessor_img.process(original)

            table_mask = preprocessor_table.create_table_mask(processed)

            bboxes = table_detector.extract_raw_tables(table_mask)

            debug_image = original.copy()

            drawer = Drawer(debug_image, side_by_side=True)

            for bbox in bboxes:
                roi_mask = bbox.roi(table_mask)
                cells_bboxes = cell_detector.extract_cells(
                    roi_mask,
                    bbox.to_tuple(),
                    merge_mode="cols",
                )

                for cell_bbox in cells_bboxes:
                    drawer.draw_structure(
                        cell_bbox.bbox.to_tuple(),
                        label=f"R{cell_bbox.row}C{cell_bbox.col}S{cell_bbox.colspan}",
                        color="darkgreen",
                        position=Position.TOP,
                    )

            drawer.save(output_dir / f"detected_cells_{test_file.stem}.png")
