from pathlib import Path

from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.table_cell_detector import TableCellDetector
from vision_core.detector.table_detector import TableDetector
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.preprocessor.table_preprocessor import TablePreprocessor


class TestCellDetector:
    def test_extract_cells(self, pdf_file: Path, output_dir: Path):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        processed = ImagePreprocessor().process(original)
        processed, _ = PageOrientationPreprocessor().process(processed)
        table_mask = TablePreprocessor().create_table_mask(processed)
        bboxes = TableDetector().extract_raw_tables(table_mask)

        items: list[tuple[tuple[int, int, int, int], str]] = []
        cell_detector = TableCellDetector()
        for bbox in bboxes:
            roi_mask = bbox.roi(table_mask)
            cells = cell_detector.extract_cells(roi_mask, bbox.to_tuple(), merge_mode="cols")
            for cell in cells:
                items.append((cell.bbox.to_tuple(), f"R{cell.row}C{cell.col}S{cell.colspan}"))

        observer.on_labeled_boxes(
            processed, items=items, stage="detected_cells", prefix=pdf_file.stem, page_number=0, color="blue"
        )
        logger.success(f"Детекция ячеек пройдена: {pdf_file.name}")
