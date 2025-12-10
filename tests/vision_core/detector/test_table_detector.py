import pytest
import cv2
from pathlib import Path
from loguru import logger
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_preprocessor import ImagePreprocessor
from vision_core.detector.table_detector import TableDetector


def test_visualize_table_mask():
    """Визуализирует процесс детекции маски таблицы"""

    test_dir = Path("./examples/test")
    output_dir = Path("./examples/output/table_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        pytest.skip(f"Папка с тестовыми файлами не найдена: {test_dir}")

    pdf_files = list(test_dir.glob("*.pdf"))

    if not pdf_files:
        pytest.skip(f"PDF файлы не найдены в {test_dir}")

    # test_file = pdf_files[8]

    for test_file in pdf_files:
        logger.info(f"Тестирование на файле: {test_file.name}")
        pdf_bytes = test_file.read_bytes()
        with PDFLoader(pdf_bytes) as loader:
            original = loader.get_page_image(0, dpi=300)

            preprocessor = ImagePreprocessor()
            processed = preprocessor.process(original)

            detector = TableDetector()
            blur = cv2.GaussianBlur(processed, (3, 3), 0)
            binary = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]

            h_mask = detector._detect_horizontal_lines(binary)
            v_mask = detector._detect_vertical_lines(binary)

            table_mask = detector._create_table_mask(h_mask, v_mask)

            bboxes = detector._find_tables_components(table_mask)

            # Визуализация линий
            line_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            for bbox in bboxes:
                pt1 = (int(bbox.x_min), int(bbox.y_min))
                pt2 = (int(bbox.x_max), int(bbox.y_max))
                cv2.rectangle(line_image, pt1, pt2, (0, 0, 255), 2)
            cv2.imwrite(
                str(output_dir / f"{test_file.stem}_detected_table.png"), line_image
            )

            cv2.drawContours

            cv2.imwrite(
                str(output_dir / f"{test_file.stem}_table_mask.png"), table_mask
            )
            logger.success("Маска таблицы сохранена")

    logger.info(f"Все результаты сохранены в {output_dir}")
