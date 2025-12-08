import pytest
import cv2
from pathlib import Path
from loguru import logger
from vision_core.loader.pdf_loader import PDFLoader


def test_real_reconciliation_act():
    """Тест на реальном акте сверки"""

    pdf_path = Path("./examples/АС Евросибэнерго-НКАЗ.pdf")

    if not pdf_path.exists():
        pytest.skip("Положите реальный акт сверки в fixtures/real_act.pdf")

    pdf_bytes = pdf_path.read_bytes()

    with PDFLoader(pdf_bytes) as loader:
        logger.info(f"Документ загружен, страниц: {loader.num_pages}")

        page_num = 0

        width, height = loader.get_page_size(page_num)
        logger.info(f"Размер страницы: {width:.2f} x {height:.2f} px")

        text = loader.get_text(page_num)
        has_text = loader.has_text_layer(page_num)
        logger.info(f"Текстовый слой: {has_text}, длина: {len(text)} символов")

        if text:
            logger.debug(f"Первые 200 символов:\n{text}")

        img = loader.get_page_image(page_num, dpi=300)
        logger.info(
            f"Изображение: {img.shape[1]}x{img.shape[0]} px, каналов: {img.shape[2]}"
        )

        output_path = Path("./examples/output/rendered_page.png")
        cv2.imwrite(str(output_path), img)
        logger.success(f"Изображение сохранено: {output_path}")

        assert loader.num_pages > 0
        assert width > 0 and height > 0
        assert isinstance(has_text, bool)
        assert img.shape[0] > 0
