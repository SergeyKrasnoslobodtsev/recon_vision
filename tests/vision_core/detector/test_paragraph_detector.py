import pytest
from pathlib import Path
from loguru import logger
from vision_core.analizer.page_analyzer import PageAnalyzer

from vision_core.utils.drawer import Drawer

import numpy as np


# command pytest tests/vision_core/detector/test_paragraph_detector.py -v -s


class TestParagraphDetector:
    """Тесты для ParagraphDetector"""

    def test_page_analyzer(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        page_analyzer: PageAnalyzer,
    ):
        """Тестирует детекцию абзацев на изображении"""

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")
        page_analyzer = PageAnalyzer()
        for test_file in pdf_files[:1]:
            logger.info(f"Тестирование на файле: {test_file.name}")

            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            page = page_analyzer.analyze_page(original)

            debug_image = original.copy()

            drawer = Drawer(debug_image, side_by_side=True)

            for paragraph in page.paragraphs:
                if paragraph.type == 0:
                    drawer.draw_structure(
                        paragraph.bbox.to_tuple(),
                        label=f"{paragraph.type} {paragraph.id}",
                        color="darkgreen",
                        position=0,
                    )
                elif paragraph.type == 1:
                    drawer.draw_structure(
                        paragraph.bbox.to_tuple(),
                        label=f"{paragraph.type} {paragraph.id}",
                        color="darkblue",
                        position=0,
                    )
                else:
                    drawer.draw_structure(
                        paragraph.bbox.to_tuple(),
                        label=f"{paragraph.type} {paragraph.id}",
                        color="darkorange",
                        position=0,
                    )

            for table in page.tables:
                drawer.draw_structure(
                    table.bbox.to_tuple(),
                    label=f"Table {table.id}",
                    color="blue",
                    position=0,
                )

            drawer.save(output_dir / f"detected_paragraphs_{test_file.stem}.png")
