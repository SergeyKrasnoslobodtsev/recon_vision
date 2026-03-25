import pytest
from pathlib import Path
from loguru import logger
from vision_core.analizer.page_analyzer import PageAnalyzer
from vision_core.entities.paragraph import ParagraphType

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
        for test_file in pdf_files[:5]:  # Ограничиваем количество тестируемых файлов для ускорения тестов
            logger.info(f"Тестирование на файле: {test_file.name}")

            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            page = page_analyzer.analyze_page(original)

            debug_image = original.copy()

            drawer = Drawer(debug_image, side_by_side=True, text_panel_width=700)
            paragraph_lines: list[str] = []

            for paragraph in sorted(page.paragraphs, key=lambda item: item.bbox.y_min):
                if paragraph.type == ParagraphType.HEADER:
                    paragraph_color = "darkgreen"
                    paragraph_fill = (144, 238, 144, 96)
                    drawer.draw_structure(
                        paragraph.bbox.to_tuple(),
                        label=f"{paragraph.type} {paragraph.id}",
                        color=paragraph_color,
                        position=0,
                        fill=paragraph_fill,
                        draw_on="left",
                    )
                elif paragraph.type == ParagraphType.FOOTER:
                    paragraph_color = "darkblue"
                    paragraph_fill = (173, 216, 230, 96)
                    drawer.draw_structure(
                        paragraph.bbox.to_tuple(),
                        label=f"{paragraph.type} {paragraph.id}",
                        color=paragraph_color,
                        position=0,
                        fill=paragraph_fill,
                        draw_on="left",
                    )
                else:
                    paragraph_color = "darkorange"
                    paragraph_fill = (255, 200, 120, 96)
                    drawer.draw_structure(
                        paragraph.bbox.to_tuple(),
                        label=f"{paragraph.type} {paragraph.id}",
                        color=paragraph_color,
                        position=0,
                        fill=paragraph_fill,
                        draw_on="left",
                    )

                drawer.draw_structure(
                    paragraph.bbox.to_tuple(),
                    label=f"{paragraph.type} {paragraph.id}",
                    color=paragraph_color,
                    position=0,
                    draw_on="right",
                )

                if normalized_text := " ".join(paragraph.text.split()):
                    drawer.draw_text_in_bbox(
                        paragraph.bbox.to_tuple(),
                        text=normalized_text,
                        fill=paragraph_color,
                        draw_on="right",
                    )
                    paragraph_lines.append(
                        f"[{paragraph.type.name}] {paragraph.id}: {normalized_text}"
                    )

            for table in page.tables:
                drawer.draw_structure(
                    table.bbox.to_tuple(),
                    label=f"Table {table.id}",
                    color="blue",
                    position=0,
                    draw_on="left",
                )
                for cell in table.cells:
                    for blob in cell.blobs:
                        blob_bbox = blob.to_tuple() if hasattr(blob, "to_tuple") else blob
                        drawer.draw_structure(
                            blob_bbox,
                            color="blue",
                            width=1,
                            fill=(135, 206, 250, 96),
                            draw_on="left",
                        )

                drawer.draw_table_structure(
                    table.bbox.to_tuple(),
                    cells=table.cells,
                    label=f"Table {table.id}",
                    color="blue",
                    draw_on="right",
                )

            drawer.draw_text_panel(
                paragraph_lines,
                title=f"Paragraphs: {len(paragraph_lines)}",
            )

            drawer.save(output_dir / f"detected_paragraphs_{test_file.stem}.png")
