"""Unit-тесты для DebugImageObserver."""

from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.entities.bbox import BBox
from vision_core.entities.cell import Cell
from vision_core.entities.page import Page
from vision_core.entities.paragraph import Paragraph, ParagraphType
from vision_core.entities.table import Table
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.pipelines.build_document import DocumentBuildPipeline


class TestDebugImageObserver:
    """Проверяет сохранение side-by-side изображений."""

    def _make_image(self, h=100, w=200, color=255):
        return np.full((h, w, 3), color, dtype=np.uint8)

    def test_saves_file_in_stage_subdir(self, tmp_path):
        observer = DebugImageObserver(output_dir=tmp_path)
        original = self._make_image(color=200)
        result = self._make_image(color=100)

        observer.on_debug_image(original, result, stage="preprocessing", page_number=0)

        saved = tmp_path / "preprocessing" / "page_000.png"
        assert saved.exists()

    def test_page_numbering(self, tmp_path):
        observer = DebugImageObserver(output_dir=tmp_path)
        original = self._make_image()
        result = self._make_image()

        observer.on_debug_image(original, result, stage="table_detection", page_number=5)

        saved = tmp_path / "table_detection" / "page_005.png"
        assert saved.exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        output_dir = tmp_path / "nested" / "debug"
        DebugImageObserver(output_dir=output_dir)

        assert output_dir.exists()

    def test_grayscale_result_image(self, tmp_path):
        """Проверяет что grayscale result-изображение корректно обрабатывается."""
        observer = DebugImageObserver(output_dir=tmp_path)
        original = self._make_image(color=200)
        result = np.full((100, 200), 100, dtype=np.uint8)  # grayscale

        observer.on_debug_image(original, result, stage="preprocessing", page_number=0)

        saved = tmp_path / "preprocessing" / "page_000.png"
        assert saved.exists()

    def test_multiple_stages_separate_dirs(self, tmp_path):
        """Разные stage сохраняются в разные поддиректории."""
        observer = DebugImageObserver(output_dir=tmp_path)
        img = self._make_image()

        observer.on_debug_image(img, img, stage="preprocessing", page_number=0)
        observer.on_debug_image(img, img, stage="table_detection", page_number=0)

        assert (tmp_path / "preprocessing" / "page_000.png").exists()
        assert (tmp_path / "table_detection" / "page_000.png").exists()

    def test_saves_detected_boxes_as_single_image(self, tmp_path):
        """Сохраняет одно изображение с детектируемыми рамками."""

        observer = DebugImageObserver(output_dir=tmp_path)
        image = self._make_image()

        observer.on_detected_boxes(
            image,
            boxes=[(10, 10, 60, 40), (80, 20, 140, 70)],
            stage="paragraph_words",
            page_number=0,
            color="darkgreen",
            fill=(144, 238, 144, 96),
        )

        saved = tmp_path / "paragraph_words" / "page_000.png"
        assert saved.exists()

    def test_saves_labeled_boxes_as_single_image(self, tmp_path):
        """Сохраняет одно изображение с рамками и подписями."""

        observer = DebugImageObserver(output_dir=tmp_path)
        image = self._make_image()

        observer.on_labeled_boxes(
            image,
            items=[((10, 10, 60, 40), "TEXT 1"), ((80, 20, 140, 70), "HEADER 2")],
            stage="paragraph_grouped",
            page_number=0,
            color="darkorange",
        )

        saved = tmp_path / "paragraph_grouped" / "page_000.png"
        assert saved.exists()

    def test_saves_canonical_page_text_overlay_with_expected_colors(self, tmp_path):
        """Подсвечивает только blobs текста из канонической страницы нужными цветами."""

        observer = DebugImageObserver(output_dir=tmp_path)
        image = self._make_image(h=100, w=120)
        page = Page(
            page_number=0,
            paragraphs=[
                Paragraph(
                    id="p1",
                    text="Параграф",
                    type=ParagraphType.TEXT,
                    bbox=BBox(x_min=10, y_min=10, x_max=60, y_max=30),
                    blobs=[BBox(x_min=18, y_min=14, x_max=42, y_max=24)],
                )
            ],
            tables=[
                Table(
                    id="0",
                    bbox=BBox(x_min=10, y_min=40, x_max=90, y_max=80),
                    num_rows=1,
                    num_cols=1,
                    cells=[
                        Cell(
                            row=0,
                            col=0,
                            value="Ячейка",
                            bbox=BBox(x_min=15, y_min=45, x_max=85, y_max=75),
                            blobs=[BBox(x_min=50, y_min=58, x_max=75, y_max=68)],
                        )
                    ],
                )
            ],
        )

        observer.on_canonical_page(
            image,
            page=page,
            stage="canonical_document",
        )

        saved = tmp_path / "canonical_document" / "page_000.png"
        assert saved.exists()

        rendered = Image.open(saved).convert("RGB")
        paragraph_pixel = rendered.getpixel((25, 18))
        table_pixel = rendered.getpixel((60, 62))
        paragraph_bbox_background = rendered.getpixel((50, 25))
        table_bbox_background = rendered.getpixel((20, 50))
        outside_background = rendered.getpixel((95, 15))

        assert paragraph_pixel != (255, 255, 255)
        assert table_pixel != (255, 255, 255)
        assert paragraph_bbox_background == (255, 255, 255)
        assert table_bbox_background == (255, 255, 255)
        assert outside_background == (255, 255, 255)
        assert paragraph_pixel[0] > paragraph_pixel[2]
        assert table_pixel[2] > table_pixel[0]

    def test_saves_canonical_document_overlay_for_real_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
    ):
        """Строит канонический документ из реального PDF и сохраняет overlay страниц."""

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = sorted(pdf_path.glob("*.pdf"))
        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        test_file = pdf_files[0]
        logger.info(f"Визуализация канонического документа: {test_file.name}")

        pipeline = DocumentBuildPipeline()
        observer = DebugImageObserver(output_dir=output_dir)
        pdf_bytes = test_file.read_bytes()
        document = pipeline.build(pdf_bytes)

        with PDFLoader(pdf_bytes) as loader:
            for page in document.pages:
                source_image = loader.get_page_image(page.page_number, dpi=pipeline.dpi)
                observer.on_canonical_page(
                    source_image,
                    page=page,
                    stage="canonical_document",
                    prefix=test_file.stem,
                )

        first_page_path = output_dir / "canonical_document" / f"{test_file.stem}_000.png"
        assert first_page_path.exists()
