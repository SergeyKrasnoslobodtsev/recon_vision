from pathlib import Path

import cv2
import numpy as np
import pytest
from loguru import logger

from vision_core.analizer.page_analyzer import PageAnalyzer
from vision_core.config import ParagraphDetectorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector.paragraph_detector import ParagraphDetector
from vision_core.ocr.base import OcrResult

# command pytest tests/vision_core/detector/test_paragraph_detector.py -v -s


class TestParagraphDetector:
    """Тесты для ParagraphDetector"""

    def test_clusterize_returns_singletons_for_small_input(self):
        """Возвращает одиночные кластеры, если данных меньше порога HDBSCAN."""

        detector = ParagraphDetector(config=ParagraphDetectorConfig(min_cluster_size=3))
        ocr_results = [
            OcrResult(text="one", confidence=0.9, bbox=(10, 10, 60, 40)),
            OcrResult(text="two", confidence=0.95, bbox=(80, 12, 140, 44)),
        ]

        clusters = detector._clusterize(ocr_results, image_shape=(200, 300))

        assert len(clusters) == 2
        assert clusters[0] == [ocr_results[0]]
        assert clusters[1] == [ocr_results[1]]

    def test_paragraph_detection_steps(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page: np.ndarray,
        page_analyzer: PageAnalyzer,
    ):
        """Тестирует пошаговую детекцию абзацев с debug-визуализацией."""

        observer = DebugImageObserver(output_dir=output_dir)

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files[:5]:
            logger.info(f"Тестирование на файле: {test_file.name}")

            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)
            processed = page_analyzer._preprocess_image(original)
            observer.on_debug_image(
                original,
                processed,
                stage="preprocessing",
                prefix=test_file.stem,
                page_number=0,
            )

            tables = page_analyzer._detect_tables(processed)
            ocr_results = page_analyzer._recognize_text(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))
            page_analyzer._fill_table_cells(tables, ocr_results)
            filtered_ocr_results = page_analyzer._exclude_table_text(ocr_results, tables)

            observer.on_detected_boxes(
                original,
                boxes=[item.bbox for item in filtered_ocr_results],
                stage="paragraph_words",
                prefix=test_file.stem,
                page_number=0,
                color="darkgreen",
                fill=(144, 238, 144, 96),
            )

            raw_clusters = page_analyzer.paragraph_detector._clusterize(filtered_ocr_results, processed.shape[:2])
            merged_clusters = page_analyzer.paragraph_detector._merge_nested_clusters(raw_clusters)
            observer.on_detected_boxes(
                original,
                boxes=[
                    page_analyzer.paragraph_detector._cluster_bbox(cluster).to_tuple() for cluster in merged_clusters
                ],
                stage="paragraph_hdbscan_clusters",
                prefix=test_file.stem,
                page_number=0,
                color="darkorange",
                fill=(255, 200, 120, 96),
            )

            paragraphs = page_analyzer.paragraph_detector.detect_paragraphs(
                filtered_ocr_results,
                processed.shape[:2],
            )

            observer.on_labeled_boxes(
                original,
                items=[
                    (paragraph.bbox.to_tuple(), f"{paragraph.type.name} {paragraph.id}") for paragraph in paragraphs
                ],
                stage="paragraph_grouped",
                prefix=test_file.stem,
                page_number=0,
                color="darkorange",
            )
