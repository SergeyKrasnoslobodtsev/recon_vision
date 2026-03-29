"""Unit-тесты для DebugObserver и интеграции observer в PageAnalyzer."""

import numpy as np
from unittest.mock import MagicMock

from vision_core.observer import PipelineObserver
from vision_core.debug_observer import DebugObserver


class TestPipelineObserverProtocol:
    """Проверяет что DebugObserver соответствует протоколу."""

    def test_debug_observer_is_pipeline_observer(self, tmp_path):
        observer = DebugObserver(output_dir=tmp_path)
        assert isinstance(observer, PipelineObserver)


class TestDebugObserver:
    """Проверяет сохранение side-by-side изображений."""

    def _make_image(self, h=100, w=200, color=255):
        return np.full((h, w, 3), color, dtype=np.uint8)

    def test_saves_file_in_stage_subdir(self, tmp_path):
        observer = DebugObserver(output_dir=tmp_path)
        original = self._make_image(color=200)
        result = self._make_image(color=100)

        observer.on_side_by_side(original, result, stage="preprocessing", page_number=0)

        saved = tmp_path / "preprocessing" / "page_000.png"
        assert saved.exists()

    def test_page_numbering(self, tmp_path):
        observer = DebugObserver(output_dir=tmp_path)
        original = self._make_image()
        result = self._make_image()

        observer.on_side_by_side(original, result, stage="table_detection", page_number=5)

        saved = tmp_path / "table_detection" / "page_005.png"
        assert saved.exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        output_dir = tmp_path / "nested" / "debug"
        observer = DebugObserver(output_dir=output_dir)

        assert output_dir.exists()

    def test_grayscale_result_image(self, tmp_path):
        """Проверяет что grayscale result-изображение корректно обрабатывается."""
        observer = DebugObserver(output_dir=tmp_path)
        original = self._make_image(color=200)
        result = np.full((100, 200), 100, dtype=np.uint8)  # grayscale

        observer.on_side_by_side(original, result, stage="preprocessing", page_number=0)

        saved = tmp_path / "preprocessing" / "page_000.png"
        assert saved.exists()

    def test_multiple_stages_separate_dirs(self, tmp_path):
        """Разные stage сохраняются в разные поддиректории."""
        observer = DebugObserver(output_dir=tmp_path)
        img = self._make_image()

        observer.on_side_by_side(img, img, stage="preprocessing", page_number=0)
        observer.on_side_by_side(img, img, stage="table_detection", page_number=0)

        assert (tmp_path / "preprocessing" / "page_000.png").exists()
        assert (tmp_path / "table_detection" / "page_000.png").exists()


class TestPageAnalyzerObserverIntegration:
    """Проверяет что PageAnalyzer вызывает observer."""

    def test_observer_called_on_preprocess(self):
        """Observer.on_side_by_side вызывается при анализе страницы."""
        from vision_core.analizer.page_analyzer import PageAnalyzer

        mock_observer = MagicMock(spec=PipelineObserver)

        mock_preprocessor = MagicMock()
        mock_preprocessor.process.return_value = np.zeros((100, 200), dtype=np.uint8)

        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = []

        mock_table_detector = MagicMock()
        mock_table_detector.detect_tables.return_value = []

        mock_paragraph_detector = MagicMock()
        mock_paragraph_detector.detect_paragraphs.return_value = []

        analyzer = PageAnalyzer(
            image_preprocessor=mock_preprocessor,
            table_detector=mock_table_detector,
            ocr_engine=mock_ocr,
            paragraph_detector=mock_paragraph_detector,
            observer=mock_observer,
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        analyzer.analyze_page(image, page_number=3)

        mock_observer.on_side_by_side.assert_called_once()
        call_kwargs = mock_observer.on_side_by_side.call_args
        assert call_kwargs.kwargs["stage"] == "preprocessing"
        assert call_kwargs.kwargs["page_number"] == 3

    def test_no_observer_no_error(self):
        """Без observer анализ проходит без ошибок."""
        from vision_core.analizer.page_analyzer import PageAnalyzer

        mock_preprocessor = MagicMock()
        mock_preprocessor.process.return_value = np.zeros((100, 200), dtype=np.uint8)

        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = []

        mock_table_detector = MagicMock()
        mock_table_detector.detect_tables.return_value = []

        mock_paragraph_detector = MagicMock()
        mock_paragraph_detector.detect_paragraphs.return_value = []

        analyzer = PageAnalyzer(
            image_preprocessor=mock_preprocessor,
            table_detector=mock_table_detector,
            ocr_engine=mock_ocr,
            paragraph_detector=mock_paragraph_detector,
            observer=None,
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        analyzer.analyze_page(image, page_number=0)
