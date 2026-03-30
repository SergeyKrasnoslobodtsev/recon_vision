"""Unit-тесты для DebugImageObserver."""

import numpy as np

from vision_core.debug_image_observer import DebugImageObserver


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
