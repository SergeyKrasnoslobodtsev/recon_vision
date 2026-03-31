from pathlib import Path

import pytest
from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.utils.image_processing import rotate_image


def _normalize_signed_angle(angle_deg: float) -> float:
    """Нормализует угол к диапазону от -180 до 180 градусов.

    Args:
        angle_deg: Угол в градусах.

    Returns:
        float: Нормализованный signed-угол.
    """
    normalized = angle_deg % 360
    if normalized > 180:
        return normalized - 360
    return normalized


class TestPreprocessingPageOrientation:
    """Тесты для page orientation preprocessing на реальных PDF-страницах."""

    @pytest.fixture(scope="class")
    def orientation_preprocessor(self) -> PageOrientationPreprocessor:
        """Возвращает реальный препроцессор ориентации страницы.

        Returns:
            PageOrientationPreprocessor: Препроцессор ориентации страницы.
        """
        return PageOrientationPreprocessor()

    @pytest.mark.parametrize(
        "input_rotation_deg",
        [
            0.5,
            1.0,
            91.0,
            -91.0,
            179.0,
            -179.0,
            269.0,
            -269.0,
        ],
    )
    def test_page_orientation_on_rotated_page(
        self,
        pdf_path: Path,
        output_dir: Path,
        pdf_loader_single_page,
        orientation_preprocessor: PageOrientationPreprocessor,
        input_rotation_deg: int,
    ):
        """Проверяет выравнивание страницы после искусственного поворота.

        Args:
            pdf_path: Путь к каталогу с тестовыми PDF.
            output_dir: Путь для сохранения debug-изображений.
            pdf_loader_single_page: Фабрика загрузки первой страницы PDF.
            orientation_preprocessor: Реальный препроцессор ориентации страницы.
            input_rotation_deg: Угол, на который была искусственно повернута страница.
        """
        observer = DebugImageObserver(output_dir=output_dir)

        if not pdf_path.exists():
            pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files[:4]:
            pytest.skip(f"PDF файлы не найдены в {pdf_path}")

        for test_file in pdf_files[1:2]:
            pdf_bytes = test_file.read_bytes()
            original = pdf_loader_single_page(pdf_bytes)

            rotated = rotate_image(original, input_rotation_deg)

            aligned, metadata = orientation_preprocessor.process(rotated)

            predicted_orientation_deg = int(metadata["orientation_deg"])
            coarse_correction_deg = _normalize_signed_angle(predicted_orientation_deg)
            total_correction_deg = coarse_correction_deg + float(metadata["deskew_angle_deg"])
            correction_error_deg = abs(_normalize_signed_angle(input_rotation_deg + total_correction_deg))

            logger.info(f"входной угол: {input_rotation_deg} полученный угол: {total_correction_deg:.2f}")

            observer.on_debug_image(
                rotated,
                aligned,
                stage="page_aligment",
                prefix=f"{input_rotation_deg}_{test_file.stem}",
                page_number=0,
            )

            assert correction_error_deg <= 1.0
            assert metadata["orientation_score"] >= 0.7

        logger.success("Тест выравнивания страницы пройден успешно")
