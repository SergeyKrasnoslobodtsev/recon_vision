from pathlib import Path

import pytest
from loguru import logger

from vision_core.debug_image_observer import DebugImageObserver
from vision_core.loader.pdf_loader import PDFLoader
from vision_core.preprocessor.image_orientation import PageOrientationPreprocessor
from vision_core.utils.image_utils import rotate_image


def _normalize_signed_angle(angle_deg: float) -> float:
    normalized = angle_deg % 360
    if normalized > 180:
        return normalized - 360
    return normalized


class TestPreprocessingPageOrientation:
    @pytest.fixture(scope="class")
    def orientation_preprocessor(self) -> PageOrientationPreprocessor:
        return PageOrientationPreprocessor()

    @pytest.mark.parametrize("input_rotation_deg", [0.5, 1.0, 91.0, -91.0, 179.0, -179.0, 269.0, -269.0])
    def test_page_orientation_on_rotated_page(
        self,
        pdf_file: Path,
        output_dir: Path,
        orientation_preprocessor: PageOrientationPreprocessor,
        input_rotation_deg: float,
    ):
        observer = DebugImageObserver(output_dir=output_dir)

        with PDFLoader(pdf_file.read_bytes()) as loader:
            original = loader.get_page_image(0, dpi=300)

        rotated = rotate_image(original, input_rotation_deg)
        aligned, metadata = orientation_preprocessor.process(rotated)

        predicted_orientation_deg = int(metadata["orientation_deg"])
        coarse_correction_deg = _normalize_signed_angle(predicted_orientation_deg)
        total_correction_deg = coarse_correction_deg + float(metadata["deskew_angle_deg"])
        correction_error_deg = abs(_normalize_signed_angle(input_rotation_deg + total_correction_deg))

        logger.info(f"{pdf_file.name} угол вход={input_rotation_deg} коррекция={total_correction_deg:.2f}")

        observer.on_debug_image(
            rotated, aligned,
            stage="page_alignment",
            prefix=f"{input_rotation_deg}_{pdf_file.stem}",
            page_number=0,
        )

        assert correction_error_deg <= 1.0
        assert metadata["orientation_score"] >= 0.7
