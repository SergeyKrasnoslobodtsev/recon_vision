import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from vision_core.preprocessor.image_analyzer import ImageAnalyzer


def test_collect_metrics_from_test_samples(
    pdf_path: Path,
    output_dir: Path,
    pdf_loader_single_page: np.ndarray,
):
    """Собирает метрики со всех файлов из examples/test"""
    if not pdf_path.exists():
        pytest.skip(f"Папка с тестовыми файлами не найдена: {pdf_path}")

    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        pytest.skip(f"PDF файлы не найдены в {pdf_path}")

    logger.info(f"Найдено PDF файлов: {len(pdf_files)}")

    analyzer = ImageAnalyzer()
    results = []

    for pdf_path in pdf_files[:1]:
        logger.info(f"Обработка: {pdf_path.name}")

        pdf_bytes = pdf_path.read_bytes()

        original = pdf_loader_single_page(pdf_bytes)
        metrics = analyzer.analyze(original)

        # Собираем результаты
        results.append(
            {
                "filename": pdf_path.name,
                "has_noise": metrics.has_noise,
                "has_blur": metrics.has_blur,
                "has_colored_bg": metrics.has_colored_bg,
                "has_skew": metrics.has_skew,
                "contrast_level": metrics.contrast_level,
                "text_density": metrics.text_density,
                "noise_ratio": metrics.noise_ratio,
                "sharpness": metrics.sharpness,
                "skew_angle": metrics.skew_angle,
            }
        )

    # Создаём DataFrame
    df = pd.DataFrame(results)

    # Сохраняем в CSV
    # output_path = Path("./examples/output/metrics_analysis.csv")
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(output_path, index=False)

    # logger.success(f"Метрики сохранены: {output_path}")

    # Выводим сводку
    logger.info("\nСводка по метрикам:")
    logger.info(f"\n{df.to_string()}")
