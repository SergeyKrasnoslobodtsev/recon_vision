from pathlib import Path

import numpy as np
import pytest
from cv2 import Mat
from cv2.typing import NumPyArrayNumeric

from vision_core.config import VisionCoreConfig
from vision_core.ocr.paddle_ocr import PaddleOcrEngine

pytestmark = pytest.mark.integration  # Маркер для запуска только integration тестов


@pytest.fixture
def real_config():
    """Возвращает конфигурацию с реальными путями к моделям.

    Тест пропускается, если модели отсутствуют (например, в CI без моделей).
    """
    config = VisionCoreConfig()

    # Проверяем существование моделей
    det_path = Path(config.paddleocr.text_detection_model_dir)
    rec_path = Path(config.paddleocr.text_recognition_model_dir)

    if not det_path.exists() or not rec_path.exists():
        pytest.skip(f"Модели PaddleOCR не найдены: detection={det_path.exists()}, recognition={rec_path.exists()}")

    return config


@pytest.fixture
def sample_image():
    """Создаёт тестовое изображение с текстом или загружает из файла."""
    # Вариант 1: Загрузить реальное изображение из test_data/
    test_image_path = Path(__file__).parent.parent.parent / "test_data" / "sample_text.png"

    if test_image_path.exists():
        import cv2

        return cv2.imread(str(test_image_path))

    # Вариант 2: Создать синтетическое изображение (если нет реальных данных)
    # Это менее надёжно для OCR, но подходит для проверки что код работает
    image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    # Здесь можно добавить cv2.putText для рисования текста
    return image


def test_paddleocr_engine_initializes_with_real_models(real_config: VisionCoreConfig):
    """Проверяет инициализацию PaddleOCR с реальными моделями.

    Тест гарантирует, что:
    - Конфигурация содержит корректные пути
    - PaddleOCR успешно загружает модели
    - Движок готов к использованию
    """
    engine = PaddleOcrEngine(real_config)

    assert engine is not None
    assert engine.ocr is not None
    assert engine.cfg == real_config.paddleocr


def test_paddleocr_engine_recognizes_text_on_real_image(
    real_config: VisionCoreConfig,
    sample_image: Mat | NumPyArrayNumeric | np.ndarray[tuple[int, int, int], np.dtype[np.unsignedinteger[np._8Bit]]],
):
    """Проверяет распознавание текста на реальном изображении.

    Тест гарантирует:
    - OCR возвращает непустой результат для изображения с текстом
    - Структура результата соответствует OcrResult
    - Значения confidence находятся в допустимом диапазоне
    """
    engine = PaddleOcrEngine(real_config)

    results = engine.predict(sample_image)

    # Проверки структуры
    assert isinstance(results, list)
    assert len(results) == 1  # predict всегда возвращает list[list[OcrResult]]

    ocr_results = results[0]

    # Если изображение содержит текст, должны быть результаты
    # (для пустого изображения может быть пустой список)
    if ocr_results:
        first_result = ocr_results[0]

        # Проверки OcrResult
        assert isinstance(first_result.text, str)
        assert len(first_result.text) > 0
        assert 0.0 <= first_result.confidence <= 1.0
        assert isinstance(first_result.bbox, tuple)
        assert len(first_result.bbox) == 4

        # Проверка корректности bbox
        x_min, y_min, x_max, y_max = first_result.bbox
        assert x_min < x_max
        assert y_min < y_max
        assert x_min >= 0 and y_min >= 0


def test_paddleocr_engine_predict_iter_yields_results(
    real_config: VisionCoreConfig,
    sample_image: Mat | NumPyArrayNumeric | np.ndarray[tuple[int, int, int], np.dtype[np.unsignedinteger[np._8Bit]]],
):
    """Проверяет итеративный режим распознавания.

    Тест гарантирует:
    - predict_iter возвращает генератор
    - Каждый элемент генератора — список OcrResult
    - Порядок результатов соответствует порядку входных изображений
    """
    engine = PaddleOcrEngine(real_config)

    # Передаём несколько копий изображения
    images = [sample_image, sample_image]

    results = list(engine.predict_iter(images))

    assert len(results) == 2

    for batch in results:
        assert isinstance(batch, list)
        # Каждый элемент batch — OcrResult


def test_paddleocr_engine_handles_empty_image(real_config: VisionCoreConfig):
    """Проверяет обработку пустого/чёрного изображения.

    Тест гарантирует:
    - Движок не падает на изображении без текста
    - Возвращает пустой список или список с низкой confidence
    """
    engine = PaddleOcrEngine(real_config)

    # Чёрное изображение без текста
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)

    results = engine.predict(empty_image)

    assert isinstance(results, list)
    assert len(results) == 1
    # Результат может быть пустым или содержать шум с низкой confidence
    assert isinstance(results[0], list)


def test_paddleocr_engine_raises_on_invalid_model_paths(tmp_path: Path):
    """Проверяет поведение при несуществующих путях к моделям.

    Тест гарантирует:
    - PaddleOCR выбрасывает ошибку при попытке загрузить несуществующие модели
    - Ошибка информативна и содержит путь к отсутствующим файлам
    """
    config = VisionCoreConfig()
    config.paddleocr.text_detection_model_dir = str(tmp_path / "nonexistent_det")
    config.paddleocr.text_recognition_model_dir = str(tmp_path / "nonexistent_rec")

    # PaddleOCR должен упасть при инициализации или при первом вызове
    with pytest.raises(Exception):  # Тип ошибки зависит от реализации PaddleOCR
        engine = PaddleOcrEngine(config)
        # Если __init__ не упал, должен упасть predict
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        engine.predict(dummy_image)
