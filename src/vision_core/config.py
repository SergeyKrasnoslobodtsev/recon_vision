from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TableDetectorConfig:
    """Конфигурация детектора таблиц

    Attributes:
        min_table_area: Минимальная площадь таблицы
        approx_poly_eps: Порог аппроксимации многоугольника
        mode_merge_cells: Режим объединения ячеек поддерживает:
        - all - объединяет строки и столбцы
        - cols - объединяет только столбцы
        - rows - объединяет только строки
        - None - не объединяет ячейки
    """

    min_table_area: int = 10000
    approx_poly_eps: float = 0.02
    mode_merge_cells: Optional[str] = "cols"


@dataclass
class TableCellDetectorConfig:
    """Конфигурация детекции ячеек

    Attributes:
        min_cell: Минимальный размер ячейки
        padding: Отступ вокруг ячейки
        threshold_line: Порог для группировки линий в пикселях
        coverage_thr: Порог покрытия для объединения линий
    """

    min_cell: int = 10
    padding: int = 10
    threshold_line: int = 4
    coverage_thr: float = 0.8


@dataclass
class TablePreprocessorConfig:
    """Конфигурация препроцессора таблиц

    Attributes:
        gaussian_blur_kernel: Размер ядра для гауссова размытия
        horizontal_length_ratio: Минимальная длина горизонтальных линий относительно высоты изображения
        vertical_length_ratio: Минимальная длина вертикальных линий относительно ширины изображения
        min_table_width_ratio: Минимальная ширина таблицы относительно ширины изображения
        min_table_height_ratio: Минимальная высота таблицы относительно ширины таблицы
    """

    gaussian_blur_kernel: int = 1

    # Пороги для детекции линий
    horizontal_length_ratio: float = 0.03
    vertical_length_ratio: float = 0.006

    # Пороги для валидации таблиц
    min_table_width_ratio: float = 0.5  # минимум 50% ширины изображения
    min_table_height_ratio: float = 0.05  # минимум 5% от ширины таблицы


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


@dataclass
class PaddleOcrConfig:
    """Конфигурация для PaddleOCR."""

    text_recognition_model_name: str = "cyrillic_PP-OCRv5_mobile_rec"
    text_recognition_model_dir: str = str(
        _DEFAULT_MODELS_DIR / "cyrillic_PP-OCRv5_mobile_rec"
    )
    text_detection_model_name: str = "PP-OCRv5_server_det"
    text_detection_model_dir: str = str(_DEFAULT_MODELS_DIR / "PP-OCRv5_server_det")
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    device: str = "gpu"


@dataclass
class ParagraphDetectorConfig:
    """Конфигурация детектора параграфов.

    Attributes:
        vertical_spacing_factor: Множитель высоты строки для определения межстрочного расстояния (eps для DBSCAN).
        min_cluster_size: Минимальное количество строк в параграфе.
        stamp_area_threshold: Порог площади для фильтрации штампов (доля от площади изображения).
        column_gap_threshold: Минимальный горизонтальный разрыв для детекции колонок (в пикселях).
        indent_margin: Допуск для определения одинакового отступа (в пикселях).
        grid_variance_threshold: Порог дисперсии расстояний между колонками для детекции text-based таблиц.
    """

    vertical_spacing_factor: float = 2.0
    min_cluster_size: int = 2
    stamp_area_threshold: float = 0.25
    column_gap_threshold: int = 50
    indent_margin: int = 10
    grid_variance_threshold: float = 100.0


@dataclass
class VisionCoreConfig:
    """Общая конфигурация для распознавания таблиц"""

    table_detector: TableDetectorConfig = field(default_factory=TableDetectorConfig)
    cell_detector: TableCellDetectorConfig = field(
        default_factory=TableCellDetectorConfig
    )
    table_preprocessor: TablePreprocessorConfig = field(
        default_factory=TablePreprocessorConfig
    )

    paddleocr: PaddleOcrConfig = field(default_factory=PaddleOcrConfig)

    paragraph_detector: ParagraphDetectorConfig = field(
        default_factory=ParagraphDetectorConfig
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionCoreConfig":
        """Создание из словаря (для загрузки из JSON/YAML)"""
        return cls(
            table_detector=TableDetectorConfig(**config_dict.get("table_detector", {})),
            cell_detector=TableCellDetectorConfig(
                **config_dict.get("cell_detector", {})
            ),
            table_preprocessor=TablePreprocessorConfig(
                **config_dict.get("table_preprocessor", {})
            ),
            paddleocr=PaddleOcrConfig(**config_dict.get("paddleocr", {})),
            paragraph_detector=ParagraphDetectorConfig(
                **config_dict.get("paragraph_detector", {})
            ),
        )
