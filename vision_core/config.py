from dataclasses import dataclass, field


@dataclass
class TableDetectorConfig:
    """Конфигурация детектора таблиц"""

    min_table_area: int = 10000
    approx_poly_eps: float = 0.02
    mode_merge_cells: str = "cols"


@dataclass
class TableCellDetectorConfig:
    """Конфигурация детектора ячеек"""

    min_cell: int = 10  # Минимальный размер ячейки по умолчанию (10)
    padding: int = 10  # Отступ по умолчанию (10)
    threshold_line: int = 4  # Порог для группировки линий в пикселях (по умолчанию 2)
    coverage_thr: float = 0.8  # Порог покрытия для объединения линий (по умолчанию 0.5)
    hough_threshold: int = 50  # Порог для детектора Хафа чем больше (по умолчанию 50)
    min_line_length: int = (
        8  # Минимальная длина линии для детектора Хафа (по умолчанию 10)
    )
    max_line_gap: int = 25  # Максимальный разрыв между сегментами линии для детектора Хафа (по умолчанию 25)
    line_angle_threshold: float = 45.0  # Порог угла для определения горизонтальных/вертикальных линий (по умолчанию 45.0)


@dataclass
class TablePreprocessorConfig:
    """Конфигурация препроцессора таблиц"""

    gaussian_blur_kernel: int = 1

    horisontal_kernel_max: int = 50  # 50
    horizontal_morph_kernel_size: int = 3
    horizontal_min_length: int = 10
    horizontal_min_intersections: int = 1

    vertical_morph_kernel_size: int = 3
    vertical_kernel_max: int = 80
    vertical_min_length: int = 120  # 120
    vertical_min_intersections: int = 2


@dataclass
class PaddleOcrConfig:
    """Конфигурация для PaddleOCR"""

    text_recognition_model_name = "cyrillic_PP-OCRv5_mobile_rec"
    text_recognition_model_dir = "../models/cyrillic_PP-OCRv5_mobile_rec"
    text_detection_model_name = "PP-OCRv5_server_det"
    text_detection_model_dir = "../models/PP-OCRv5_server_det"
    use_doc_orientation_classify = False
    use_doc_unwarping = False
    use_textline_orientation = False
    device = "cpu"


@dataclass
class VisionCoreConfig:
    """Общая конфигурация для распознавания таблиц"""

    detector: TableDetectorConfig = field(default_factory=TableDetectorConfig)
    cell_detector: TableCellDetectorConfig = field(
        default_factory=TableCellDetectorConfig
    )
    preprocessor: TablePreprocessorConfig = field(
        default_factory=TablePreprocessorConfig
    )

    paddleocr: PaddleOcrConfig = field(default_factory=PaddleOcrConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionCoreConfig":
        """Создание из словаря (для загрузки из JSON/YAML)"""
        return cls(
            detector=TableDetectorConfig(**config_dict.get("detector", {})),
            cell_detector=TableCellDetectorConfig(
                **config_dict.get("cell_detector", {})
            ),
            preprocessor=TablePreprocessorConfig(**config_dict.get("preprocessor", {})),
            paddleocr=PaddleOcrConfig(**config_dict.get("paddleocr", {})),
        )
