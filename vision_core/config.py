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

    min_cell: int = 10
    padding: int = 10
    threshold_line: int = 2
    hough_threshold: int = 50
    min_line_length: int = 10
    max_line_gap: int = 25
    line_angle_threshold: float = 45.0


@dataclass
class TablePreprocessorConfig:
    """Конфигурация препроцессора таблиц"""

    gaussian_blur_kernel: int = 3
    horisontal_kernel_max: int = 50
    horizontal_morph_kernel_size: int = 3
    horizontal_min_length: int = 120
    horizontal_min_intersections: int = 1
    vertical_morph_kernel_size: int = 3
    vertical_kernel_max: int = 250
    vertical_min_length: int = 120
    vertical_min_intersections: int = 2


@dataclass
class TableRecognitionConfig:
    """Общая конфигурация для распознавания таблиц"""

    detector: TableDetectorConfig = field(default_factory=TableDetectorConfig)
    cell_detector: TableCellDetectorConfig = field(
        default_factory=TableCellDetectorConfig
    )
    preprocessor: TablePreprocessorConfig = field(
        default_factory=TablePreprocessorConfig
    )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TableRecognitionConfig":
        """Создание из словаря (для загрузки из JSON/YAML)"""
        return cls(
            detector=TableDetectorConfig(**config_dict.get("detector", {})),
            cell_detector=TableCellDetectorConfig(
                **config_dict.get("cell_detector", {})
            ),
            preprocessor=TablePreprocessorConfig(**config_dict.get("preprocessor", {})),
        )
