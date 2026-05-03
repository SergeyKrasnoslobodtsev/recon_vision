from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImagePreprocessorConfig:
    """Конфигурация для препроцессинга изображений перед распознаванием текста

    Attributes:
        kernel: Размер ядра для размытия при unsharp masking
        sigma: Стандартное отклонение для гауссова размытия при unsharp masking
        amount: Коэффициент усиления для unsharp masking
    """

    kernel: int = 5
    sigma: float = 1.5
    amount: float = 1.0


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
    mode_merge_cells: str | None = "cols"


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
        gamma: Параметр гамма-коррекции для улучшения видимости линий на изображении
        block_size: Размер блока для адаптивного порога
        C: Константа вычитания для адаптивного порога
    """

    gamma: float = 10.0
    block_size: int = 15
    C: int = 5


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


@dataclass
class PaddleOcrConfig:
    """Конфигурация для PaddleOCR.

    Attributes:
        text_recognition_model_name: Название модели для распознавания текста
        text_recognition_model_dir: Директория с моделью для распознавания текста
        text_detection_model_name: Название модели для детекции текста
        text_detection_model_dir: Директория с моделью для детекции текста
        use_doc_orientation_classify: Использовать ли классификатор ориентации документа
        use_doc_unwarping: Использовать ли распрямление документа
        use_textline_orientation: Использовать ли классификатор ориентации текстовой строки
        device: Устройство для выполнения модели (cpu или gpu)
    """

    text_recognition_model_name: str = "cyrillic_PP-OCRv5_mobile_rec"
    text_recognition_model_dir: str = str(_DEFAULT_MODELS_DIR / "cyrillic_PP-OCRv5_mobile_rec")
    text_detection_model_name: str = "PP-OCRv5_server_det"
    text_detection_model_dir: str = str(_DEFAULT_MODELS_DIR / "PP-OCRv5_server_det")
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    device: str = "gpu"


@dataclass
class PageOrientationPreprocessorConfig:
    """Конфигурация препроцессора определения ориентации страницы.

    Attributes:
        model_name: Название модели классификации ориентации документа.
        model_dir: Директория с моделью классификации ориентации документа.
        enabled: Включено ли выравнивание страницы в pipeline.
        min_orientation_score: Минимальный score классификатора ориентации для применения поворота.
        max_skew_deg: Максимальный угол для fine deskew.
        coarse_step_deg: Шаг грубого поиска угла deskew.
        fine_window_deg: Окно точного поиска вокруг лучшего угла грубого этапа.
        fine_step_deg: Шаг точного поиска угла deskew.
        min_abs_deskew_angle_deg: Минимальный по модулю угол, который стоит применять.
    """

    model_name: str = "PP-LCNet_x1_0_doc_ori"
    model_dir: str = str(_DEFAULT_MODELS_DIR / "PP-LCNet_x1_0_doc_ori")
    min_orientation_score: float = 0.6
    # edge rotation detection
    edge_scan_range_deg: float = 45.0  # максимальный угол для сканирования краев
    edge_scan_step_deg: float = 0.5  # шаг между углами для сканирования краев
    edge_scan_deviation_deg: float = 0.5  # максимальное отклонение между углами с разных краев
    edge_scan_size: int = -1  # -1 = весь край
    edge_scan_depth: float = 0.5  # доля ширины/высоты для сканирования


@dataclass
class ParagraphPreprocessorConfig:
    """Конфигурация препроцессора параграфов.

    Attributes:
        adaptive_block_size: Размер блока для адаптивной бинаризации (нечётное число).
        adaptive_c: Константа вычитания для адаптивной бинаризации.
        dilate_kw_ratio: Ширина ядра дилатации как доля median_row_height.
        dilate_kh_ratio: Высота ядра дилатации как доля median_row_height.
        dilate_kw_min: Минимальная ширина ядра дилатации в пикселях.
        dilate_kh_min: Минимальная высота ядра дилатации в пикселях.
        erode_kw: Ширина горизонтального ядра эрозии в пикселях.
        page_bottom_cutoff: Доля высоты страницы, ниже которой регионы отсекаются.
    """

    adaptive_block_size: int = 21
    adaptive_c: int = 20
    dilate_kw_ratio: float = 0.4
    dilate_kh_ratio: float = 0.8
    dilate_kw_min: int = 10
    dilate_kh_min: int = 7
    erode_kw: int = 9
    page_bottom_cutoff: float = 0.9


@dataclass
class ParagraphDetectorConfig:
    """Конфигурация детектора параграфов.

    Attributes:
        reading_order_band_k: Коэффициент для ширины полосы при определении порядка чтения
            (band_tolerance = median_row_height * reading_order_band_k).
    """

    reading_order_band_k: float = 3.0


@dataclass
class VisionCoreConfig:
    """Общая конфигурация для распознавания таблиц"""

    image_preprocessor: ImagePreprocessorConfig = field(default_factory=ImagePreprocessorConfig)
    table_detector: TableDetectorConfig = field(default_factory=TableDetectorConfig)
    cell_detector: TableCellDetectorConfig = field(default_factory=TableCellDetectorConfig)
    table_preprocessor: TablePreprocessorConfig = field(default_factory=TablePreprocessorConfig)

    paddleocr: PaddleOcrConfig = field(default_factory=PaddleOcrConfig)
    page_orientation_preprocessor: PageOrientationPreprocessorConfig = field(
        default_factory=PageOrientationPreprocessorConfig
    )

    paragraph_preprocessor: ParagraphPreprocessorConfig = field(default_factory=ParagraphPreprocessorConfig)
    paragraph_detector: ParagraphDetectorConfig = field(default_factory=ParagraphDetectorConfig)
    dpi: int = 200
    ocr_confidence_threshold: float = 0.5

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionCoreConfig":
        """Создание из словаря (для загрузки из JSON/YAML)"""
        return cls(
            dpi=config_dict.get("dpi", 300),
            ocr_confidence_threshold=config_dict.get("ocr_confidence_threshold", 0.7),
            image_preprocessor=ImagePreprocessorConfig(**config_dict.get("image_preprocessor", {})),
            table_detector=TableDetectorConfig(**config_dict.get("table_detector", {})),
            cell_detector=TableCellDetectorConfig(**config_dict.get("cell_detector", {})),
            table_preprocessor=TablePreprocessorConfig(**config_dict.get("table_preprocessor", {})),
            paddleocr=PaddleOcrConfig(**config_dict.get("paddleocr", {})),
            page_orientation_preprocessor=PageOrientationPreprocessorConfig(
                **config_dict.get("page_orientation_preprocessor", {})
            ),
            paragraph_detector=ParagraphDetectorConfig(**config_dict.get("paragraph_detector", {})),
        )
