from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


@dataclass
class PageOrientationPreprocessorConfig:
    """Конфигурация препроцессора определения ориентации страницы.

    Attributes:
        model_name (str): Название модели классификации ориентации документа.
        model_dir (str): Директория с моделью классификации ориентации документа.
        min_orientation_score (float): Минимальный score классификатора ориентации для применения поворота.
        gamma (float): Параметр гамма-коррекции для улучшения видимости линий на изображении.
        block_size (int): Размер блока для адаптивного порога при предобработке изображения перед классификацией ориентации.
        C (int): Константа вычитания для адаптивного порога при предобработке изображения перед классификацией ориентации.
        scale_horizontal_line (int): Минимальная длина горизонтальной линии для учёта при вычислении угла наклона страницы.
        height_vertical_line (int): Минимальная длина вертикальной линии для учёта при вычислении угла наклона страницы.

        Note:
            Параметры `scale_horizontal_line` и `height_vertical_line` используются для определения кандидатов таблиц.
            Для расчета угла наклона страницы используется вырезанное бинарное изображение, на котором повторно
            используется алгоритм поиска горизонтальных линий с параметром `scale = 10`

    """

    model_name: str = "PP-LCNet_x1_0_doc_ori"
    model_dir: str = str(_DEFAULT_MODELS_DIR / "PP-LCNet_x1_0_doc_ori")
    min_orientation_score: float = 0.79

    gamma: float = 10.0
    block_size: int = 15
    C: int = 5
    scale_horizontal_line: int = 40
    height_vertical_line: int = 40


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
class TablePreprocessorConfig:
    """Конфигурация препроцессора таблиц

    Attributes:
        gamma: Параметр гамма-коррекции для улучшения видимости линий на изображении
        block_size: Размер блока для адаптивного порога
        C: Константа вычитания для адаптивного порога
    """

    gamma: float = 10.0
    block_size: int = 11
    C: int = 5


@dataclass
class TableDetectorConfig:
    """Конфигурация детектора таблиц

    Attributes:


        mode_merge_cells: Режим объединения ячеек поддерживает:
        - all - объединяет строки и столбцы
        - cols - объединяет только столбцы
        - rows - объединяет только строки
        - None - не объединяет ячейки
    """

    min_area_frac: float = 0.005
    min_w: float = 50
    min_h: float = 30
    min_overlap_ratio: float = 0.5

    # ## detected raw table candidates
    # scale_horizontal_line: int = 40
    # height_vertical_line: int = 40
    # border_tol: int = 8
    # scale_width: float = 0.35
    # scale_height: float = 0.08
    # min_density: float = 0.01
    # intersection_over_min_thr: float = 0.2

    # # detected raw horizontal lines
    # scale_horizontal_line: int = 20
    # min_line_length_ratio: int = 8
    # max_line_gap_ratio: int = 10

    # # axis horizontal lines
    # h_axis_tol: int = 5
    # h_merge_gap: int = 5
    # h_min_line_length: int = 200

    # # detected raw vertical lines
    # max_line_gap_ratio_v: float = 0.1
    # v_axis_tol: int = 5
    # v_merge_gap: int = 5

    mode_merge_cells: str | None = "cols"


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
    table_preprocessor: TablePreprocessorConfig = field(default_factory=TablePreprocessorConfig)

    paddleocr: PaddleOcrConfig = field(default_factory=PaddleOcrConfig)
    page_orientation_preprocessor: PageOrientationPreprocessorConfig = field(
        default_factory=PageOrientationPreprocessorConfig
    )

    paragraph_preprocessor: ParagraphPreprocessorConfig = field(default_factory=ParagraphPreprocessorConfig)
    paragraph_detector: ParagraphDetectorConfig = field(default_factory=ParagraphDetectorConfig)
    dpi: int = 200
    ocr_confidence_threshold: float = 0.45

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionCoreConfig":
        """Создание из словаря (для загрузки из JSON/YAML)"""
        return cls(
            dpi=config_dict.get("dpi", 200),
            ocr_confidence_threshold=config_dict.get("ocr_confidence_threshold", 0.45),
            image_preprocessor=ImagePreprocessorConfig(**config_dict.get("image_preprocessor", {})),
            table_detector=TableDetectorConfig(**config_dict.get("table_detector", {})),
            table_preprocessor=TablePreprocessorConfig(**config_dict.get("table_preprocessor", {})),
            paddleocr=PaddleOcrConfig(**config_dict.get("paddleocr", {})),
            page_orientation_preprocessor=PageOrientationPreprocessorConfig(
                **config_dict.get("page_orientation_preprocessor", {})
            ),
            paragraph_detector=ParagraphDetectorConfig(**config_dict.get("paragraph_detector", {})),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VisionCoreConfig":
        """Загрузка конфигурации из YAML файла"""
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "VisionCoreConfig":
        """Загрузка конфигурации из JSON файла"""
        import json

        with open(json_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def save_to_yaml(cls, config: "VisionCoreConfig", yaml_path: str):
        """Сохранение конфигурации в YAML файл"""
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(config.__dict__, f, default_flow_style=False)

    @classmethod
    def save_to_json(cls, config: "VisionCoreConfig", json_path: str):
        """Сохранение конфигурации в JSON файл"""
        import json

        with open(json_path, "w") as f:
            json.dump(config.__dict__, f, indent=4)
