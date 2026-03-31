from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImagePreprocessorConfig:
    """Конфигурация для препроцессинга изображений перед распознаванием текста

    Attributes:
        denoise_h: Параметр для удаления шума чем выше, тем сильнее удаление (0 - без удаления, 3-5 - легкое удаление)
        clip_limit: Порог для контрастного ограничителя CLAHE (1.0 - без усиления, 2.0 - сильное усиление)
        tile_size: Размер тайла для CLAHE (меньше - более локальный контраст)
        kernel_size: Размер ядра для морфологических операций для усиления светлых штрихов (3-5 обычно достаточно)
        blackhat_gain: Коэффициент усиления для операции blackhat (0.2-0.3 может помочь, но зависит от качества скана)
        kernel_size_morph: Размер ядра для морфологических операций для оценки фона (обычно 21)
    """

    denoise_h: int = 3
    clip_limit: float = 1.9
    tile_size: int = 8
    kernel_size: int = 5
    blackhat_gain: float = 0.3
    kernel_size_morph: int = 21


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
    min_orientation_score: float = 0.7
    max_skew_deg: float = 10.0
    coarse_step_deg: float = 0.5
    fine_window_deg: float = 0.5
    fine_step_deg: float = 0.1
    min_abs_deskew_angle_deg: float = 0.05


@dataclass
class ParagraphDetectorConfig:
    """Конфигурация детектора параграфов.

    Attributes:
        min_cluster_size: Минимальный размер кластера для объединения выделенных слов в параграф
    """

    min_cluster_size: int = 3


@dataclass
class VisionCoreConfig:
    """Общая конфигурация для распознавания таблиц"""

    table_detector: TableDetectorConfig = field(default_factory=TableDetectorConfig)
    cell_detector: TableCellDetectorConfig = field(default_factory=TableCellDetectorConfig)
    table_preprocessor: TablePreprocessorConfig = field(default_factory=TablePreprocessorConfig)

    paddleocr: PaddleOcrConfig = field(default_factory=PaddleOcrConfig)
    page_orientation_preprocessor: PageOrientationPreprocessorConfig = field(
        default_factory=PageOrientationPreprocessorConfig
    )

    paragraph_detector: ParagraphDetectorConfig = field(default_factory=ParagraphDetectorConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionCoreConfig":
        """Создание из словаря (для загрузки из JSON/YAML)"""
        return cls(
            table_detector=TableDetectorConfig(**config_dict.get("table_detector", {})),
            cell_detector=TableCellDetectorConfig(**config_dict.get("cell_detector", {})),
            table_preprocessor=TablePreprocessorConfig(**config_dict.get("table_preprocessor", {})),
            paddleocr=PaddleOcrConfig(**config_dict.get("paddleocr", {})),
            page_orientation_preprocessor=PageOrientationPreprocessorConfig(
                **config_dict.get("page_orientation_preprocessor", {})
            ),
            paragraph_detector=ParagraphDetectorConfig(**config_dict.get("paragraph_detector", {})),
        )
