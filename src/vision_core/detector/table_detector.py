import numpy as np
from loguru import logger

from vision_core.config import TableCellDetectorConfig, TableDetectorConfig, TablePreprocessorConfig
from vision_core.debug_image_observer import DebugImageObserver
from vision_core.detector import helper_table_detector as table_helper
from vision_core.entities.table import Table
from vision_core.preprocessor.table_preprocessor import TablePreprocessor
from vision_core.utils import geometry_utils, image_utils


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(
        self,
        preprocessor_config: TablePreprocessorConfig | None = None,
        table_detector_config: TableDetectorConfig | None = None,
        debug_image: DebugImageObserver | None = None,
    ):
        """
        Инициализация детектора таблиц.

        Args:
            preprocessor_config: Конфигурация для предобработки таблиц.
                Если None, используется конфигурация по умолчанию.
            table_detector_config: Конфигурация для детектора таблиц.
                Если None, используется конфигурация по умолчанию.
            debug_image: Наблюдатель для отладки изображений. Если None, отладка отключена.
        """
        self.cfg = table_detector_config or TableDetectorConfig()
        self.preprocessor = TablePreprocessor(preprocessor_config, debug_image)
        self._debug_image = debug_image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Создаёт маску таблиц для текущего изображения.

        Args:
            image: Изображение страницы.

        Returns:
            mask (np.ndarray): Бинарная маска таблиц.
        """
        binary_image = self.preprocessor.process(image)

        return binary_image

    def detect_tables(self, image: np.ndarray) -> list[Table]:
        """Детектирует таблицы на изображении и извлекает их ячейки.
        Args:
            image: Изображение страницы.
        Returns:
            list[Table]: Список найденных таблиц с их ячейками.
        """
        # Создаем маску таблицы локально для текущего изображения.
        binary_image = self._preprocess(image)
        h_line_mask = image_utils.compute_horizontal_line_mask(binary_image, scale=40)
        v_line_mask = image_utils.compute_vertical_line_mask(binary_image, median_height=40)
        table_mask = image_utils.get_mask(h_line_mask, v_line_mask)

        if self._debug_image:
            self._debug_image.on_debug_image(
                src_image=table_mask,
                stage="4_table_preprocessor",
                prefix="mask",
                page_number=0,
            )

        # Извлекаем bounding boxes таблиц
        table_bboxes = table_helper.extract_raw_tables(
            table_mask,
            border_tol=8,
            scale_width=0.35,
            scale_height=0.08,
            min_density=0.01,
            intersection_over_min_thr=0.2,
        )
        if self._debug_image:
            self._debug_image.on_detected_boxes(
                image=image,
                boxes=[bbox.to_tuple() for bbox in table_bboxes],
                stage="4_table_preprocessor",
                prefix="candidates",
                page_number=0,
            )
        tables: list[Table] = []

        for idx, bbox in enumerate(table_bboxes):
            # Детектируем ячейки внутри таблицы
            roi_mask = bbox.roi(binary_image)
            h_raw_lines = table_helper.extract_raw_horizontal_lines(
                roi_mask,
                scale=20,
                min_line_length=max(1, int(bbox.width // 8)),
                max_line_gap=max(1, int(bbox.width // 10)),
            )

            h_lines = table_helper.extract_lines(
                np.asarray(h_raw_lines, dtype=np.int32),
                table_helper.LineAxis.Y,
                axis_tol=5,
                merge_gap=5,
                min_len=200,
            )
            logger.debug(f"Горизонтальных линий в таблице {idx}: {len(h_lines)}")
            if len(h_lines) == 0:
                logger.debug(f"Пропущена таблица {idx}: не найдено горизонтальных линий")
                continue

            median_height = geometry_utils.median_axis_step(
                np.array([axis for axis, _ in h_lines], dtype=np.int32),
                default=0,
                min_step=1,
            )
            if median_height <= 0:
                logger.debug(f"Пропущена таблица {idx}: не удалось определить медианную высоту строк")
                continue

            logger.debug(f"Медианная высота строк в таблице {idx}: {median_height}")

            v_raw_lines = table_helper.extract_raw_vertical_lines(
                roi_mask,
                median_height=max(1, median_height),
                min_line_length=max(1, median_height),
                max_line_gap=max(1, int(median_height * 0.1)),
            )

            v_lines = table_helper.extract_lines(
                np.asarray(v_raw_lines, dtype=np.int32),
                table_helper.LineAxis.X,
                axis_tol=5,
                merge_gap=5,
                min_len=median_height,
            )
            if len(v_lines) == 0:
                logger.debug(f"Пропущена таблица {idx}: не найдено вертикальных линий")
                continue

            logger.debug(f"Вертикальных линий в таблице {idx}: {len(v_lines)}")
            median_width = geometry_utils.median_axis_step(
                np.array([axis for axis, _ in v_lines], dtype=np.int32),
                default=0,
                min_step=1,
            )
            if median_width <= 0:
                logger.debug(f"Пропущена таблица {idx}: не удалось определить медианную ширину столбцов")
                continue

            logger.debug(f"Медианная ширина столбцов в таблице {idx}: {median_width}")

            x_min, y_min, x_max, y_max = bbox.to_tuple()

            row_ys = table_helper.complete_grid_axes(
                table_helper.axes_to_abs(h_lines, y_min),
                axis_min=y_min,
                axis_max=y_max,
                expected_step=median_height // 2,
            )

            col_xs = table_helper.complete_grid_axes(
                table_helper.axes_to_abs(v_lines, x_min),
                axis_min=x_min,
                axis_max=x_max,
                expected_step=median_width // 2,
            )

            table = table_helper.build_table_from_grid(
                bbox=bbox,
                row_ys=row_ys,
                col_xs=col_xs,
                h_lines_box=h_lines,
                v_lines_box=v_lines,
                merge_mode=self.cfg.mode_merge_cells,
                table_id=f"table_{idx}",
            )

            if not table.is_valid() or not table.validate_structure():
                logger.debug(f"Пропущена таблица {table.id}: невалидная структура")
                continue

            logger.debug(f"Найдена таблица {table.id}: c {table.num_rows} строк и {table.num_cols} столбцов")
            tables.append(table)

        return tables
