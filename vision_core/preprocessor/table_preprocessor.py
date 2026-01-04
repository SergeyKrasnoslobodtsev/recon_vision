import cv2
import numpy as np
from typing import Optional
from vision_core.config import TablePreprocessorConfig

from loguru import logger


class TablePreprocessor:
    def __init__(self, cfg: Optional[TablePreprocessorConfig] = None):
        """Предобработчик для таблиц

        Args:
            cfg: Конфигурация предобработчика таблиц
        """
        if cfg is None:
            cfg = TablePreprocessorConfig()

        self.kernel_gauss = cfg.gaussian_blur_kernel

    def create_table_mask(self, image: np.ndarray):
        """Создание маски таблицы из изображения"""
        processed = self._processing(image)
        # 0.005 соотношение в 2 раза меньше чем горизонтальная линия
        # остается больше ложных линий, но потом мы их удалим с помощью clean_mask
        min_lenght_h = int(processed.shape[0] * 0.03)
        min_lenght_v = int(processed.shape[1] * 0.006)
        logger.debug(
            f"Min lengths - Horizontal: {min_lenght_h}, Vertical: {min_lenght_v}"
        )
        h_mask = self._detect_horizontal_lines(processed, min_lenght_h)
        v_mask = self._detect_vertical_lines(processed, min_lenght_v)
        table_mask = self._create_table_mask(h_mask, v_mask, min_lenght_h, min_lenght_v)
        return table_mask

    def _processing(self, image: np.ndarray):
        blur = cv2.GaussianBlur(
            image,
            (self.kernel_gauss, self.kernel_gauss),
            0,
        )
        binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1
        ]
        return binary

    def _create_table_mask(
        self,
        h_mask: np.ndarray,
        v_mask: np.ndarray,
        min_length_h: int,
        min_length_v: int,
    ):
        """
        Создает маску таблицы путем комбинирования масок горизонтальных и вертикальных линий.
        Метод обрабатывает области, похожие на таблицы, выполняя следующие шаги:
        1. Находит пересечения между горизонтальными и вертикальными масками
        2. Определяет области, похожие на таблицы, используя ограничивающие прямоугольники
        3. Очищает каждую область индивидуально для удаления шума
        4. Объединяет очищенные горизонтальные и вертикальные маски
        Такой подход улучшает производительность за счет обработки только релевантных областей
        и закрашивания остальной части изображения черным цветом, что ускоряет операцию _clean_mask.

        Args:
            h_mask (np.ndarray): Бинарная маска, содержащая обнаруженные горизонтальные линии
            v_mask (np.ndarray): Бинарная маска, содержащая обнаруженные вертикальные линии
            min_length_h (int): Минимальная длина горизонтальных линий для фильтрации
            min_length_v (int): Минимальная длина вертикальных линий для фильтрации

        Returns:
            np.ndarray: Очищенная бинарная маска, содержащая обнаруженную структуру таблицы,
                        где линии таблицы белые, а фон черный
        """

        intersec = cv2.bitwise_and(h_mask, v_mask)
        mask = cv2.bitwise_or(h_mask, v_mask)
        regions = self._raw_bouding_boxes(mask)

        cleaned_h_mask = np.zeros_like(h_mask, dtype=np.uint8)
        cleaned_v_mask = np.zeros_like(v_mask, dtype=np.uint8)

        for x, y, w, h in regions:
            h_roi = h_mask[y : y + h, x : x + w]
            v_roi = v_mask[y : y + h, x : x + w]
            intersec_roi = intersec[y : y + h, x : x + w]

            # установим 1 пересечени, так как есть таблицы
            # где линии с одним пересечением
            # для вертикальных линий так нельзя делать
            # так как буквы будут давать ложные линии и в дальнейшем
            # делить ячейку по ложной линии
            cleaned_h = self._clean_mask(
                h_roi,
                intersec_roi,
                min_length=min_length_h,
                min_intersections=1,
                type="horizontal",
            )

            cleaned_v = self._clean_mask(
                v_roi,
                intersec_roi,
                min_length=min_length_v,
                min_intersections=2,
                type="vertical",
            )

            cleaned_h_mask[y : y + h, x : x + w] = cleaned_h
            cleaned_v_mask[y : y + h, x : x + w] = cleaned_v

        cleaned_mask = cv2.add(cleaned_h_mask, cleaned_v_mask)
        return cleaned_mask

    def _clean_mask(
        self,
        mask: np.ndarray,
        intersec: np.ndarray,
        min_length: int = 120,
        min_intersections: int = 2,
        type: str = "horizontal",
    ):
        """
        Метод фильтрации линий на маске.

        Args:
            mask (np.ndarray): Бинарная маска линий (горизонтальных или вертикальных).
            intersec (np.ndarray): Маска точек пересечения горизонтальных и вертикальных линий.
            min_length (int, optional): Минимальный размер связной компоненты (площадь/«длина» линии) для сохранения. Defaults to 120.
            min_intersections (int, optional): Минимальное количество пересечений на концах линии для её сохранения. Defaults to 2.
            type (str, optional): Ориентация линии: "horizontal" или "vertical". Defaults to "horizontal".

        Returns:
            np.ndarray: Очищенная бинарная маска линий после фильтрации по длине и количеству пересечений.
        """
        num_mask_labels, labels_mask, stats_mask, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        output_mask = np.zeros_like(mask, dtype=np.uint8)

        for lbl in range(1, num_mask_labels):
            component_area = stats_mask[lbl, cv2.CC_STAT_AREA]

            # Пропускаем слишком маленькие по площади компоненты (слишком короткие линии)
            if component_area < min_length:
                continue
            # TODO: оптимизировать алгоритм передать сырые квадрты
            comp_boolean_mask = labels_mask == lbl
            x, y, w, h = cv2.boundingRect(comp_boolean_mask.astype(np.uint8))

            # if np.sum(intersec[y : y + h, x : x + w]) == 0:
            #     continue

            # Если пересечения с двух концов линии есть, сохраняем компонент
            if type == "horizontal":
                left = np.any(
                    intersec[
                        comp_boolean_mask
                        & (np.arange(mask.shape[1]) < x + w // 3)[None, :]
                    ]
                )
                right = np.any(
                    intersec[
                        comp_boolean_mask
                        & (np.arange(mask.shape[1]) > x + 2 * w // 3)[None, :]
                    ]
                )
                crosses = int(left) + int(right)
            else:  # vertical
                top = np.any(
                    intersec[
                        comp_boolean_mask
                        & (np.arange(mask.shape[0]) < y + h // 3)[:, None]
                    ]
                )
                bottom = np.any(
                    intersec[
                        comp_boolean_mask
                        & (np.arange(mask.shape[0]) > y + 2 * h // 3)[:, None]
                    ]
                )
                crosses = int(top) + int(bottom)

            if crosses >= min_intersections:
                output_mask[comp_boolean_mask] = 255

        return output_mask

    def _raw_bouding_boxes(self, mask: np.ndarray):
        h, w = mask.shape
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        valid_boxes: list[tuple[int, int, int, int]] = []

        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)

            # гипотиза в документах все таблицы расположены по всей ширине
            # мы можем определить что таблица не может быть меньше половины ширины изображения
            # также мы понимаем, что высота таблицы не может быть меньше 5% от ее ширины
            # w = 100 h = 10

            if box_w < 0.5 * w:
                continue

            if box_h < 0.05 * box_w:
                continue

            valid_boxes.append((x, y, box_w, box_h))

        return valid_boxes

    def _detect_horizontal_lines(self, binary: np.ndarray, kernel_width: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        reconstructed_horizontal = cv2.dilate(horizontal, kernel_dilate, iterations=2)

        return reconstructed_horizontal

    def _detect_vertical_lines(self, binary: np.ndarray, kernel_height: int):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        reconstructed_vertical = cv2.dilate(vertical, kernel_dilate, iterations=2)

        return reconstructed_vertical
