import cv2
import numpy as np
from typing import Optional
from vision_core.config import TablePreprocessorConfig


class TablePreprocessor:
    def __init__(self, cfg: Optional[TablePreprocessorConfig] = None):
        """Предобработчик для таблиц

        Args:
            cfg: Конфигурация предобработчика таблиц
        """
        if cfg is None:
            cfg = TablePreprocessorConfig()

        self.gaussian_blur_kernel = cfg.gaussian_blur_kernel
        self.horisontal_kernel_max = cfg.horisontal_kernel_max
        self.horizontal_morph_kernel_size = cfg.horizontal_morph_kernel_size
        self.horizontal_min_length = cfg.horizontal_min_length
        self.horizontal_min_intersections = cfg.horizontal_min_intersections
        self.vertical_morph_kernel_size = cfg.vertical_morph_kernel_size
        self.vertical_kernel_max = cfg.vertical_kernel_max
        self.vertical_min_length = cfg.vertical_min_length
        self.vertical_min_intersections = cfg.vertical_min_intersections

    def create_table_mask(self, image: np.ndarray):
        """Создание маски таблицы из изображения"""
        processed = self._processing(image)

        h_mask = self._detect_horizontal_lines(processed, self.horisontal_kernel_max)
        v_mask = self._detect_vertical_lines(processed, self.vertical_kernel_max)
        table_mask = self._create_table_mask(h_mask, v_mask)
        return table_mask

    def _processing(self, image: np.ndarray):
        blur = cv2.GaussianBlur(
            image,
            (self.gaussian_blur_kernel, self.gaussian_blur_kernel),
            0,
        )
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return binary

    def _create_table_mask(self, h_mask: np.ndarray, v_mask: np.ndarray):
        # удаляем шумы
        intersec = cv2.bitwise_and(h_mask, v_mask)
        cleaned_h_mask = self._clean_mask(
            h_mask,
            intersec,
            min_length=self.horizontal_min_length,
            min_intersections=self.horizontal_min_intersections,
            type="horizontal",
        )
        cleaned_v_mask = self._clean_mask(
            v_mask,
            intersec,
            min_length=self.vertical_min_length,
            min_intersections=self.vertical_min_intersections,
            type="vertical",
        )
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
        num_mask_labels, labels_mask, stats_mask, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        output_mask = np.zeros_like(mask, dtype=np.uint8)

        for lbl in range(1, num_mask_labels):
            length = stats_mask[lbl, cv2.CC_STAT_AREA]

            # Пропускаем слишком короткие линии
            if length < min_length:
                continue
            # TODO: optimize this part
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

    def _detect_horizontal_lines(self, binary: np.ndarray, kernel_max: int = 50):
        h, w = binary.shape
        kernel_width = max(10, min(50, w // kernel_max))  # не меньше 10, не больше 50
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        reconstructed_horizontal = cv2.dilate(horizontal, kernel_dilate, iterations=2)

        return reconstructed_horizontal

    def _detect_vertical_lines(self, binary: np.ndarray, kernel_max: int = 250):
        h, w = binary.shape
        kernel_height = max(5, min(50, h // kernel_max))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        reconstructed_vertical = cv2.dilate(vertical, kernel_dilate, iterations=2)

        return reconstructed_vertical
