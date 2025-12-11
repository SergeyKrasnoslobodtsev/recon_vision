import cv2
import numpy as np


class TablePreprocessor:
    def __init__(self):
        pass

    def create_table_mask(self, image: np.ndarray):
        processed = self._processing(image)

        h_mask = self._detect_horizontal_lines(processed)
        v_mask = self._detect_vertical_lines(processed)
        table_mask = self._create_table_mask(h_mask, v_mask)
        return table_mask

    def _processing(self, image: np.ndarray):
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return binary

    def _create_table_mask(self, h_mask: np.ndarray, v_mask: np.ndarray):
        # удаляем шумы
        intersec = cv2.bitwise_and(h_mask, v_mask)
        cleaned_h_mask = self._clean_mask(h_mask, intersec, type="horizontal")
        cleaned_v_mask = self._clean_mask(v_mask, intersec, type="vertical")
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

        for lbl in range(
            1, num_mask_labels
        ):  # Начинаем с 1, чтобы пропустить фон (метка 0)
            length = stats_mask[lbl, cv2.CC_STAT_AREA]
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

            # current_comp_intersections_map_bool = comp_boolean_mask & (intersec > 0)

            # num_intersection_pixels = np.sum(current_comp_intersections_map_bool)

            # if not np.any(current_comp_intersections_map_bool):
            #     crosses = 0
            # else:
            #     num_intersection_blobs, _ = cv2.connectedComponents(
            #         current_comp_intersections_map_bool.astype(np.uint8), connectivity=8
            #     )
            #     crosses = num_intersection_blobs - 1

            # if crosses > min_intersections:
            #     output_mask[comp_boolean_mask] = 255

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
