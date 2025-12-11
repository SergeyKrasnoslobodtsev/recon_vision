import numpy as np
import cv2
from vision_core.entities.bbox import BBox
from vision_core.preprocessor.table_preprocessor import TablePreprocessor


class TableDetector:
    """Детектор таблиц на изображении"""

    def __init__(self, min_table_area: int = 10000):
        """
        Args:
            min_table_area: Минимальная площадь таблицы
        """
        self.min_table_area = min_table_area
        self.preprocessor = TablePreprocessor()

    def extract_tables(self, image: np.ndarray):
        table_mask = self.preprocessor.create_table_mask(image)

        return self._find_tables(table_mask)

    def _find_tables(self, table_mask: np.ndarray):
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes: list[BBox] = []
        for cnt in contours:
            arclen = cv2.arcLength(cnt, True)
            eps = 0.02 * arclen
            c_poly = cv2.approxPolyDP(cnt, eps, True)
            if len(c_poly) != 4:
                continue
            if cv2.contourArea(c_poly) < self.min_table_area:
                continue
            pts = c_poly.reshape(4, 2)
            ordered_pts = self._order_points(pts)
            x_min, y_min = ordered_pts[:, 0].min(), ordered_pts[:, 1].min()
            x_max, y_max = ordered_pts[:, 0].max(), ordered_pts[:, 1].max()
            bboxes.append(BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

        return bboxes

    def _order_points(self, pts):
        from scipy.spatial import distance as dist

        xSorted = pts[np.argsort(pts[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype="float32")
