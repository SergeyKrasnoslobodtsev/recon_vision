import numpy as np
from loguru import logger
from sklearn.cluster import HDBSCAN

from vision_core.config import ParagraphDetectorConfig
from vision_core.entities.bbox import BBox
from vision_core.entities.paragraph import Paragraph, ParagraphType
from vision_core.ocr.base import OcrResult


class ParagraphDetector:
    """Детектирует текстовые параграфы на изображении через кластеризацию OCR-боксов.

    Использует DBSCAN с анизотропной нормализацией признаков: вертикальное
    расстояние нормируется на медианную высоту строки, горизонтальное — на
    характерную ширину страницы. Это позволяет корректно объединять близкие
    строки в абзацы и разделять колонки.
    """

    def __init__(self, config: ParagraphDetectorConfig | None = None):
        """Инициализирует детектор с конфигурацией или значениями по умолчанию.

        Args:
            config: Конфигурация детектора (eps, min_samples и пр.).
        """
        self.cfg = config or ParagraphDetectorConfig()
        logger.debug(f"ParagraphDetector инициализирован: eps=, min_samples={self.cfg.min_cluster_size}")

    def detect_paragraphs(
        self,
        ocr_results: list[OcrResult],
        image_shape: tuple[int, int],
    ) -> list[Paragraph]:
        """Детектирует параграфы: кластеризация OCR-боксов, слияние, классификация.

        Args:
            ocr_results: Сырые OCR-результаты.
            image_shape: Размер изображения (height, width).

        Returns:
            Список параграфов с типом и объединённым bbox.

        Raises:
            ValueError: Некорректный image_shape.
        """
        if not ocr_results:
            logger.warning("OCR-результаты пусты, параграфы не обнаружены")
            return []

        if len(image_shape) != 2 or any(s <= 0 for s in image_shape):
            raise ValueError(f"Некорректный image_shape: {image_shape}")

        logger.debug(f"Начата детекция параграфов для {len(ocr_results)} OCR-результатов")

        clusters = self._clusterize(ocr_results, image_shape)
        clusters = self._merge_nested_clusters(clusters)

        clusters_sorted = sorted(
            clusters,
            key=lambda cluster: min(item.bbox[1] for item in cluster),
        )

        paragraphs: list[Paragraph] = []
        for idx, cluster in enumerate(clusters_sorted, start=1):
            paragraph = self._cluster_to_paragraph(cluster, paragraph_id=str(idx))
            paragraph.type = self._classify_paragraph_type(
                paragraph,
                image_height=image_shape[0],
                image_width=image_shape[1],
            )
            paragraphs.append(paragraph)

        logger.info(
            f"Обнаружено параграфов: {len(paragraphs)} "
            f"(HEADER={sum(p.type == ParagraphType.HEADER for p in paragraphs)}, "
            f"FOOTER={sum(p.type == ParagraphType.FOOTER for p in paragraphs)}, "
            f"TEXT={sum(p.type == ParagraphType.TEXT for p in paragraphs)})"
        )
        return paragraphs

    # ------------------------------------------------------------------ #
    # Кластеризация
    # ------------------------------------------------------------------ #

    def _clusterize(
        self,
        ocr_results: list[OcrResult],
        image_shape: tuple[int, int],
    ) -> list[list[OcrResult]]:
        """Кластеризует OCR-боксы через HDBSCAN с анизотропной нормализацией.

        Признаки:
        - Левый и правый край bbox (x_min, x_max).
        - Центр строки (cx, cy).
        - Высота строки.
        - Уверенность OCR как слабый дополнительный сигнал.

        Нормализация:
        - Горизонтальные признаки: делим на 10% ширины страницы.
        - Вертикальные признаки: делим на медианную высоту строки.

        Args:
            ocr_results: Список OCR-строк.
            image_shape: (height, width) изображения.

        Returns:
            Список кластеров OCR-строк.
        """
        min_cluster_size = max(1, int(self.cfg.min_cluster_size))
        if len(ocr_results) < min_cluster_size:
            logger.debug(
                "Недостаточно OCR-результатов для HDBSCAN: "
                f"n_samples={len(ocr_results)}, min_cluster_size={min_cluster_size}. "
                "Возвращаем отдельные боксы."
            )
            return [[item] for item in ocr_results]

        features, scale_y, scale_x = self._build_features(ocr_results, image_shape)
        if features.size == 0 or scale_y <= 0 or scale_x <= 0:
            logger.warning("Невалидные признаки или масштаб, возвращаем отдельные боксы")
            return [[item] for item in ocr_results]

        logger.debug(
            f"Нормализация: scale_y (медиана высоты)={scale_y:.2f}, scale_x (0.1*ширина страницы)={scale_x:.2f}"
        )

        clustering = HDBSCAN(
            copy=False,
            min_cluster_size=self.cfg.min_cluster_size,
            metric="euclidean",
        ).fit(features)

        labels = clustering.labels_
        clusters: dict[int, list[OcrResult]] = {}
        for ocr, lbl in zip(ocr_results, labels, strict=False):
            clusters.setdefault(lbl, []).append(ocr)

        result: list[list[OcrResult]] = []
        for label, cluster in clusters.items():
            if label == -1:
                # Шумовые одиночки остаются отдельными параграфами
                result.extend([[item] for item in cluster])
            else:
                result.append(cluster)

        logger.debug(f"HDBSCAN: обнаружено {len(result)} кластеров (вкл. шум)")
        return result

    def _build_features(
        self,
        ocr_results: list[OcrResult],
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, float, float]:
        """Строит матрицу признаков с анизотропной нормализацией.

        Вектор признаков на строку:
        [x_min_norm, x_max_norm, cx_norm, cy_norm, h_norm, confidence],
        где:
        - x_min_norm и x_max_norm отражают геометрию левого и правого края строки;
        - cx_norm и cy_norm отражают положение центра строки;
        - h_norm отражает вертикальную близость строк через масштаб высоты;
        - confidence оставляется ненормализованным как слабый вспомогательный признак.

        Args:
            ocr_results: Список OCR-строк.
            image_shape: (height, width) изображения.

        Returns:
            np.ndarray: Матрица признаков формы (n, 6).
            float: scale_y (медиана высоты строки).
            float: scale_x (эвристика для горизонтали).
        """
        if not ocr_results:
            return np.empty((0, 6), dtype=float), 0.0, 0.0

        _, image_width = image_shape
        heights: list[float] = []
        rows: list[list[float]] = []

        for item in ocr_results:
            x_min, y_min, x_max, y_max = item.bbox
            h = max(0.0, y_max - y_min)
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            heights.append(h)
            rows.append([x_min, x_max, cx, cy, h, item.confidence])

        median_h = float(np.median(heights)) if heights else 1.0
        scale_y = median_h if median_h > 0 else 1.0

        # Эвристика: 10% ширины страницы как характерный горизонтальный масштаб
        scale_x = 0.1 * image_width if image_width > 0 else 1.0

        features = np.array(rows, dtype=float)
        # Учитываем и границы, и центр строки, чтобы лучше различать выравнивание абзацев.
        features[:, [0, 1, 2]] /= scale_x
        features[:, [3, 4]] /= scale_y

        return features, scale_y, scale_x

    # ------------------------------------------------------------------ #
    # Слияние вложенных кластеров
    # ------------------------------------------------------------------ #

    def _merge_nested_clusters(
        self,
        clusters: list[list[OcrResult]],
    ) -> list[list[OcrResult]]:
        """Объединяет кластеры, если центр bbox одного лежит внутри bbox другого.

        Используется union-find для транзитивного слияния. Выбирается минимальный
        по площади внешний контейнер, чтобы избежать переслияния.

        Args:
            clusters: Список кластеров OCR-строк.

        Returns:
            Список кластеров после слияния.
        """
        if not clusters:
            return []

        cluster_bboxes = [self._cluster_bbox(cluster) for cluster in clusters]
        parents = list(range(len(clusters)))

        def find(i: int) -> int:
            """Находит корень компоненты связности с компрессией пути."""
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def union(child: int, parent: int) -> None:
            """Объединяет компоненты child и parent."""
            parents[find(child)] = find(parent)

        for inner_idx, inner_bbox in enumerate(cluster_bboxes):
            candidates: list[tuple[float, int]] = []
            for outer_idx, outer_bbox in enumerate(cluster_bboxes):
                if inner_idx == outer_idx:
                    continue
                if outer_bbox.area <= inner_bbox.area:
                    continue
                if outer_bbox.contains_center(inner_bbox):
                    candidates.append((outer_bbox.area, outer_idx))
            if candidates:
                _, best_outer = min(candidates, key=lambda item: item[0])
                union(inner_idx, best_outer)

        merged: dict[int, list[OcrResult]] = {}
        for idx, cluster in enumerate(clusters):
            root = find(idx)
            merged.setdefault(root, []).extend(cluster)

        logger.debug(f"Слияние вложенных: {len(clusters)} -> {len(merged)} кластеров")
        return [merged[key] for key in sorted(merged.keys())]

    def _cluster_bbox(self, cluster: list[OcrResult]) -> BBox:
        """Возвращает объединённый bbox кластера.

        Args:
            cluster: Список OCR-строк в кластере.

        Returns:
            BBox, охватывающий все строки кластера.
        """
        boxes = [BBox.from_tuple(ocr.bbox) for ocr in cluster]
        return BBox(
            x_min=min(b.x_min for b in boxes),
            y_min=min(b.y_min for b in boxes),
            x_max=max(b.x_max for b in boxes),
            y_max=max(b.y_max for b in boxes),
        )

    # ------------------------------------------------------------------ #
    # Преобразование и классификация
    # ------------------------------------------------------------------ #

    def _cluster_to_paragraph(
        self,
        cluster: list[OcrResult],
        paragraph_id: str,
    ) -> Paragraph:
        """Создаёт Paragraph из кластера OCR-боксов.

        Args:
            cluster: Список OCR-строк.
            paragraph_id: Уникальный идентификатор параграфа.

        Returns:
            Paragraph с объединённым текстом, bbox и списком исходных блобов.
        """
        texts = [item.text for item in cluster]
        bbox = self._cluster_bbox(cluster)
        blobs = [BBox.from_tuple(item.bbox) for item in cluster]
        return Paragraph(
            id=paragraph_id,
            text="\n".join(texts),
            type=ParagraphType.TEXT,
            bbox=bbox,
            blobs=blobs,
        )

    def _classify_paragraph_type(
        self,
        paragraph: Paragraph,
        image_height: int,
        image_width: int,
    ) -> ParagraphType:
        """Классифицирует параграф по положению и относительному размеру.

        Логика:
        - Если параграф большой (> 60% ширины и > 40% высоты) -> TEXT.
        - Если в верхних 15% страницы -> HEADER.
        - Если в нижних 15% страницы -> FOOTER.
        - Иначе -> TEXT.

        Args:
            paragraph: Параграф для классификации.
            image_height: Высота изображения.
            image_width: Ширина изображения.

        Returns:
            ParagraphType (HEADER, FOOTER или TEXT).
        """
        bbox = paragraph.bbox
        width = bbox.x_max - bbox.x_min
        height = bbox.y_max - bbox.y_min

        if width > 0.6 * image_width and height > 0.4 * image_height:
            return ParagraphType.TEXT

        if paragraph.is_at_top(image_height, threshold=0.15):
            return ParagraphType.HEADER
        if paragraph.is_at_bottom(image_height, threshold=0.15):
            return ParagraphType.FOOTER

        return ParagraphType.TEXT
