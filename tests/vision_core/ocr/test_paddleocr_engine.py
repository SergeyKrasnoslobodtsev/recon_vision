import sys
import types
import importlib

import numpy as np
import pytest

from vision_core.config import VisionCoreConfig


def _install_fake_paddleocr_module(monkeypatch, fake_cls):
    """Ensures `from paddleocr import PaddleOCR` works even if paddleocr isn't installed."""
    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = fake_cls
    monkeypatch.setitem(sys.modules, "paddleocr", mod)


def _make_vc_config_with_dirs(det_dir: str, rec_dir: str) -> VisionCoreConfig:
    cfg = VisionCoreConfig()
    cfg.paddleocr.text_detection_model_dir = det_dir
    cfg.paddleocr.text_recognition_model_dir = rec_dir
    return cfg


def test_init_raises_when_detection_dir_missing(monkeypatch, tmp_path):
    class FakePaddleOCR:
        def __init__(self, **kwargs):  # pragma: no cover
            raise AssertionError(
                "PaddleOCR ctor must not be called when path checks fail"
            )

        def predict_iter(self, images):
            yield from ()

    _install_fake_paddleocr_module(monkeypatch, FakePaddleOCR)

    from vision_core.ocr import paddle_ocr as paddle_ocr_mod

    importlib.reload(paddle_ocr_mod)

    rec_dir = tmp_path / "rec"
    rec_dir.mkdir(parents=True, exist_ok=True)
    det_dir = tmp_path / "det_missing"

    cfg = _make_vc_config_with_dirs(str(det_dir), str(rec_dir))

    with pytest.raises(
        FileNotFoundError, match=r"Text detection model directory not found"
    ):
        paddle_ocr_mod.PaddleOcrEngine(cfg)


def test_init_raises_when_recognition_dir_missing(monkeypatch, tmp_path):
    class FakePaddleOCR:
        def __init__(self, **kwargs):  # pragma: no cover
            raise AssertionError(
                "PaddleOCR ctor must not be called when path checks fail"
            )

        def predict_iter(self, images):
            yield from ()

    _install_fake_paddleocr_module(monkeypatch, FakePaddleOCR)

    from vision_core.ocr import paddle_ocr as paddle_ocr_mod

    importlib.reload(paddle_ocr_mod)

    # Директория распознавания отсутствует, директория детекции существует
    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = tmp_path / "rec_missing"

    cfg = _make_vc_config_with_dirs(str(det_dir), str(rec_dir))

    with pytest.raises(
        FileNotFoundError, match=r"Text recognition model directory not found"
    ):
        paddle_ocr_mod.PaddleOcrEngine(cfg)


def test_predict_iter_accepts_single_ndarray_and_yields_ocrresults(
    monkeypatch, tmp_path
):
    captured = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured["ctor_kwargs"] = kwargs

        def predict_iter(self, images):
            assert isinstance(images, list)
            assert len(images) == 1
            yield {
                "rec_boxes": [[1, 2, 3, 4]],
                "rec_texts": ["hello"],
                "rec_scores": [0.9],
            }

    _install_fake_paddleocr_module(monkeypatch, FakePaddleOCR)

    from vision_core.ocr import paddle_ocr as paddle_ocr_mod

    importlib.reload(paddle_ocr_mod)

    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_vc_config_with_dirs(str(det_dir), str(rec_dir))
    eng = paddle_ocr_mod.PaddleOcrEngine(cfg)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    batch = next(eng.predict_iter(img))

    assert isinstance(batch, list)
    assert len(batch) == 1
    assert batch[0].text == "hello"
    assert batch[0].confidence == 0.9
    assert batch[0].bboxes == (1, 2, 3, 4)

    # Проверим есть параметры инициализации PaddleOCR из конфигурации
    ctor = captured["ctor_kwargs"]
    assert (
        ctor["text_recognition_model_name"] == cfg.paddleocr.text_recognition_model_name
    )
    assert (
        ctor["text_recognition_model_dir"] == cfg.paddleocr.text_recognition_model_dir
    )
    assert ctor["text_detection_model_name"] == cfg.paddleocr.text_detection_model_name
    assert ctor["text_detection_model_dir"] == cfg.paddleocr.text_detection_model_dir
    assert ctor["device"] == cfg.paddleocr.device


def test_predict_iter_preserves_input_order_for_multiple_regions(monkeypatch, tmp_path):
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass

        def predict_iter(self, images):
            for i, _ in enumerate(images):
                yield {
                    "rec_boxes": [[i, i + 1, i + 2, i + 3]],
                    "rec_texts": [f"region_{i}"],
                    "rec_scores": [0.5 + i * 0.1],
                }

    _install_fake_paddleocr_module(monkeypatch, FakePaddleOCR)

    from vision_core.ocr import paddle_ocr as paddle_ocr_mod

    importlib.reload(paddle_ocr_mod)

    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    eng = paddle_ocr_mod.PaddleOcrEngine(
        _make_vc_config_with_dirs(str(det_dir), str(rec_dir))
    )

    regions = [
        np.zeros((5, 5, 3), dtype=np.uint8),
        np.zeros((6, 6, 3), dtype=np.uint8),
        np.zeros((7, 7, 3), dtype=np.uint8),
    ]

    results = list(eng.predict_iter(regions))
    assert len(results) == 3
    assert results[0][0].text == "region_0"
    assert results[1][0].text == "region_1"
    assert results[2][0].text == "region_2"


def test_predict_materializes_iter_results(monkeypatch, tmp_path):
    class FakePaddleOCR:
        def __init__(self, **kwargs):
            pass

        def predict_iter(self, images):
            for _ in images:
                yield {"rec_boxes": [], "rec_texts": [], "rec_scores": []}

    _install_fake_paddleocr_module(monkeypatch, FakePaddleOCR)

    from vision_core.ocr import paddle_ocr as paddle_ocr_mod

    importlib.reload(paddle_ocr_mod)

    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    eng = paddle_ocr_mod.PaddleOcrEngine(
        _make_vc_config_with_dirs(str(det_dir), str(rec_dir))
    )

    regions = [
        np.zeros((5, 5, 3), dtype=np.uint8),
        np.zeros((5, 5, 3), dtype=np.uint8),
    ]
    out = eng.predict(regions)

    assert isinstance(out, list)
    assert out == [[], []]
