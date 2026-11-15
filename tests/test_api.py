import numpy as np
import pytest


def test_api_inspect(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import cv2
    from fastapi.testclient import TestClient

    from defectinspect import api
    from defectinspect.onnx_detector import Detection

    class Fake:
        def predict(self, img):
            return [Detection(5, "scratches", 0.9, (10, 10, 50, 60))]

    api._state.update(det=Fake(), rules=api.load_rules(None))
    c = TestClient(api.app)
    ok, png = cv2.imencode(".png", np.full((200, 200, 3), 128, np.uint8))
    r = c.post("/inspect", files={"file": ("p.png", png.tobytes(), "image/png")}).json()
    assert r["verdict"] == "FAIL" and r["defects"][0]["class"] == "scratches" and r["annotated_png_base64"]
    assert c.post("/inspect", files={"file": ("x.png", b"nope", "image/png")}).status_code == 422
