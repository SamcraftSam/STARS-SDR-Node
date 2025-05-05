import os
import io
import json
import base64
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from sender import APTMQTTService, ImageEncoder
from receiver_t import ImageStorageHandler


class MockPublisher:
    def __init__(self):
        self.sent_payload = None

    def send(self, payload):
        self.sent_payload = payload


@pytest.fixture
def dummy_image():
    return (np.ones((32, 32), dtype=np.uint8) * 128)


@pytest.fixture
def service():
    publisher = MockPublisher()
    encoder = ImageEncoder()
    return APTMQTTService(publisher, encoder), publisher


@pytest.fixture
def handler(tmp_path):
    return ImageStorageHandler(save_dir=tmp_path), tmp_path


def test_send_image_valid(service, dummy_image):
    svc, publisher = service
    svc.send_image(dummy_image, "NOAA-19", "Berlin", (52.5, 13.4))
    payload = publisher.sent_payload
    assert payload["satellite"] == "NOAA-19"
    assert "image_data" in payload


def test_send_blank_image(service):
    svc, publisher = service
    blank = np.zeros((16, 16), dtype=np.uint8)
    svc.send_image(blank, "NOAA-18", "Void", (0.0, 0.0))
    assert "image_data" in publisher.sent_payload


def test_receiver_valid(handler):
    h, path = handler
    arr = (np.random.rand(8, 8) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "satellite": "TEST",
        "location": "Nowhere",
        "timestamp": "2025-01-01T12:00:00Z",
        "image_data": encoded
    }
    h.handle(payload)
    files = list(Path(path).glob("TEST_*"))
    assert len(files) > 0


def test_receiver_missing_data(handler):
    h, path = handler
    payload = {
        "satellite": "X",
        "location": "Y",
        "timestamp": "2025-01-01T12:00:00Z"
    }
    before = set(Path(path).glob("*"))
    h.handle(payload)
    after = set(Path(path).glob("*"))
    assert before == after


def test_receiver_invalid_base64(handler):
    h, path = handler
    payload = {
        "satellite": "X",
        "location": "Y",
        "timestamp": "2025-01-01T12:00:00Z",
        "image_data": "notbase64!!!"
    }
    before = set(Path(path).glob("*"))
    h.handle(payload)
    after = set(Path(path).glob("*"))
    assert before == after


def test_receiver_malformed_timestamp(handler):
    h, path = handler
    arr = (np.random.rand(8, 8) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "satellite": "Z",
        "location": "Broken, Time",
        "timestamp": "not-a-time",
        "image_data": encoded
    }
    h.handle(payload)
    saved = list(Path(path).glob("Z_Broken_Time_*"))
    assert len(saved) > 0
