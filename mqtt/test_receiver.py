import unittest
from receiver_t import ImageStorageHandler
import base64
import numpy as np
from PIL import Image
import io
import os


class TestReceiver(unittest.TestCase):
    def setUp(self):
        self.handler = ImageStorageHandler(save_dir="test_received")

    def test_handle_valid_payload(self):
        # Создаём dummy PNG
        array = (np.ones((16, 16), dtype=np.uint8) * 222)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        payload = {
            "satellite": "TEST-SAT",
            "location": "Nowhere, Earth",
            "timestamp": "2025-01-01T12:00:00Z",
            "image_data": encoded
        }

        self.handler.handle(payload)

        # Проверка что PNG файл создан
        files = list(self.handler.save_dir.glob("TEST-SAT_*"))
        self.assertTrue(len(files) > 0)
        print(f"✓ Saved: {files[0]}")
        for f in files:
            os.remove(f)
                
    def test_missing_image_data(self):
        payload = {
            "satellite": "X",
            "location": "Y",
            "timestamp": "2025-01-01T12:00:00Z"
            # No image_data
        }
        self.handler.handle(payload) 

    def test_invalid_base64_data(self):
        payload = {
            "satellite": "X",
            "location": "Y",
            "timestamp": "2025-01-01T12:00:00Z",
            "image_data": "thisisnotbase64!"
        }
        self.handler.handle(payload) 

    def test_malformed_timestamp(self):
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
        self.handler.handle(payload) 

if __name__ == '__main__':
    unittest.main()
