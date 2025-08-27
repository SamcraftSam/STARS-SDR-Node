import unittest
from unittest.mock import MagicMock
import numpy as np
from sender import APTMQTTService, ImageEncoder, MetadataBuilder


class MockPublisher:
    def __init__(self):
        self.sent_payload = None

    def send(self, payload):
        self.sent_payload = payload


class TestSender(unittest.TestCase):
    def setUp(self):
        self.publisher = MockPublisher()
        self.encoder = ImageEncoder()
        self.service = APTMQTTService(self.publisher, self.encoder)

    def test_send_image(self):
        dummy = np.ones((32, 32), dtype=np.uint8) * 123
        self.service.send_image(dummy, "NOAA-20", "Testville", (0.0, 0.0))
        payload = self.publisher.sent_payload

        self.assertIsNotNone(payload)
        self.assertEqual(payload["satellite"], "NOAA-20")
        self.assertEqual(payload["location"], "Testville")
        self.assertEqual(payload["coordinates"], (0.0, 0.0))
        self.assertEqual(payload["image_format"], "png")
        self.assertIn("image_data", payload)

    def test_send_image_missing_coords(self):
        dummy = np.ones((32, 32), dtype=np.uint8)
        self.service.send_image(dummy, "NOAA-21", "Nowhere", coordinates=(None, None))
        payload = self.publisher.sent_payload
        self.assertEqual(payload["coordinates"], (None, None))

    def test_send_blank_image(self):
        blank = np.zeros((16, 16), dtype=np.uint8)
        self.service.send_image(blank, "NOAA-22", "NullIsland", (0.0, 0.0))
        payload = self.publisher.sent_payload
        self.assertIn("image_data", payload)
        self.assertGreater(len(payload["image_data"]), 10)

    def test_send_invalid_image_type(self):
        with self.assertRaises(Exception):
            self.service.send_image("not-an-array", "Fake", "Nowhere", (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
