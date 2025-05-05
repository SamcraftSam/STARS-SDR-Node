import paho.mqtt.client as mqtt
import json
import base64
from datetime import datetime, timezone
import io
from PIL import Image
from typing import Optional, Union


class IMQTTClient:
    def connect(self): ...
    def disconnect(self): ...
    def switch_topic(self, topic: str): ...
    def send(self, payload: dict): ...


class ImageEncoder:
    def encode_image(self, image_array) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image_array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


class MetadataBuilder:
    def __init__(self, satellite: str, location: str, coordinates: Union[str, tuple], timestamp: Optional[str] = None):
        self.satellite = satellite
        self.location = location
        self.coordinates = coordinates
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def build(self) -> dict:
        return {
            "satellite": self.satellite,
            "location": self.location,
            "coordinates": self.coordinates,
            "timestamp": self.timestamp,
        }


class MQTTImagePublisher(IMQTTClient):
    def __init__(self, broker_host="localhost", broker_port=1883, topic="apt/images"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client = mqtt.Client()

    def switch_topic(self, topic: str):
        self.topic = topic

    def connect(self):
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def send(self, payload: dict):
        self.client.publish(self.topic, json.dumps(payload))


class APTMQTTService:
    def __init__(self, publisher: IMQTTClient, encoder: ImageEncoder):
        self.publisher = publisher
        self.encoder = encoder

    def send_image(self, image_array, satellite: str, location: str,
                   coordinates: Union[str, tuple], timestamp: Optional[str] = None):
        metadata = MetadataBuilder(satellite, location, coordinates, timestamp).build()
        metadata["image_format"] = "png"
        metadata["image_data"] = self.encoder.encode_image(image_array)
        self.publisher.send(metadata)


# Example use
if __name__ == '__main__':
    print("[TEST] Sending sample image over MQTT")
    publisher = MQTTImagePublisher("broker.hivemq.com")
    encoder = ImageEncoder()
    service = APTMQTTService(publisher, encoder)

    publisher.connect()

    import numpy as np
    dummy_image = np.zeros((100, 100), dtype=np.uint8)
    service.send_image(dummy_image, "NOAA-19", "Berlin, Germany", coordinates=(52.52, 13.40))

    publisher.disconnect()
