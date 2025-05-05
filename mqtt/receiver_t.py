import paho.mqtt.client as mqtt
import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class IMessageHandler(Protocol):
    def handle(self, payload: dict) -> None: ...


class ImageStorageHandler:
    def __init__(self, save_dir="received"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def handle(self, payload: dict) -> None:
        try:
            satellite = payload.get("satellite", "unknown_sat")
            location = payload.get("location", "unknown_loc")
            location = location.replace(",", "").replace(" ", "_")
            timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
            ts = timestamp.replace(":", "-").replace("T", "_").split(".")[0]
            filename = f"{satellite}_{location}_{ts}.png"
            filepath = self.save_dir / filename

            image_data = payload["image_data"]
            decoded = base64.b64decode(image_data)

            with open(filepath, "wb") as f:
                f.write(decoded)

            print(f"[✓] Saved image to {filepath}")
        except Exception as e:
            print(f"[✗] Failed to process message: {e}")


class MQTTReceiver:
    def __init__(self, handler: IMessageHandler, broker_host="localhost", broker_port=1883, topic="apt/images"):
        self.handler = handler
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client = mqtt.Client(protocol=mqtt.MQTTv311)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.handler.handle(payload)
        except Exception as e:
            print(f"[MQTT] Invalid payload: {e}")

    def start(self):
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_forever()


if __name__ == "__main__":
    print("[*] MQTT receiver running")
    handler = ImageStorageHandler(save_dir="received")
    receiver = MQTTReceiver(handler, broker_host="broker.hivemq.com", topic="apt/images")
    receiver.start()
