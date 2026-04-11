import asyncio
import os
import json
import paho.mqtt.client as mqtt

class IOTBridge:
    def __init__(self, in_vision_queue: asyncio.Queue, in_ui_queue: asyncio.Queue):
        self.in_vision_queue = in_vision_queue
        self.in_ui_queue = in_ui_queue
        
        # HA supervisor injects these config variables
        self.mqtt_host = os.getenv("MQTT_HOST", "core-mosquitto")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
        self.mqtt_user = os.getenv("MQTT_USER", "")
        self.mqtt_pass = os.getenv("MQTT_PASSWORD", "")
        self.topic_prefix = "homeassistant"

        self.client = mqtt.Client(client_id="polyvision_agent")
        if self.mqtt_user:
            self.client.username_pw_set(self.mqtt_user, self.mqtt_pass)
        
        self.connected = False
            
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[IOTBridge] Connected to MQTT Broker.")
            self.connected = True
            self._publish_discovery()
        else:
            print(f"[IOTBridge] Failed to connect: {rc}")

    def _publish_discovery(self):
        """
        Creates the Home Assistant entities automatically via MQTT discovery.
        """
        # Camera Entity to show the Matplotlib results
        camera_config = {
            "name": "PolyVision Segmentation Cam",
            "unique_id": "polyvision_seg_cam",
            "topic": f"{self.topic_prefix}/camera/polyvision/segmentation",
            "device": {
                "identifiers": ["polyvision_01"],
                "name": "PolyVision Marine System"
            }
        }
        self.client.publish(f"{self.topic_prefix}/camera/polyvision_seg_cam/config", json.dumps(camera_config), retain=True)

        # JSON Sensor for numerical data
        sensor_config = {
             "name": "PolyVision Active Classes",
             "unique_id": "polyvision_active_classes",
             "state_topic": f"{self.topic_prefix}/sensor/polyvision/classes",
             "value_template": "{{ value_json.active_count }}",
             "json_attributes_topic": f"{self.topic_prefix}/sensor/polyvision/classes",
             "device": {
                "identifiers": ["polyvision_01"],
                "name": "PolyVision Marine System"
            }
        }
        self.client.publish(f"{self.topic_prefix}/sensor/polyvision_classes/config", json.dumps(sensor_config), retain=True)

        print("\n" + "="*60)
        print("🎉 MQTT Discovery Payloads Sent!")
        print("To view your PolyVision dashboard, add a 'Manual' card to your HA dashboard")
        print("and paste the following YAML snippet:\n")
        print("type: vertical-stack")
        print("cards:")
        print("  - type: picture-entity")
        print("    entity: camera.polyvision_segmentation_cam")
        print("    show_name: false")
        print("    show_state: false")
        print("  - type: entities")
        print("    entities:")
        print("      - entity: sensor.polyvision_active_classes")
        print("        name: Active Tracking Classes")
        print("="*60 + "\n")

    def _connect(self):
        try:
             self.client.on_connect = self._on_connect
             self.client.connect(self.mqtt_host, self.mqtt_port, 60)
             self.client.loop_start()
        except Exception as e:
             print(f"[IOTBridge] Could not connect to MQTT Broker at {self.mqtt_host}:{self.mqtt_port}.")
             print(f"[IOTBridge] Reason: {e}")
             print("[IOTBridge] Running in Standalone Mode (No Home Assistant Integration).")

    async def run(self):
        print("[IOTBridge] Starting...")
        await asyncio.to_thread(self._connect)
        
        while True:
            # Process Formatted UI images
            if not self.in_ui_queue.empty():
                payload = await self.in_ui_queue.get()
                if self.connected and payload.rendered_image_bytes:
                    await asyncio.to_thread(
                         self.client.publish, 
                         f"{self.topic_prefix}/camera/polyvision/segmentation",
                         payload.rendered_image_bytes
                    )
                     
                    state_doc = {
                        "active_count": len(payload.unique_ids),
                        "detected_class_ids": list(map(int, payload.unique_ids))
                    }
                    
                    await asyncio.to_thread(
                         self.client.publish,
                         f"{self.topic_prefix}/sensor/polyvision/classes",
                         json.dumps(state_doc)
                    )
                self.in_ui_queue.task_done()
                
            # Here we could consume from in_vision_queue directly if we wanted metrics before UI formats them
            if not self.in_vision_queue.empty():
                _ = await self.in_vision_queue.get()
                self.in_vision_queue.task_done()

            # Prevent high CPU usage idlelooping
            await asyncio.sleep(0.05)
