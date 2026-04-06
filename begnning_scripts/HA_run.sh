#!/usr/bin/env sh

#!/bin/bash
echo "--- Starting Marine Surveillance Pipeline ---"

# Extract settings from the Home Assistant UI
export INPUT_VIDEO=$(jq --raw-output '.input_video_path' /data/options.json)
export INFERENCE_TEMP=$(jq --raw-output '.inference_temperature' /data/options.json)

# The MQTT credentials are automatically injected by HA's Supervisor
export MQTT_HOST=$MQTT_HOST
export MQTT_PORT=$MQTT_PORT
export MQTT_USER=$MQTT_USER
export MQTT_PASSWORD=$MQTT_PASSWORD

echo "Target Video: $INPUT_VIDEO"
echo "Connecting to MQTT Broker at: $MQTT_HOST:$MQTT_PORT"

# Launch the Python script
python3 /app/segformer_final_script.py

