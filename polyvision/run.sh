#!/usr/bin/env bash

# This script is invoked by the HA Supervisor upon container start.

echo "--- Starting PolyVision Multi-Agent System ---"

# The /data/options.json is automatically mounted by HA Supervisor
# containing the config UI choices made by the user.
export INPUT_VIDEO=$(jq --raw-output '.input_video_path' /data/options.json)
export INFERENCE_TEMP=$(jq --raw-output '.inference_temperature' /data/options.json)

# MQTT logic
    echo "Notice: MQTT_HOST not found in environment. Using default fallback: core-mosquitto"
    export MQTT_HOST="core-mosquitto"
else
    echo "MQTT Credentials detected. Broker: $MQTT_HOST:$MQTT_PORT"
fi

echo "---"
echo "Notice: If your RTSP stream fails, PolyVision will look for a local fallback video file."
echo "We recommend uploading a file to: /config/polyvision/fallback.mp4 using the HA File Editor or Samba add-on."
echo "---"
echo "Configuration loaded. Initial Target Video: $INPUT_VIDEO"

# Launch the Orchestrator
python3 /app/src/main.py
