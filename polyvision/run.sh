#!/usr/bin/env bash

# This script is invoked by the HA Supervisor upon container start.

echo "--- Starting PolyVision Multi-Agent System ---"

# The /data/options.json is automatically mounted by HA Supervisor
# containing the config UI choices made by the user.
export INPUT_VIDEO=$(jq --raw-output '.input_video_path' /data/options.json)
export INFERENCE_TEMP=$(jq --raw-output '.inference_temperature' /data/options.json)

# MQTT logic
if [ -z "$MQTT_HOST" ]; then
    echo "Warning: MQTT_HOST not found in environment. The HA Supervisor should inject this because mqtt: true is in config.yaml."
else
    echo "MQTT Credentials detected. Broker: $MQTT_HOST:$MQTT_PORT"
fi

echo "Configuration loaded. Target Video: $INPUT_VIDEO"

# Launch the Orchestrator
python3 /app/src/main.py
