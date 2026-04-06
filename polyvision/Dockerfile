ARG BUILD_FROM=python:3.10-slim
FROM $BUILD_FROM

# Install system dependencies (jq is crucial for parsing HA config)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip requirements. CPU-only torch to keep image smaller for general usage.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers opencv-python paho-mqtt matplotlib numpy pydantic

# Copy the execution script and source code
COPY run.sh /app/
COPY src/ /app/src/

# Make the wrapper script executable
RUN chmod a+x /app/run.sh

# Hand over execution to the wrapper, per Supervisor norms
CMD [ "/app/run.sh" ]
