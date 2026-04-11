ARG BUILD_FROM=python:3.10-slim
FROM $BUILD_FROM

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Allow pip to install packages globally (safe in a container)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install pip requirements. CPU-only torch to keep image smaller for general usage.
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir transformers opencv-python paho-mqtt matplotlib "numpy<2" pydantic

# Copy the execution script and source code
COPY run.sh /app/
COPY src/ /app/src/

# Make the wrapper script executable
RUN chmod a+x /app/run.sh

# Hand over execution to the wrapper, per Supervisor norms
CMD [ "/app/run.sh" ]
