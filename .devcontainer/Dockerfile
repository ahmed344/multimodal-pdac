FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y git

# Switch to the existing non-root user (UID 1000)
USER 1000

# Allow pip to install system-wide in this isolated container
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# # Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt
