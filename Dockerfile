FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    nano \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Instalar SOLO dependencias ligeras
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["bash"]
