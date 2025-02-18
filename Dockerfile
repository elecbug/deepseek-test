FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3 python3-venv python3-dev git-lfs
RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:graphics-drivers/ppa -y
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-535

RUN mkdir /app
WORKDIR /app

RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat

WORKDIR /app/deepseek-llm-7b-chat
RUN git lfs fetch --include="*" --exclude=""
RUN git lfs checkout

WORKDIR /app

RUN python3 -m venv venv
RUN /app/venv/bin/pip install --upgrade pip setuptools

RUN /app/venv/bin/pip install torch torchvision torchaudio
RUN /app/venv/bin/pip install fastapi uvicorn transformers accelerate python-multipart pydantic
RUN /app/venv/bin/pip install -U bitsandbytes

RUN /app/venv/bin/python --version

COPY templates templates
COPY main.py main.py

ENTRYPOINT ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
