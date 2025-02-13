FROM ubuntu:latest
ENV WORKERS=1

RUN apt-get update && \
    apt-get install -y python3 python3-venv git-lfs

RUN mkdir /app
WORKDIR /app

RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat

WORKDIR /app/deepseek-llm-7b-chat
RUN git lfs fetch --include="*" --exclude=""
RUN git lfs checkout

WORKDIR /app

RUN python3 -m venv venv
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install fastapi uvicorn torch transformers accelerate python-multipart pydantic

COPY templates templates
COPY main.py main.py

CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]