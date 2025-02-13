FROM ubuntu:latest
ENV WORKERS=1

RUN apt-get update && \
    apt-get install -y python3 python3-venv git-lfs

RUN mkdir /app
WORKDIR /app

COPY main.py main.py

RUN git lfs install
RUN git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

RUN python3 -m venv venv
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install fastapi uvicorn torch transformers

CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", $WORKERS]