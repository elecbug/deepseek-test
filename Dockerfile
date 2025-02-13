FROM ubuntu:latest

RUN apt-get update --fix-missing && \
    apt-get install -y python3 python3-venv git-lfs

COPY deepseek-llm-7b-base deepseek-llm-7b-base

RUN mkdir /app
WORKDIR /app

COPY main.py main.py

RUN python3 -m venv venv
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install fastapi uvicorn torch

CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]