apt:
	sudo apt-get update
	sudo apt-get install -y git git-lfs docker.io python3 python3-venv || true
download:
	git lfs install
	git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

build:
	sudo docker build -t ds .
run:
	sudo docker run -dit --name ds -p 8000:80 ds
stop:
	sudo docker stop ds
	sudo docker rm ds
log:
	sudo docker logs ds