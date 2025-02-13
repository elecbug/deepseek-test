apt:
	sudo apt-get update
	sudo apt-get install git docker.io git-lfs python3 python3-venv
download:
	git lfs install
	git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

build:
	sudo docker build -t ds .
run:
	sudo docker run -dit --name ds --port 8000:80 ds
stop:
	sudo docker stop ds
	sudo docker rm ds
log:
	sudo docker logs ds