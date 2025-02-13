apt:
	sudo apt-get update
	sudo apt-get install -y docker.io nvidia-container-toolkit || true
	sudo systemctl restart docker
	
build:
	sudo docker build -t ds .
run:
	sudo docker run -dit --name ds -p 80:8000 --env WORKERS=1 --gpus all ds
stop:
	sudo docker stop ds
	sudo docker rm ds
log:
	sudo docker logs ds
	sudo docker ps