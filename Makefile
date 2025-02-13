apt:
	sudo apt-get update
	sudo apt-get install -y git git-lfs docker.io python3 python3-venv || true
	
build:
	sudo docker build -t ds .
run:
	sudo docker run -dit --name ds -p 80:8000 --env WORKERS=1 ds
stop:
	sudo docker stop ds
	sudo docker rm ds
log:
	sudo docker logs ds
	sudo docker ps