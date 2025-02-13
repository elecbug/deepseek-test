apt:
	sudo apt-get update
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
		&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
		sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
		sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo apt-get update
	sudo apt-get install -y nvidia-container-toolkit || true
	sudo apt-get install -y docker.io || true
	sudo nvidia-ctk runtime configure --runtime=docker
	sudo systemctl restart docker
	
build:
	sudo docker build -t ds .
run:
	sudo docker run -dit --name ds -p 8000:8000 --env WORKERS=1 --gpus all ds
stop:
	sudo docker stop ds
	sudo docker rm ds
log:
	sudo docker logs ds
	sudo docker ps
	
nvidia:
	sudo docker info | grep -i runtime
	sudo docker exec -it ds nvidia-smi