build:
	sudo docker build -t ds .
run:
	sudo docker run -dit --name ds --port 8000:80 ds
stop:
	sudo docker stop ds
	sudo docker rm ds
log:
	sudo docker logs ds