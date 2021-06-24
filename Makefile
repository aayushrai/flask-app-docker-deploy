DOCKER_TAG := face-recog-flask-server-deploy

build-docker:
	docker build . -t $(DOCKER_TAG)

run-docker:
	docker run --rm -p 5000:5000 --name facerecog-server $(DOCKER_TAG)

