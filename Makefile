PROJECT_NAME = pyspark_encoders
DOCKER_IMG := $(PROJECT_NAME):latets
DOCKER_RUN := docker run --rm -t

build:
	docker build -f docker/Dockerfile -t $(DOCKER_IMG) .

force-build:
	docker build --no-cache -t $(DOCKER_IMG) .

shell:
	$(DOCKER_RUN) -i  -p 8888:8888  --entrypoint=/bin/bash $(DOCKER_IMG)

run: build
	$(DOCKER_RUN) -p 8888:8888 -v $(PWD)/examples:/app/examples $(DOCKER_IMG)

mypy: build
	$(DOCKER_RUN) $(DOCKER_IMG) mypy $(PROJECT_NAME) tests

flake: build
	$(DOCKER_RUN) $(DOCKER_IMG) flake8 $(PROJECT_NAME) tests

bandit: build
	$(DOCKER_RUN) $(DOCKER_IMG) bandit $(PROJECT_NAME) tests

format:
	black $(PROJECT_NAME) tests

lint: mypy flake bandit

jup:
	docker run --rm -it -p 8888:8888 jupyter/pyspark-notebook /bin/bash
