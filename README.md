# PyTorch-Project-Template
Basic template for PyTorch proects.

# Setup Docker Container
To build the Docker container run the following script and replace [IMAGE_NAME] with a name for the image:
```bash
docker build -t [IMAGE_NAME] --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```
To start the docker container run the following command from the project's root and replace [IMAGE_NAME] with the name chosen before and [CONTAINER_NAME] with a name for the container.:
```bash
docker run --rm --shm-size 16G --name [CONTAINER_NAME] --gpus '"device=0"' --cpus=16 -v $(pwd):/workspace/ -it [IMAGE_NAME] bash
```
