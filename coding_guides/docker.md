# Docker Guide for ML Engineers

## Introduction to Docker
**Docker** is a containerization platform that allows ML engineers to create, deploy, and manage applications in isolated environments. It simplifies dependency management and ensures **reproducibility** across different systems.

---
## 1. Installing Docker
### Installation on Linux
```bash
sudo apt update
sudo apt install docker.io
```

### Installation on macOS
Use **Docker Desktop**: [Download here](https://www.docker.com/products/docker-desktop/)

### Verifying Installation
```bash
docker --version
```

---
## 2. Key Docker Concepts
- **Images**: A blueprint for containers, containing all dependencies.
- **Containers**: Running instances of Docker images.
- **Dockerfile**: A script defining how to build a Docker image.
- **Docker Hub**: A repository for sharing Docker images.

---
## 3. Running a Simple Container
```bash
docker run hello-world
```
This downloads and runs a small test container.

---
## 4. Working with Docker Images
### Pulling an Image
```bash
docker pull python:3.9
```

### Listing Available Images
```bash
docker images
```

### Removing an Image
```bash
docker rmi python:3.9
```

---
## 5. Creating a Dockerfile for ML Applications
### Sample `Dockerfile`
```dockerfile
# Use an official Python image as the base
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Define the command to run the ML model
CMD ["python", "train.py"]
```

### Building the Image
```bash
docker build -t my-ml-app .
```

### Running the Container
```bash
docker run my-ml-app
```

---
## 6. Managing Containers
### Listing Running Containers
```bash
docker ps
```

### Stopping a Container
```bash
docker stop <container_id>
```

### Removing a Container
```bash
docker rm <container_id>
```

---
## 7. Exposing Ports & Running APIs
For ML models deployed as APIs, expose ports in `Dockerfile`:
```dockerfile
EXPOSE 5000
CMD ["python", "app.py"]
```
Run the container with port forwarding:
```bash
docker run -p 5000:5000 my-ml-app
```

---
## 8. Using Docker Compose for ML Pipelines
### Sample `docker-compose.yml`
```yaml
version: '3'
services:
  web:
    build: .
    ports:
      - "5000:5000"
```
Start with:
```bash
docker-compose up
```

---
## 9. Saving and Sharing Docker Images
### Saving an Image
```bash
docker save -o my-ml-app.tar my-ml-app
```

### Loading an Image
```bash
docker load -i my-ml-app.tar
```

### Pushing an Image to Docker Hub
```bash
docker tag my-ml-app mydockerhubuser/my-ml-app

docker push mydockerhubuser/my-ml-app
```

---
## 10. Running Containers on GPU (NVIDIA Docker)
### Installing NVIDIA Docker
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Running a Container with GPU Support
```bash
docker run --gpus all tensorflow/tensorflow:latest-gpu
```

---
## Conclusion
Docker simplifies **dependency management, model deployment, and scalability** for ML engineers. By using **Docker Compose and GPU support**, ML pipelines become more efficient and reproducible.

For more advanced topics, check out **Kubernetes, CI/CD with Docker, and MLOps workflows**!

Happy coding! 🚀
