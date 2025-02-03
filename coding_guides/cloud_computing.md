# Basic Networking & Cloud Computing Guide for ML Engineers

## Introduction to Networking & Cloud Computing
Machine Learning models often rely on **networking** for data transfer, model deployment, and distributed training. **Cloud computing** provides scalable resources to store data, train models, and deploy applications efficiently.

---
## 1. Basic Networking Concepts
### Key Terms
- **IP Address**: Unique identifier for devices on a network.
- **DNS (Domain Name System)**: Converts domain names into IP addresses.
- **HTTP & HTTPS**: Protocols for web communication.
- **Load Balancer**: Distributes incoming traffic across multiple servers.
- **Latency & Bandwidth**: Measure of network speed and capacity.

### Checking Network Connectivity
```bash
ping google.com
```

### Checking Open Ports
```bash
netstat -tulnp  # Linux
```

---
## 2. Cloud Computing for ML Engineers
### Why Use Cloud Computing?
- **Scalability**: Auto-scale resources based on demand.
- **Cost-Effective**: Pay-as-you-go pricing.
- **Flexibility**: Use pre-configured ML environments.
- **Collaboration**: Store and share data globally.

### Popular Cloud Providers
- **AWS (Amazon Web Services)** – EC2, S3, SageMaker
- **Google Cloud Platform (GCP)** – Vertex AI, BigQuery
- **Microsoft Azure** – Azure ML, Blob Storage

---
## 3. Setting Up a Virtual Machine (VM) in the Cloud
### Creating an AWS EC2 Instance (Linux)
```bash
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t2.micro \
    --key-name my-key \
    --security-groups my-security-group
```

### Connecting to the VM via SSH
```bash
ssh -i my-key.pem ubuntu@your-instance-ip
```

---
## 4. Storing Data in the Cloud
### Uploading Files to AWS S3
```bash
aws s3 cp dataset.csv s3://my-ml-bucket/
```

### Downloading Files from GCP Storage
```bash
gsutil cp gs://my-ml-bucket/dataset.csv .
```

---
## 5. Deploying ML Models on the Cloud
### Using Flask for Model Deployment
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Running the Flask App on a Cloud Server
```bash
python app.py
```
Access the API via:
```bash
curl -X POST http://your-server-ip:5000/predict -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---
## 6. Containerizing ML Applications with Docker
### Creating a Dockerfile
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

### Building and Running the Container
```bash
docker build -t ml-app .
docker run -p 5000:5000 ml-app
```

---
## 7. Scaling ML Deployments with Kubernetes
### Deploying a Model in Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-container
        image: mydockerhub/ml-model:latest
        ports:
        - containerPort: 5000
```

### Applying Kubernetes Configurations
```bash
kubectl apply -f deployment.yaml
```

---
## 8. Secure Model Deployment
### Setting Up a Firewall (AWS Security Group Example)
```bash
aws ec2 authorize-security-group-ingress \
    --group-name my-security-group \
    --protocol tcp --port 5000 --cidr 0.0.0.0/0
```

### Using HTTPS with Let’s Encrypt
```bash
sudo certbot --nginx -d yourdomain.com
```

---
## Conclusion
Networking and cloud computing are essential for **scaling ML models, handling data storage, and deploying real-time applications**. Mastering cloud environments ensures **efficient and cost-effective machine learning solutions**.

For more advanced topics, check out **MLOps, CI/CD Pipelines, and Distributed Training**!

Happy coding! 🚀
