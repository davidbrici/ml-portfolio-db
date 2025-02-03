# Kubernetes Guide for ML Engineers

## Introduction to Kubernetes
**Kubernetes (K8s)** is an open-source platform for **automating deployment, scaling, and management** of containerized applications. It is widely used in **MLOps** to orchestrate ML model training, inference, and scaling.

---
## 1. Installing Kubernetes
### Installing Minikube (For Local Testing)
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```
Start Minikube:
```bash
minikube start
```

### Installing kubectl (Kubernetes CLI)
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

Verify installation:
```bash
kubectl version --client
```

---
## 2. Key Kubernetes Concepts
- **Pods**: The smallest deployable unit in Kubernetes.
- **Deployments**: Manage the lifecycle of pods and ensure desired state.
- **Services**: Expose applications running in Kubernetes.
- **ConfigMaps & Secrets**: Manage environment variables and sensitive data.
- **Volumes**: Manage persistent storage for containers.

---
## 3. Deploying a Simple ML Application
### Creating a Deployment YAML File
Save the following as `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-app
  template:
    metadata:
      labels:
        app: ml-app
    spec:
      containers:
      - name: ml-app
        image: mydockerhubuser/ml-app:latest
        ports:
        - containerPort: 5000
```

### Applying the Deployment
```bash
kubectl apply -f deployment.yaml
```

### Checking the Status
```bash
kubectl get pods
```

---
## 4. Exposing the Application with a Service
Create a `service.yaml` file:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-app-service
spec:
  selector:
    app: ml-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```
Apply the service:
```bash
kubectl apply -f service.yaml
```
Find the service URL:
```bash
kubectl get services
```

---
## 5. Scaling and Auto-scaling
### Scaling Manually
```bash
kubectl scale deployment ml-app --replicas=5
```

### Enabling Autoscaling
```bash
kubectl autoscale deployment ml-app --cpu-percent=50 --min=1 --max=10
```

Check autoscaler status:
```bash
kubectl get hpa
```

---
## 6. Using ConfigMaps and Secrets
### Creating a ConfigMap
```bash
kubectl create configmap ml-config --from-literal=MODEL_PATH=/models/model.pkl
```
Mount in `deployment.yaml`:
```yaml
env:
- name: MODEL_PATH
  valueFrom:
    configMapKeyRef:
      name: ml-config
      key: MODEL_PATH
```

### Creating a Secret
```bash
kubectl create secret generic ml-secret --from-literal=API_KEY=myapikey123
```
Mount in `deployment.yaml`:
```yaml
env:
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: ml-secret
      key: API_KEY
```

---
## 7. Persistent Storage
### Creating a Persistent Volume
Save as `pv.yaml`:
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ml-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/mnt/data"
```
Apply the volume:
```bash
kubectl apply -f pv.yaml
```

---
## 8. Monitoring and Logging
### Viewing Logs
```bash
kubectl logs <pod-name>
```

### Monitoring Resources
```bash
kubectl top pods
```

### Describing a Pod
```bash
kubectl describe pod <pod-name>
```

---
## 9. Running ML Workloads on Kubernetes
### Using Kubernetes Jobs for Batch ML Tasks
Save as `job.yaml`:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-ml-model
spec:
  template:
    spec:
      containers:
      - name: train-job
        image: mydockerhubuser/train-ml
        command: ["python", "train.py"]
      restartPolicy: Never
```
Apply the job:
```bash
kubectl apply -f job.yaml
```
Check the job status:
```bash
kubectl get jobs
```

---
## 10. Deploying ML Models with Kubernetes
For deploying models, use:
- **FastAPI or Flask** for serving.
- **Kubernetes Ingress** for external access.
- **Istio** for traffic management.

Example model deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: mydockerhubuser/model-server:latest
        ports:
        - containerPort: 8080
```

Expose using a service:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: LoadBalancer
```

Apply configurations:
```bash
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml
```

---
## Conclusion
Kubernetes is essential for **scaling ML workloads, automating deployments, and managing containerized ML applications**. Mastering Kubernetes ensures efficient **MLOps workflows**.

For more advanced topics, check out **Kubeflow, Helm, and CI/CD with Kubernetes**!

Happy deploying! 🚀
