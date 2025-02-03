# Basic PyTorch Guide for ML Engineers

## Introduction to PyTorch
**PyTorch** is an open-source machine learning framework developed by Facebook. It is widely used for **deep learning, neural networks, and AI research** due to its **dynamic computation graph** and **easy debugging capabilities**.

---
## 1. Installing and Importing PyTorch
### Installation
If PyTorch is not installed, use:
```bash
pip install torch torchvision torchaudio
```

### Importing PyTorch
```python
import torch
print(torch.__version__)  # Check PyTorch version
```

---
## 2. Creating Tensors
### Defining a Tensor
```python
tensor = torch.tensor([[1, 2], [3, 4]])
print(tensor)
```

### Creating Random Tensors
```python
rand_tensor = torch.rand((3,3))
print(rand_tensor)
```

### Tensor Operations
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

result = torch.add(a, b)  # Element-wise addition
print(result)
```

---
## 3. Converting Between NumPy and PyTorch
```python
import numpy as np

# Convert NumPy array to Tensor
a = np.array([[1, 2], [3, 4]])
tensor_a = torch.from_numpy(a)

# Convert Tensor to NumPy array
numpy_a = tensor_a.numpy()
```

---
## 4. Building a Simple Neural Network
### Defining a Model
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = SimpleNN()
print(model)
```

---
## 5. Training a Model
### Defining Loss and Optimizer
```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

### Training Loop
```python
for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.rand(1, 4))  # Dummy input
    loss = criterion(output, torch.tensor([[1.0]]))
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

---
## 6. Loading and Preprocessing Data
### Using PyTorch Datasets and DataLoaders
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---
## 7. Evaluating and Making Predictions
### Evaluating the Model
```python
with torch.no_grad():
    test_input = torch.rand(1, 4)
    prediction = model(test_input)
    print(prediction)
```

---
## 8. Saving and Loading Models
### Saving the Model
```python
torch.save(model.state_dict(), "model.pth")
```

### Loading the Model
```python
model.load_state_dict(torch.load("model.pth"))
```

---
## 9. PyTorch with GPU Support
### Checking GPU Availability
```python
print("CUDA Available:", torch.cuda.is_available())
```

### Moving Model to GPU
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

---
## 10. PyTorch for Deployment
### Converting Model to TorchScript for Production
```python
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model_scripted.pt")
```

---
## Conclusion
PyTorch is a **flexible and powerful deep learning framework** that is widely used in research and production. Mastering PyTorch enables ML engineers to build **efficient deep learning models**.

For more advanced topics, check out additional guides on **TensorFlow, ONNX, and MLOps**!

Happy coding! 🚀
