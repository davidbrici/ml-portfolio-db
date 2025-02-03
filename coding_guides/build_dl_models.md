# Basic Deep Learning Guide for ML Engineers

## Introduction to Deep Learning
**Deep Learning** is a subset of Machine Learning that focuses on neural networks with multiple layers. It is widely used for **image recognition, NLP, speech processing, and autonomous systems**.

---
## 1. Setting Up the Environment
### Installing Required Libraries
```bash
pip install tensorflow keras torch torchvision
```

### Importing Libraries
```python
import tensorflow as tf
import torch
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
```

---
## 2. Understanding Neural Networks
### Components of a Neural Network
- **Neurons:** Basic computing unit
- **Layers:** Input, Hidden, and Output layers
- **Activation Functions:** Sigmoid, ReLU, Tanh
- **Loss Function:** Measures prediction error
- **Optimizer:** Updates weights to minimize loss

### Common Activation Functions
```python
import tensorflow.keras.activations as activations

relu_output = activations.relu(np.array([-1, 0, 1]))
sigmoid_output = activations.sigmoid(np.array([-1, 0, 1]))
```

---
## 3. Building a Simple Neural Network
### Using TensorFlow/Keras
```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Using PyTorch
```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = SimpleNN()
```

---
## 4. Training a Neural Network
### Preparing Data
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, size=(1000,))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### Training in TensorFlow
```python
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

### Training in PyTorch
```python
import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, X_train, y_train):
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs.squeeze(), torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
```

---
## 5. Evaluating Model Performance
### Using TensorFlow
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

### Using PyTorch
```python
def evaluate(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32))
        predictions = (outputs.squeeze() > 0.5).float()
        accuracy = (predictions.numpy() == y_test).mean()
    print(f"Test Accuracy: {accuracy:.2f}")
```

---
## 6. Convolutional Neural Networks (CNNs) for Image Data
### CNN Architecture in TensorFlow
```python
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### CNN Architecture in PyTorch
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32*26*26, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

cnn_model = CNN()
```

---
## 7. Deploying a Deep Learning Model
### Saving and Loading Models
#### TensorFlow
```python
model.save("model.h5")
loaded_model = tf.keras.models.load_model("model.h5")
```

#### PyTorch
```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

---
## Conclusion
Deep Learning is a **powerful tool** for **complex pattern recognition and AI applications**. This guide provides a **foundation for training and deploying models**.

For more advanced topics, check out **Transfer Learning, Transformers, and GANs**!

Happy coding! 🚀
