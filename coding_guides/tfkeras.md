# Basic TensorFlow Guide for ML Engineers

## Introduction to TensorFlow
**TensorFlow** is an open-source library developed by Google for deep learning and machine learning. It is widely used for **training, deploying, and running models efficiently on CPUs, GPUs, and TPUs**.

---
## 1. Installing and Importing TensorFlow
### Installation
If TensorFlow is not installed, use:
```bash
pip install tensorflow
```

### Importing TensorFlow
```python
import tensorflow as tf
print(tf.__version__)  # Check TensorFlow version
```

---
## 2. Creating Tensors
### Defining a Tensor
```python
tensor = tf.constant([[1, 2], [3, 4]])
print(tensor)
```

### Creating Random Tensors
```python
rand_tensor = tf.random.uniform(shape=(3,3))
print(rand_tensor)
```

### Tensor Operations
```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

result = tf.add(a, b)  # Element-wise addition
print(result)
```

---
## 3. Building a Simple Neural Network
### Creating a Sequential Model
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Compiling the Model
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Summary of the Model
```python
model.summary()
```

---
## 4. Loading and Preprocessing Data
### Using Built-in Datasets
```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### Normalizing Data
```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

### Converting Labels to Categorical Format
```python
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

---
## 5. Training the Model
```python
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
```

---
## 6. Evaluating and Making Predictions
### Evaluating the Model
```python
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)
```

### Making Predictions
```python
predictions = model.predict(x_test[:5])
print(predictions)
```

---
## 7. Saving and Loading Models
### Saving the Model
```python
model.save("my_model.h5")
```

### Loading the Model
```python
new_model = tf.keras.models.load_model("my_model.h5")
```

---
## 8. TensorFlow with GPU Support
### Checking GPU Availability
```python
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
```

---
## 9. TensorFlow for Deployment
### Converting Model to TensorFlow Lite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### Saving TensorFlow Lite Model
```python
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

---
## Conclusion
TensorFlow is a **powerful deep learning framework** for building and deploying ML models efficiently. Mastering TensorFlow enables ML engineers to develop **state-of-the-art AI applications**.

For more advanced topics, check out additional guides on **Keras, PyTorch, and MLOps**!

Happy coding! 🚀
