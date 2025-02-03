# ML Algorithm Guide for ML Engineers

## Introduction to Machine Learning Algorithms
Machine Learning (ML) algorithms can be broadly categorized into **supervised, unsupervised, and reinforcement learning** techniques. This guide covers fundamental ML algorithms with explanations and Python examples.

---
## 1. Supervised Learning
Supervised learning algorithms learn from labeled data.

### 1.1 Linear Regression (For Regression Problems)
**Use Case:** Predicting continuous values (e.g., house prices).

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[6]])
print(prediction)  # Output: [12.]
```

### 1.2 Logistic Regression (For Classification Problems)
**Use Case:** Binary classification (e.g., spam detection).

```python
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)
print(model.predict([[2.5]]))  # Predicts 0 or 1
```

### 1.3 Decision Tree Classifier
**Use Case:** Handling non-linear relationships in classification problems.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)
print(model.predict([[2.5]]))
```

### 1.4 Random Forest (Ensemble Learning)
**Use Case:** Handling overfitting and improving accuracy.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
print(model.predict([[2.5]]))
```

### 1.5 Support Vector Machines (SVM)
**Use Case:** Classification with clear margins.

```python
from sklearn.svm import SVC

model = SVC()
model.fit(X, y)
print(model.predict([[2.5]]))
```

---
## 2. Unsupervised Learning
Unsupervised learning algorithms find hidden patterns in data.

### 2.1 K-Means Clustering
**Use Case:** Customer segmentation, anomaly detection.

```python
from sklearn.cluster import KMeans

X = np.array([[1], [2], [5], [6], [8], [9]])
model = KMeans(n_clusters=2)
model.fit(X)
print(model.labels_)  # Cluster assignments
```

### 2.2 Principal Component Analysis (PCA)
**Use Case:** Dimensionality reduction for large datasets.

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[2, 3], [5, 8], [9, 10], [4, 6]])
model = PCA(n_components=1)
X_reduced = model.fit_transform(X)
print(X_reduced)
```

---
## 3. Reinforcement Learning (RL)
RL is used for sequential decision-making problems.

### 3.1 Q-Learning (Basic RL Algorithm)
**Use Case:** Game playing, robotics.

```python
import numpy as np

# Initialize Q-table
Q_table = np.zeros((5, 2))  # 5 states, 2 actions
print(Q_table)
```

---
## 4. Deep Learning (Neural Networks)
Deep learning algorithms handle complex data like images and text.

### 4.1 Artificial Neural Networks (ANN)
**Use Case:** Predicting complex patterns.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 4.2 Convolutional Neural Networks (CNNs)
**Use Case:** Image classification.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

### 4.3 Recurrent Neural Networks (RNNs) - LSTMs
**Use Case:** Time-series forecasting, NLP.

```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50),
    Dense(1)
])
```

---
## Conclusion
This guide introduces essential **ML algorithms** for classification, clustering, dimensionality reduction, reinforcement learning, and deep learning. Mastering these techniques will help **ML engineers build robust models**.

For more advanced topics, check out guides on **MLOps, Model Deployment, and Hyperparameter Tuning**!

Happy coding! 🚀
