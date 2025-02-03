# Basic Scikit-Learn Guide for ML Engineers

## Introduction to Scikit-Learn
**Scikit-Learn** is a popular Python library for **machine learning**. It provides efficient tools for **data preprocessing, model selection, training, and evaluation**.

---
## 1. Installing and Importing Scikit-Learn
### Installation
If Scikit-Learn is not installed, use:
```bash
pip install scikit-learn
```

### Importing Scikit-Learn
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

---
## 2. Loading a Dataset
### Built-in Datasets
```python
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target
print(X.shape, y.shape)  # Output: (150, 4) (150,)
```

### Loading a CSV File
```python
df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]
```

---
## 3. Splitting Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)  # Output: (120, 4) (30, 4)
```

---
## 4. Data Preprocessing
### Standardizing Features
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---
## 5. Training a Machine Learning Model
### Linear Regression Example
```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

### Making Predictions
```python
y_pred = model.predict(X_test_scaled)
```

---
## 6. Evaluating Model Performance
```python
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

---
## 7. Commonly Used ML Algorithms
### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

### Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
```

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

---
## 8. Model Selection & Cross-Validation
### K-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("Mean Accuracy:", scores.mean())
```

---
## 9. Hyperparameter Tuning
### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
```

### Randomized Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {"max_depth": randint(1, 10)}
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)
```

---
## 10. Saving and Loading Models
### Saving a Model
```python
import joblib
joblib.dump(model, "model.pkl")
```

### Loading a Model
```python
model = joblib.load("model.pkl")
```

---
## Conclusion
Scikit-Learn provides **powerful tools for data preprocessing, model training, evaluation, and optimization**. Mastering it will help you build **robust machine learning models** efficiently.

For more advanced topics, check out additional guides on **TensorFlow, PyTorch, and MLOps**!

Happy coding! 🚀
