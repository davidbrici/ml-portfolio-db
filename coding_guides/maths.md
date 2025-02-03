# Basic Mathematics for ML Engineers

## Introduction
Mathematics is the foundation of **Machine Learning (ML)**. Understanding mathematical concepts helps in **building models, optimizing algorithms, and interpreting results**. This guide covers essential mathematical topics used in ML.

---
## 1. Linear Algebra
**Linear Algebra** is the backbone of ML algorithms, particularly in handling data as matrices and vectors.

### Vectors and Matrices
```python
import numpy as np

# Defining vectors
vector = np.array([2, 3, 5])

# Defining matrices
matrix = np.array([[1, 2], [3, 4]])
```

### Matrix Operations
```python
# Matrix addition
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])
C = A + B  # Element-wise addition

# Matrix multiplication
D = np.dot(A, B)
```

### Eigenvalues and Eigenvectors
```python
values, vectors = np.linalg.eig(A)
print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)
```

---
## 2. Probability and Statistics
Probability and statistics are essential for making predictions and interpreting data.

### Probability Distributions
```python
from scipy.stats import norm

# Generating normal distribution data
x = np.linspace(-3, 3, 100)
pdf = norm.pdf(x, 0, 1)  # Mean=0, Std=1
```

### Descriptive Statistics
```python
import pandas as pd

data = pd.Series([10, 20, 30, 40, 50])
print("Mean:", data.mean())
print("Median:", data.median())
print("Standard Deviation:", data.std())
```

---
## 3. Calculus in ML
Calculus is used in ML for optimization (e.g., **gradient descent**).

### Derivatives and Gradients
```python
import sympy as sp

x = sp.Symbol('x')
function = x**2 + 3*x + 5
derivative = sp.diff(function, x)
print("Derivative:", derivative)
```

### Gradient Descent
```python
def gradient_descent(learning_rate=0.1, epochs=10):
    x = 10  # Initial value
    for _ in range(epochs):
        gradient = 2 * x  # Derivative of x^2
        x = x - learning_rate * gradient
    return x

print("Optimized value:", gradient_descent())
```

---
## 4. Optimization Techniques
Optimization helps in improving the performance of ML models.

### Cost Function
```python
def cost_function(m):
    return (m - 3) ** 2  # Example cost function
```

### Gradient Descent Example
```python
def simple_gradient_descent():
    m = 10  # Initial value
    learning_rate = 0.1
    for _ in range(10):
        m = m - learning_rate * 2 * (m - 3)
    return m

print("Optimal m:", simple_gradient_descent())
```

---
## 5. Probability in Machine Learning
### Bayes’ Theorem
```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

posterior = bayes_theorem(0.5, 0.8, 0.6)
print("Posterior Probability:", posterior)
```

---
## 6. Principal Component Analysis (PCA)
PCA is used for **dimensionality reduction** in ML.

### PCA Implementation
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

X, _ = load_iris(return_X_y=True)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

---
## 7. Linear Regression and Least Squares
Linear regression fits a line to data points using the **least squares method**.

### Simple Linear Regression
```python
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X, y)
print("Predicted value:", model.predict([[6]]))
```

---
## 8. Logistic Regression (Classification)
Used for **binary classification**.

### Logistic Function
```python
import scipy.special as sp

def sigmoid(x):
    return sp.expit(x)
```

### Applying Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])
model = LogisticRegression()
model.fit(X, y)
print("Prediction:", model.predict([[2.5]]))
```

---
## Conclusion
Mathematics is crucial for **understanding, developing, and optimizing ML models**. Mastering **Linear Algebra, Probability, Calculus, and Optimization** will significantly improve an ML engineer's ability to build better models.

For more advanced topics, check out **Fourier Transforms, Information Theory, and Advanced Optimization**!

Happy coding! 🚀