# Basic NumPy Guide for ML Engineers

## Introduction to NumPy
**NumPy (Numerical Python)** is a fundamental library for scientific computing in Python. It provides support for **multi-dimensional arrays** and **mathematical operations**, making it essential for **machine learning**.

---
## 1. Installing and Importing NumPy
### Installation
If you haven’t installed NumPy yet, use:
```bash
pip install numpy
```

### Importing NumPy
```python
import numpy as np
```

---
## 2. Creating NumPy Arrays
### Creating a 1D Array
```python
arr1d = np.array([1, 2, 3, 4, 5])
print(arr1d)  # Output: [1 2 3 4 5]
```

### Creating a 2D Array
```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

### Creating Arrays with Zeros, Ones, and Random Values
```python
zeros = np.zeros((3,3))  # 3x3 matrix of zeros
ones = np.ones((2,2))    # 2x2 matrix of ones
randoms = np.random.rand(3,3)  # 3x3 matrix of random values
print(zeros, ones, randoms)
```

### Creating a Range of Values
```python
arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
print(arr, linspace_arr)
```

---
## 3. Array Properties
```python
print(arr2d.shape)   # Shape of the array
print(arr2d.size)    # Total number of elements
print(arr2d.dtype)   # Data type of elements
```

---
## 4. Indexing and Slicing
### Accessing Elements
```python
print(arr1d[2])  # Accessing the 3rd element (Index 2)
```

### Slicing Arrays
```python
print(arr1d[1:4])  # Extracts elements at indices 1 to 3
print(arr2d[:, 1]) # Extracts the second column
```

### Modifying Values
```python
arr1d[2] = 99
print(arr1d)  # Output: [1 2 99 4 5]
```

---
## 5. Mathematical Operations
### Basic Arithmetic
```python
arr = np.array([1, 2, 3, 4])
print(arr + 5)  # Element-wise addition
print(arr * 2)  # Element-wise multiplication
```

### Matrix Multiplication
```python
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
result = np.dot(mat1, mat2)
print(result)
# Output:
# [[19 22]
#  [43 50]]
```

### Aggregation Functions
```python
arr = np.array([1, 2, 3, 4, 5])
print(arr.mean())  # Mean
print(arr.sum())   # Sum
print(arr.std())   # Standard Deviation
print(arr.max())   # Maximum value
print(arr.min())   # Minimum value
```

---
## 6. Reshaping and Transposing
### Reshaping Arrays
```python
arr = np.arange(6).reshape(2, 3)
print(arr)
# Output:
# [[0 1 2]
#  [3 4 5]]
```

### Transposing Arrays
```python
print(arr.T)  # Swaps rows and columns
```

---
## 7. Filtering and Boolean Indexing
```python
arr = np.array([1, 2, 3, 4, 5, 6])
filtered_arr = arr[arr > 3]
print(filtered_arr)  # Output: [4 5 6]
```

---
## 8. Stacking and Concatenation
### Concatenating Arrays
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate([arr1, arr2])
print(concatenated)  # Output: [1 2 3 4 5 6]
```

### Stacking Arrays
```python
stacked = np.vstack([arr1, arr2])  # Vertical Stack
print(stacked)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

---
## 9. Broadcasting
NumPy allows operations between arrays of different shapes using broadcasting.
```python
arr = np.array([[1], [2], [3]])
vector = np.array([1, 2, 3])
result = arr + vector
print(result)
# Output:
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]
```

---
## 10. Random Number Generation
```python
rand_num = np.random.rand(3, 3)  # Random values between 0 and 1
rand_int = np.random.randint(1, 10, (3, 3))  # Random integers between 1 and 10
print(rand_num, rand_int)
```

---
## Conclusion
NumPy is a **powerful library** that provides high-performance array operations essential for **machine learning workflows**. Mastering NumPy will help you **efficiently manipulate data** and prepare it for **ML models**.

For more advanced topics, check out additional guides on **Pandas, Scikit-learn, TensorFlow, and PyTorch**!

Happy coding! 🚀