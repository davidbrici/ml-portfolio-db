# Python Guide for ML Engineers

## Introduction to Python for ML Engineers
Python is the most widely used programming language for Machine Learning due to its simplicity, readability, and extensive ecosystem of ML libraries like **NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch**.

---
## 1. Python Syntax Essentials
### Variables and Data Types
```python
# Defining variables
integer_var = 10       # Integer
float_var = 10.5       # Float
string_var = "Hello"  # String
list_var = [1, 2, 3]   # List
tuple_var = (1, 2, 3)  # Tuple
dict_var = {"key": "value"}  # Dictionary
```

### Basic Operators
```python
x, y = 10, 3
print(x + y)  # Addition: 13
print(x - y)  # Subtraction: 7
print(x * y)  # Multiplication: 30
print(x / y)  # Division: 3.33
print(x % y)  # Modulus: 1
print(x ** y) # Exponentiation: 1000
print(x // y) # Floor Division: 3
```

---
## 2. Control Flow
### Conditional Statements
```python
num = 10
if num > 0:
    print("Positive number")
elif num == 0:
    print("Zero")
else:
    print("Negative number")
```

### Loops
```python
# For loop
for i in range(5):
    print(i)  # 0 1 2 3 4

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

---
## 3. Functions & Lambda Expressions
### Defining Functions
```python
def add(a, b):
    return a + b

print(add(5, 3))  # Output: 8
```

### Lambda Functions
```python
square = lambda x: x ** 2
print(square(5))  # Output: 25
```

---
## 4. List & Dictionary Comprehensions
```python
# List Comprehension
squares = [x**2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]

# Dictionary Comprehension
double_dict = {x: x * 2 for x in range(5)}
print(double_dict)  # {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
```

---
## 5. Working with NumPy & Pandas
### NumPy Basics
```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr.mean())  # Output: 2.5
```

### Pandas Basics
```python
import pandas as pd

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print(df.head())
```

---
## 6. Basic File Handling
```python
# Writing to a file
with open("sample.txt", "w") as f:
    f.write("Hello, Python!")

# Reading from a file
with open("sample.txt", "r") as f:
    content = f.read()
    print(content)
```

---
## 7. Object-Oriented Programming (OOP) Basics
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "I am an animal"

cat = Animal("Whiskers")
print(cat.speak())  # Output: I am an animal
```

---
## 8. Using Matplotlib for Simple Visualizations
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Plot")
plt.show()
```

---
## 9. Writing Clean, Efficient Python Code
- Follow **PEP 8** style guidelines.
- Use meaningful variable names.
- Keep functions small and focused.
- Use list comprehensions for efficiency.
- Avoid unnecessary loops and redundant code.
- Debug using `print()` or `pdb`.

---
## Conclusion
This guide introduces the essential **Python concepts for ML Engineers**. Mastering these basics will help you build **efficient and scalable machine learning models**.

For more advanced topics, check out additional guides on **NumPy, Scikit-learn, TensorFlow, and PyTorch**!

Happy coding! 🚀
