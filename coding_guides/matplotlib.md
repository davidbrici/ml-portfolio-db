# Basic Matplotlib Guide for ML Engineers

## Introduction to Matplotlib
**Matplotlib** is a powerful Python library for data visualization. It allows ML engineers to **create plots, histograms, scatter plots, bar charts, and more**, making it essential for **understanding data and model performance**.

---
## 1. Installing and Importing Matplotlib
### Installation
If Matplotlib is not installed, use:
```bash
pip install matplotlib
```

### Importing Matplotlib
```python
import matplotlib.pyplot as plt
```

---
## 2. Creating a Basic Line Plot
```python
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 50]

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Basic Line Plot")
plt.show()
```

---
## 3. Customizing Plots
### Changing Line Styles & Colors
```python
plt.plot(x, y, linestyle="--", marker="o", color="r", label="Data")
plt.legend()
plt.show()
```

### Adding Grid
```python
plt.plot(x, y)
plt.grid(True)
plt.show()
```

---
## 4. Scatter Plot
```python
plt.scatter(x, y, color='g', marker='o')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Scatter Plot")
plt.show()
```

---
## 5. Bar Chart
```python
categories = ["A", "B", "C", "D"]
values = [10, 20, 15, 25]

plt.bar(categories, values, color='b')
plt.xlabel("Categories")
plt.ylabel("Values")
plt.title("Bar Chart Example")
plt.show()
```

---
## 6. Histogram
```python
import numpy as np

data = np.random.randn(1000)
plt.hist(data, bins=30, color='purple', alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram Example")
plt.show()
```

---
## 7. Subplots
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y, color='r')
axes[0].set_title("Line Plot")

axes[1].bar(categories, values, color='g')
axes[1].set_title("Bar Chart")

plt.show()
```

---
## 8. Pie Chart
```python
labels = ['A', 'B', 'C', 'D']
sizes = [10, 20, 30, 40]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['red', 'blue', 'green', 'yellow'])
plt.title("Pie Chart Example")
plt.show()
```

---
## 9. Adding Annotations
```python
plt.plot(x, y, marker="o")
plt.annotate("Peak", xy=(5, 50), xytext=(3, 40), arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()
```

---
## 10. Saving Figures
```python
plt.plot(x, y)
plt.savefig("plot.png", dpi=300)
plt.show()
```

---
## Conclusion
Matplotlib is an essential library for **data visualization in ML workflows**. Mastering it helps in **analyzing data distributions, model predictions, and overall performance**.

For more advanced topics, check out additional guides on **Seaborn, NumPy, Pandas, and Scikit-learn**!

Happy plotting! 🚀