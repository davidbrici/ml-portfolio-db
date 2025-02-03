# Exploratory Data Analysis (EDA) Guide for ML Engineers

## Introduction to EDA
**Exploratory Data Analysis (EDA)** is a crucial step in the ML pipeline to **understand, visualize, and preprocess data** before modeling. EDA helps in detecting missing values, identifying patterns, and ensuring data quality.

---
## 1. Setting Up the Environment
### Installing Required Libraries
```bash
pip install pandas numpy matplotlib seaborn
```

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---
## 2. Loading the Dataset
```python
df = pd.read_csv("data.csv")
print(df.head())  # Display the first 5 rows
```

### Checking Data Shape
```python
print(df.shape)  # (rows, columns)
```

### Checking Column Data Types
```python
print(df.dtypes)
```

---
## 3. Handling Missing Values
### Checking for Missing Values
```python
print(df.isnull().sum())
```

### Filling Missing Values
```python
df.fillna(df.mean(), inplace=True)  # Fill numerical columns with mean
```

### Dropping Rows with Missing Values
```python
df.dropna(inplace=True)
```

---
## 4. Descriptive Statistics
### Summary Statistics
```python
print(df.describe())
```

### Checking Unique Values in Categorical Columns
```python
print(df['Category_Column'].unique())
```

### Value Counts for Categorical Data
```python
print(df['Category_Column'].value_counts())
```

---
## 5. Data Visualization
### Histogram for Distribution
```python
plt.hist(df['Numerical_Column'], bins=20)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Numerical Column")
plt.show()
```

### Box Plot for Outliers
```python
sns.boxplot(x=df['Numerical_Column'])
plt.title("Box Plot of Numerical Column")
plt.show()
```

### Correlation Heatmap
```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()
```

### Pairplot for Relationships
```python
sns.pairplot(df, hue='Category_Column')
plt.show()
```

---
## 6. Detecting and Handling Outliers
### Using IQR Method
```python
Q1 = df['Numerical_Column'].quantile(0.25)
Q3 = df['Numerical_Column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Numerical_Column'] >= lower_bound) & (df['Numerical_Column'] <= upper_bound)]
```

### Using Z-score
```python
from scipy import stats
df = df[(np.abs(stats.zscore(df['Numerical_Column'])) < 3)]
```

---
## 7. Feature Engineering
### Creating New Features
```python
df['New_Feature'] = df['Column1'] * df['Column2']
```

### Encoding Categorical Data
```python
df = pd.get_dummies(df, columns=['Category_Column'], drop_first=True)
```

### Normalization and Scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Numerical_Column']] = scaler.fit_transform(df[['Numerical_Column']])
```

---
## 8. Splitting Data for Model Training
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---
## Conclusion
EDA is essential for **understanding data distribution, handling missing values, detecting outliers, and feature engineering**. A well-executed EDA improves **model performance and interpretability**.

For more advanced topics, check out **Feature Selection, PCA, and Data Augmentation**!

Happy exploring! 🚀
