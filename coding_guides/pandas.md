# Basic Pandas Guide for ML Engineers

## Introduction to Pandas
**Pandas** is a powerful data analysis and manipulation library in Python. It provides flexible **data structures (Series & DataFrame)** that allow easy handling of structured data, making it an essential tool for **Machine Learning workflows**.

---
## 1. Installing and Importing Pandas
### Installation
If Pandas is not installed, use:
```bash
pip install pandas
```

### Importing Pandas
```python
import pandas as pd
```

---
## 2. Pandas Data Structures
### Creating a Series (1D Array-like)
```python
s = pd.Series([10, 20, 30, 40])
print(s)
# Output:
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64
```

### Creating a DataFrame (2D Table-like)
```python
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Salary": [50000, 60000, 70000]
}
df = pd.DataFrame(data)
print(df)
```

---
## 3. Reading & Writing Data
### Reading CSV
```python
df = pd.read_csv("data.csv")
print(df.head())  # First 5 rows
```

### Writing to CSV
```python
df.to_csv("output.csv", index=False)
```

### Reading Excel
```python
df = pd.read_excel("data.xlsx")
```

### Writing to Excel
```python
df.to_excel("output.xlsx", index=False)
```

---
## 4. Exploring Data
```python
print(df.shape)   # (rows, columns)
print(df.info())  # Column details
print(df.describe())  # Summary statistics
print(df.columns)  # List column names
print(df.dtypes)  # Data types of each column
```

---
## 5. Selecting & Filtering Data
### Selecting Columns
```python
print(df["Name"])  # Single column
print(df[["Name", "Age"]])  # Multiple columns
```

### Selecting Rows by Index
```python
print(df.iloc[0])  # First row
print(df.iloc[1:3])  # Slicing rows
```

### Selecting Rows by Condition
```python
filtered_df = df[df["Age"] > 25]
print(filtered_df)
```

---
## 6. Handling Missing Data
### Checking for Missing Values
```python
print(df.isnull().sum())
```

### Dropping Missing Values
```python
df.dropna(inplace=True)  # Remove rows with NaN
```

### Filling Missing Values
```python
df.fillna(value={"Age": df["Age"].mean()}, inplace=True)  # Fill with mean
```

---
## 7. Data Manipulation
### Adding a New Column
```python
df["Bonus"] = df["Salary"] * 0.10
print(df.head())
```

### Renaming Columns
```python
df.rename(columns={"Age": "Years"}, inplace=True)
```

### Dropping Columns
```python
df.drop(columns=["Bonus"], inplace=True)
```

### Sorting Data
```python
df.sort_values(by="Salary", ascending=False, inplace=True)
```

---
## 8. Grouping & Aggregation
### Grouping by a Column
```python
grouped = df.groupby("Age")["Salary"].mean()
print(grouped)
```

### Applying Multiple Aggregations
```python
agg_df = df.groupby("Age").agg({"Salary": ["mean", "sum"]})
print(agg_df)
```

---
## 9. Merging & Joining Data
### Concatenating DataFrames
```python
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
concat_df = pd.concat([df1, df2], ignore_index=True)
print(concat_df)
```

### Merging DataFrames
```python
df1 = pd.DataFrame({"ID": [1, 2], "Name": ["Alice", "Bob"]})
df2 = pd.DataFrame({"ID": [1, 2], "Salary": [50000, 60000]})
merged_df = pd.merge(df1, df2, on="ID")
print(merged_df)
```

---
## 10. Pivot Tables & Crosstab
### Pivot Table
```python
pivot_df = df.pivot_table(values="Salary", index="Age", aggfunc="mean")
print(pivot_df)
```

### Crosstab
```python
cross_tab = pd.crosstab(df["Age"], df["Salary"])
print(cross_tab)
```

---
## 11. Applying Functions
### Applying a Function to a Column
```python
df["Updated_Salary"] = df["Salary"].apply(lambda x: x * 1.05)
```

### Applying a Function Row-wise
```python
def custom_function(row):
    return row["Age"] * 2

df["Double_Age"] = df.apply(custom_function, axis=1)
```

---
## Conclusion
Pandas is an **indispensable tool** for **data analysis and preprocessing** in ML. Mastering Pandas helps in **efficiently handling large datasets** and preparing them for machine learning models.

For more advanced topics, check out additional guides on **NumPy, Scikit-learn, TensorFlow, and PyTorch**!

Happy coding! 🚀