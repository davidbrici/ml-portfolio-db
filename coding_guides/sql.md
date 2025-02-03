# SQL Guide for ML Engineers

## Introduction to SQL for ML Engineers
**SQL (Structured Query Language)** is essential for **data extraction, transformation, and analysis** in machine learning workflows. ML engineers often use SQL databases like **MySQL, PostgreSQL, SQLite, and BigQuery** to store and manipulate structured data.

---
## 1. Setting Up SQL Databases
### Installing SQLite (Lightweight Database)
```bash
pip install sqlite3
```

### Connecting to SQLite Database
```python
import sqlite3

conn = sqlite3.connect("ml_data.db")
cursor = conn.cursor()
```

### Connecting to MySQL Database
```python
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="ml_database"
)
cursor = conn.cursor()
```

---
## 2. Creating and Managing Tables
### Creating a Table
```sql
CREATE TABLE dataset (
    id INT PRIMARY KEY,
    name TEXT,
    age INT,
    salary FLOAT
);
```

### Inserting Data
```sql
INSERT INTO dataset (id, name, age, salary) VALUES (1, 'Alice', 30, 70000);
INSERT INTO dataset (id, name, age, salary) VALUES (2, 'Bob', 35, 80000);
```

### Updating Data
```sql
UPDATE dataset SET salary = 85000 WHERE name = 'Bob';
```

### Deleting Data
```sql
DELETE FROM dataset WHERE id = 2;
```

---
## 3. Querying Data
### Selecting All Data
```sql
SELECT * FROM dataset;
```

### Filtering Data
```sql
SELECT * FROM dataset WHERE age > 30;
```

### Sorting Data
```sql
SELECT * FROM dataset ORDER BY salary DESC;
```

### Counting Rows
```sql
SELECT COUNT(*) FROM dataset;
```

### Grouping and Aggregation
```sql
SELECT age, AVG(salary) FROM dataset GROUP BY age;
```

---
## 4. Using SQL for ML Workflows
### Loading SQL Data into Pandas
```python
import pandas as pd

conn = sqlite3.connect("ml_data.db")
df = pd.read_sql("SELECT * FROM dataset", conn)
print(df.head())
```

### Querying with Pandas SQL
```python
from pandasql import sqldf

query = "SELECT * FROM df WHERE age > 30;"
filtered_df = sqldf(query, locals())
print(filtered_df)
```

### Exporting SQL Query Results to CSV
```python
df.to_csv("ml_dataset.csv", index=False)
```

---
## 5. Advanced SQL Queries
### Joining Tables
```sql
SELECT employees.name, departments.department_name
FROM employees
JOIN departments ON employees.dept_id = departments.id;
```

### Subqueries
```sql
SELECT name, salary FROM dataset
WHERE salary > (SELECT AVG(salary) FROM dataset);
```

### Window Functions (Running Totals, Ranking)
```sql
SELECT name, salary, RANK() OVER (ORDER BY salary DESC) AS salary_rank FROM dataset;
```

---
## 6. Indexing and Performance Optimization
### Creating an Index for Faster Queries
```sql
CREATE INDEX idx_salary ON dataset (salary);
```

### Using EXPLAIN to Analyze Query Performance
```sql
EXPLAIN QUERY PLAN SELECT * FROM dataset WHERE salary > 70000;
```

---
## 7. Automating SQL Queries in Python
### Executing SQL Queries in Python
```python
query = "SELECT * FROM dataset WHERE age > 30;"
cursor.execute(query)
results = cursor.fetchall()
for row in results:
    print(row)
```

### Running Batch Insert Queries
```python
data = [(3, 'Charlie', 40, 90000), (4, 'David', 25, 60000)]
cursor.executemany("INSERT INTO dataset (id, name, age, salary) VALUES (?, ?, ?, ?)", data)
conn.commit()
```

---
## Conclusion
SQL is a powerful tool for **querying, processing, and analyzing structured data** in ML workflows. Using SQL efficiently allows ML engineers to extract insights and prepare datasets before training models.

For more advanced topics, check out **BigQuery, NoSQL Databases, and SQL for Feature Engineering**!

Happy querying! 🚀
