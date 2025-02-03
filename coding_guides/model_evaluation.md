# Basic Model Evaluation Guide for ML Engineers

## Introduction to Model Evaluation
Model evaluation is essential in machine learning to assess a model’s **performance, accuracy, and generalization ability**. It helps in selecting the best model and tuning hyperparameters for better predictions.

---
## 1. Why is Model Evaluation Important?
- Prevents **overfitting** and **underfitting**.
- Measures how well a model **generalizes** to unseen data.
- Helps in selecting the best **hyperparameters**.

---
## 2. Train-Test Split
Splitting data into **training** and **testing** sets ensures the model is evaluated on unseen data.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---
## 3. Classification Metrics
### Accuracy
**Accuracy** measures how many predictions are correct.
```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Precision, Recall, and F1 Score
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

---
## 4. Regression Metrics
### Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error
print("MAE:", mean_absolute_error(y_test, y_pred))
```

### Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")
```

### R² Score
```python
from sklearn.metrics import r2_score
print("R² Score:", r2_score(y_test, y_pred))
```

---
## 5. Cross-Validation
Cross-validation helps in assessing a model's performance across different subsets of data.
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Score:", scores.mean())
```

---
## 6. ROC Curve and AUC Score (for Classification)
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probabilities)
auc_score = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

---
## 7. Model Evaluation in Deep Learning
For deep learning models using **Keras**:
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
```

---
## Conclusion
Model evaluation ensures the model is **accurate, generalizes well, and avoids overfitting**. Choosing the right metrics depends on the **problem type (classification vs regression)**.

For more advanced topics, check out **Feature Importance, Model Interpretability, and A/B Testing**!

Happy evaluating! 🚀
