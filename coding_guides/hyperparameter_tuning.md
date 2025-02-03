# Basic Hyperparameter Tuning Guide for ML Engineers

## Introduction to Hyperparameter Tuning
Hyperparameter tuning is the process of optimizing the configuration settings that control the training process of a machine learning model. Selecting the right hyperparameters improves **model accuracy, generalization, and efficiency**.

---
## 1. What are Hyperparameters?
Hyperparameters are settings that **control the learning process** and are not learned from the training data. Examples include:

- **Learning rate** – Controls step size in gradient descent.
- **Batch size** – Number of training samples per update.
- **Number of epochs** – How many times the model sees the dataset.
- **Regularization strength** – Helps prevent overfitting.

---
## 2. Manual Hyperparameter Tuning
```python
from sklearn.linear_model import LogisticRegression

# Trying different values manually
model = LogisticRegression(C=1.0, solver='liblinear')
model.fit(X_train, y_train)
```
Pros:
- Simple and intuitive.
- Works well for small models.

Cons:
- Time-consuming.
- Not scalable for complex models.

---
## 3. Grid Search
**Grid Search** tests all possible hyperparameter combinations.
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
```
Pros:
- Exhaustive search guarantees finding the best parameters.

Cons:
- Computationally expensive.

---
## 4. Random Search
**Random Search** selects random combinations of hyperparameters.
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {
    'C': uniform(0.1, 10)
}
random_search = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=5, cv=5)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
```
Pros:
- Faster than Grid Search.
- Works well with high-dimensional parameter spaces.

Cons:
- Might miss the optimal set of hyperparameters.

---
## 5. Bayesian Optimization
Bayesian Optimization models hyperparameter tuning as a **probabilistic function**.
```python
from skopt import BayesSearchCV

search = BayesSearchCV(LogisticRegression(), {
    'C': (0.1, 10.0)
}, n_iter=10, cv=5)
search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
```
Pros:
- More efficient than Grid Search and Random Search.
- Works well with complex models.

Cons:
- More challenging to implement.

---
## 6. Hyperparameter Tuning with Deep Learning
### Tuning Learning Rate using Keras
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

for lr in [0.001, 0.01, 0.1]:
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
```

### Using Optuna for Neural Networks
```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return history.history['accuracy'][-1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print("Best Parameters:", study.best_params)
```

---
## 7. Automating Hyperparameter Tuning with Hyperopt
```python
from hyperopt import fmin, tpe, hp, Trials

def objective(params):
    model = LogisticRegression(C=params['C'])
    model.fit(X_train, y_train)
    return -model.score(X_test, y_test)

space = {'C': hp.uniform('C', 0.1, 10)}
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)
print("Best Parameters:", best)
```

---
## Conclusion
Hyperparameter tuning is essential for improving ML models. **Grid Search, Random Search, Bayesian Optimization, and Automated Tuning** all have different trade-offs. Choosing the right technique depends on the problem, dataset size, and computational resources.

For more advanced topics, check out **Neural Architecture Search, Evolutionary Algorithms, and Meta-Learning**!

Happy tuning! 🚀
