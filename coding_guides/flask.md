# Basic Flask Guide for ML Engineers

## Introduction to Flask
**Flask** is a lightweight Python web framework that is commonly used for deploying **machine learning models** as APIs. It is easy to set up and works well with **RESTful APIs** for serving ML models.

---
## 1. Installing and Importing Flask
### Installation
If Flask is not installed, use:
```bash
pip install flask
```

### Importing Flask
```python
from flask import Flask, request, jsonify
```

---
## 2. Creating a Simple Flask API
### Basic Flask App
```python
from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask!"

if __name__ == "__main__":
    app.run(debug=True)
```

Run the script and access `http://127.0.0.1:5000/` in your browser.

---
## 3. Creating a Machine Learning API
### Loading a Trained Model
```python
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model.pkl")  # Load your trained ML model

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
```

### Testing the API with cURL
```bash
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---
## 4. Adding Swagger UI for API Documentation
### Installing Flasgger
```bash
pip install flasgger
```

### Integrating Swagger with Flask
```python
from flasgger import Swagger
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
Swagger(app)

model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict using the trained ML model
    ---
    parameters:
      - name: features
        in: body
        required: true
        schema:
          type: object
          properties:
            features:
              type: array
              items:
                type: number
    responses:
      200:
        description: Model prediction
    """
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
```

### Accessing Swagger UI
Once running, open `http://127.0.0.1:5000/apidocs/` to view and interact with the API documentation.

---
## 5. Deploying the Flask App
### Running on a Different Host and Port
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
```

### Deploying on AWS EC2 or Any Cloud Server
1. Install Flask and dependencies on the server.
2. Run `nohup python app.py &` to keep the Flask app running in the background.
3. Use **NGINX** or **Gunicorn** for production.

---
## Conclusion
Flask is an **excellent choice for serving ML models** as REST APIs. Adding **Swagger UI** makes it easier to document and test the API.

For more advanced topics, check out additional guides on **FastAPI, Docker, and Kubernetes**!

Happy coding! 🚀