# Basic Computer Vision (CV) Guide for ML Engineers

## Introduction to Computer Vision
**Computer Vision (CV)** is a subfield of AI that enables machines to interpret and process visual data. CV is widely used in **image classification, object detection, facial recognition, medical imaging, and more.**

---
## 1. Setting Up the Environment
### Installing Required Libraries
```bash
pip install opencv-python numpy matplotlib torch torchvision tensorflow
```

### Importing Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model
```

---
## 2. Loading and Displaying Images
### Loading an Image with OpenCV
```python
image = cv2.imread("sample.jpg")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Displaying an Image with Matplotlib
```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
```

---
## 3. Image Preprocessing
### Resizing an Image
```python
resized_image = cv2.resize(image, (224, 224))
```

### Converting to Grayscale
```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### Normalizing an Image
```python
normalized_image = image / 255.0
```

---
## 4. Image Augmentation
### Applying Basic Transformations with OpenCV
```python
flipped = cv2.flip(image, 1)  # Horizontal flip
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
```

### Using PyTorch for Augmentation
```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])
```

---
## 5. Object Detection
### Using Pre-trained Haar Cascade Classifier
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Using YOLO for Object Detection
```python
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
```

---
## 6. Image Classification
### Using a Pretrained CNN Model (TensorFlow)
```python
model = load_model("model.h5")
image = cv2.resize(image, (224, 224))
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
print("Predicted Class:", np.argmax(prediction))
```

### Using PyTorch Pretrained Model
```python
import torchvision.models as models
model = models.resnet18(pretrained=True)
model.eval()
```

---
## 7. Edge Detection
### Canny Edge Detection
```python
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## 8. Facial Recognition
### Face Detection with OpenCV DNN
```python
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
```

---
## 9. Optical Character Recognition (OCR)
### Using Tesseract OCR
```bash
pip install pytesseract
```
```python
import pytesseract
text = pytesseract.image_to_string(gray_image)
print(text)
```

---
## 10. Working with Video
### Capturing Video from Webcam
```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---
## Conclusion
Computer Vision is a powerful tool for **image processing, object detection, and feature extraction**. Understanding **image preprocessing, deep learning models, and real-time applications** is crucial for ML engineers.

For more advanced topics, check out **Deep Learning for CV, Object Tracking, and Generative Models**!

Happy coding! 🚀