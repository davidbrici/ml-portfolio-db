# OpenCV Guide for ML Engineers

## Introduction to OpenCV
**OpenCV (Open Source Computer Vision Library)** is a popular library for **image processing, computer vision, and real-time applications**. ML engineers use OpenCV for tasks like **image augmentation, feature extraction, object detection, and face recognition**.

---
## 1. Installing OpenCV
```bash
pip install opencv-python
pip install opencv-python-headless  # If working on a server
```

### Importing OpenCV
```python
import cv2
import numpy as np
```

---
## 2. Loading and Displaying Images
### Loading an Image
```python
image = cv2.imread("sample.jpg")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Checking Image Properties
```python
print(image.shape)  # (height, width, channels)
print(image.dtype)  # Data type of image pixels
```

---
## 3. Resizing and Cropping
### Resizing an Image
```python
resized_image = cv2.resize(image, (300, 300))
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Cropping an Image
```python
cropped_image = image[50:200, 50:200]  # [y1:y2, x1:x2]
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## 4. Converting Color Spaces
### Converting to Grayscale
```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Converting to HSV
```python
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## 5. Drawing on Images
### Drawing a Line
```python
cv2.line(image, (50, 50), (200, 200), (0, 255, 0), 3)
cv2.imshow("Line Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Drawing a Rectangle
```python
cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), 2)
cv2.imshow("Rectangle Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Adding Text
```python
cv2.putText(image, "Hello OpenCV", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("Text Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## 6. Image Filtering
### Blurring an Image
```python
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Edge Detection with Canny
```python
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## 7. Feature Detection
### Detecting Corners
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow("Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Detecting Faces using Haar Cascade
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
## 8. Video Processing
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

### Saving Video
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
```

---
## Conclusion
OpenCV is a **powerful library** for **image processing, feature extraction, and real-time applications** in ML workflows. Mastering OpenCV enables ML engineers to **process visual data efficiently** before feeding it into deep learning models.

For more advanced topics, check out **Deep Learning with OpenCV, Object Detection with YOLO, and OpenCV with TensorFlow**!

Happy coding! 🚀
