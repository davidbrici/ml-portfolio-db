# Pillow Guide for ML Engineers

## Introduction to Pillow
**Pillow** is a powerful Python library for **image processing**. It is widely used by **ML engineers** for tasks such as **image resizing, format conversion, filtering, and augmentation**.

---
## 1. Installing Pillow
```bash
pip install pillow
```

### Importing Pillow
```python
from PIL import Image
```

---
## 2. Opening and Displaying Images
### Opening an Image
```python
image = Image.open("sample.jpg")
image.show()
```

### Checking Image Properties
```python
print(image.format)  # JPEG, PNG, etc.
print(image.size)    # (width, height)
print(image.mode)    # RGB, CMYK, etc.
```

---
## 3. Resizing and Cropping
### Resizing an Image
```python
resized_image = image.resize((200, 200))
resized_image.show()
```

### Cropping an Image
```python
cropped_image = image.crop((50, 50, 200, 200))  # (left, top, right, bottom)
cropped_image.show()
```

---
## 4. Rotating and Flipping
### Rotating an Image
```python
rotated_image = image.rotate(45)
rotated_image.show()
```

### Flipping an Image
```python
flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
flipped_image.show()
```

---
## 5. Converting Image Formats
### Converting to Grayscale
```python
grayscale_image = image.convert("L")
grayscale_image.show()
```

### Saving in a Different Format
```python
image.save("output.png", format="PNG")
```

---
## 6. Drawing on Images
### Adding Text and Shapes
```python
from PIL import ImageDraw, ImageFont

draw = ImageDraw.Draw(image)
draw.text((50, 50), "ML Engineer", fill="white")
image.show()
```

---
## 7. Applying Filters
### Using Built-in Filters
```python
from PIL import ImageFilter

blurred_image = image.filter(ImageFilter.BLUR)
blurred_image.show()
```

### Enhancing Sharpness
```python
from PIL import ImageEnhance

sharpness = ImageEnhance.Sharpness(image)
enhanced_image = sharpness.enhance(2.0)  # Increase sharpness
enhanced_image.show()
```

---
## 8. Image Augmentation for ML
### Applying Multiple Transformations
```python
from PIL import ImageOps

image = ImageOps.autocontrast(image)  # Auto contrast adjustment
image = ImageOps.equalize(image)  # Equalize histogram
image.show()
```

### Creating Image Variations
```python
import random

def random_transform(image):
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        image = image.rotate(random.randint(0, 360))
    return image

augmented_image = random_transform(image)
augmented_image.show()
```

---
## 9. Extracting Image Metadata
```python
from PIL.ExifTags import TAGS

exif_data = image._getexif()
if exif_data:
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        print(f"{tag_name}: {value}")
```

---
## 10. Batch Processing Multiple Images
```python
import os

directory = "images/"
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(directory, filename))
        img_resized = img.resize((200, 200))
        img_resized.save(os.path.join("output/", filename))
```

---
## Conclusion
Pillow is a **versatile tool** for **image preprocessing, augmentation, and analysis** in ML workflows. Mastering Pillow enables ML engineers to **efficiently handle image datasets** before feeding them into models.

For more advanced topics, check out **OpenCV, TensorFlow Image Processing, and PyTorch Data Loaders**!

Happy coding! 🚀