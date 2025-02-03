# Hand Gesture Recognition Using Transfer Learning

## Overview
This project demonstrates a hand gesture recognition model using transfer learning with MobileNetV2. The model classifies hand gestures into 20 distinct categories by leveraging pre-trained MobileNetV2 features for efficient and accurate classification. The workflow includes data handling, exploration, preprocessing, model training, evaluation, and exporting.

## Features
- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet.
- **Custom Layers**: Fine-tuned for gesture classification.
- **Data Exploration**: Includes visualizations, quality checks, and class distribution analysis.
- **Data Augmentation**: Enhances dataset diversity.
- **Comprehensive Evaluation**: Confusion matrix, accuracy metrics, and prediction analysis.
- **Exportable Model**: Easily saves and reloads the trained model for deployment.

## Dataset
The dataset comprises 24,000 images across 20 gesture classes, with 900 images per class for training and 300 images per class for testing.

**Download Dataset**: [Hand Gesture Recognition Dataset](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset)

## Installation
### Prerequisites
- Python 3.8 or later
- Virtual environment (recommended)

### Steps
1. Clone the repository and navigate to the directory:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Create and activate a virtual environment:
   ``` bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
   ```

3.  Install the required dependencies:
   ```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn 
   ```

## Workflow

### 1. Data Handling
- Load images from directories, resize to 224x224 pixels, and split into training, validation, and test sets.
- Extract class names for visualizations and analysis.

### 2. Inspecting Data
- Visualize raw, unaugmented images to assess dataset quality.
- Analyze class distributions to ensure balanced representation.

### 3. Data Preprocessing
- Normalize pixel values to [0, 1].
- Apply augmentations (random flips, rotations, zooms) to enhance diversity.

### 4. Model Building
- Use MobileNetV2 as the base model for feature extraction.
- Add dense, dropout, and softmax layers for classification.

### 5. Model Training
- Batch Processing by divided data into mini-batches for efficient training.
- Training runs for 5 iterations over the entire dataset.

### 6. Model Evaluation
- Evaluate performance on the test dataset using accuracy and loss metrics.
- Visualize a confusion matrix to identify class-level performance.
- Analyze predictions to identify misclassifications and areas for improvement.

**Training Performance:**

| Metric      | Training Set      | Validation Set | Test Set      |
| ----------- | ----------- | ----------- | ----------- |
| Accuracy      | 99%       | 99%       | 97%       |
| Loss   | 0.02        | 0.03        | 0.07        |


### 7. Exporting the Model
- Save the trained model in HDF5 format for future use:
  ```bash
  model.save('hand_gesture_model.h5')
   ```

## Results
The model achieved high test accuracy, though some misclassifications occurred between visually similar gestures. These areas can be improved through better data augmentation and model optimization.

## Future Work
- Add more gesture classes to the dataset.
- Experiment with alternative architectures for improved accuracy.
- Deploy the model in real-time applications (e.g., mobile apps or web interfaces).
- Optimize hyperparameters to refine model performance.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **Dataset**: Arya Rishabh on Kaggle  
- **Pre-trained Model**: MobileNetV2 from TensorFlow/Keras
