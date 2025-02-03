# Bone Fracture Detection

## Overview
This project leverages deep learning and computer vision to detect bone fractures in X-ray images. Using a pre-trained ResNet18 model, the goal is to automate fracture detection, potentially improving healthcare efficiency and reducing misdiagnosis.

## Dataset
- **Source:** [Kaggle Bone Fracture Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)
- **Categories:** "Fractured" and "Not Fractured"
- **Format:** X-ray images for training, validation, and testing

## Objective
- Develop an image classification model for detecting bone fractures.
- Use a ResNet18 model with transfer learning for improved accuracy.
- Evaluate performance using loss metrics.

## Steps in the Notebook
1. **Data Exploration**  
   - Visualize sample images from training, validation, and test sets.
2. **Data Preprocessing**  
   - Define a custom dataset class.
   - Apply image transformations (resizing, normalization).
3. **Model Training**  
   - Load pre-trained ResNet18 weights.
   - Modify the final layer for binary classification.
   - Train the model using binary cross-entropy loss.
4. **Evaluation**  
   - Compute training and validation loss.
   - Visualize model performance.

## Implementation Details
- **Tools Used:** PyTorch, Torchvision, Matplotlib, PIL
- **Training Setup:**
  - Training images processed with `torch.utils.data.Dataset` and `DataLoader`.
  - Model trained on GPU if available.
  - Binary cross-entropy loss for classification.
  - Adam optimizer for weight updates.

## Results
- Training and validation loss monitored across epochs.
- Future improvements:
  - Data augmentation to enhance model generalization.
  - Hyperparameter tuning for better accuracy.
