# VGG16 Image Classifier for Binary Classification

## Overview
This project implements a binary image classification model using the VGG16 architecture from TensorFlow and Keras. It leverages transfer learning by using the pre-trained weights from VGG16 trained on ImageNet and adds custom layers for binary classification tasks.

## Features
- Pre-trained VGG16 base model for fast training.
- Custom classification layers for binary classification.
- Data augmentation using Keras' `ImageDataGenerator`.
- Early stopping and model checkpointing to prevent overfitting.
- Evaluation with accuracy/loss plots and confusion matrix.

## Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
## The dataset should be organized as follows:
/dataset
    /train
        /class1
        /class2
    /validation
        /class1
        /class2
    /test
        /class1
        /class2
## How to Use
Organize your dataset into three directories: Training_Dataset, Validation_Dataset, and Test_Dataset, each containing subdirectories for the two classes.
Update the paths in the script to point to your dataset directories.
Run the script to train the model.
## Model Training
The model uses the following configuration:

Architecture: VGG16 (pre-trained on ImageNet) with custom layers added for binary classification.
Input Size: Images are resized to 224x224 pixels.
Batch Size: 32
Loss Function: Binary Cross-Entropy
Optimizer: Adam

## Results
The model outputs both training and validation accuracy and loss for each epoch. After training, the model's performance on the test dataset is displayed, including:

Test Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
