# Quality Inspection

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objective)
3. [Conclusion](#conclusion)
4. [Technologies Used](#technologies-used)
5. [How to Use](#how-to-use)

## Overview

This project focuses on developing a machine learning model for quality inspection in a production environment. The primary goal is to classify images of products or objects to determine whether they meet quality standards or have defects. The images are analyzed using various image processing and machine learning techniques to automate the inspection process.

The dataset consists of images of products, where each image is labeled as either "defective" or "non-defective." The task involves creating a classification model to detect defects based on the visual data. This project employs PyTorch and PyTorch Lightning for model development and training, along with Optuna for hyperparameter tuning to find the best-performing model.

## Objective

The main objectives of this project are as follows:

1. Data Preprocessing:

   - Load and preprocess the quality inspection images.
   - Perform data augmentation techniques to improve model generalization.
   - Split the dataset into training, validation, and test sets for proper model evaluation.

2. Model Architecture:

   - Develop a deep learning model using Convolutional Neural Networks (CNN) to perform defect classification.
   - Use transfer learning or custom architectures to improve model performance.
   - Implement various activation functions, batch sizes, and dropout ratios to optimize model performance.

3. Hyperparameter Tuning:

   - Use Optuna for hyperparameter optimization, tuning parameters like the learning rate, optimizer type, and layer configurations.

4. Model Training:

   - Train the model using PyTorch Lightning for simplified workflow management.
   - Implement GPU acceleration to speed up model training using cloud-based environments.

5. Evaluation:

   - Evaluate model performance using accuracy, confusion matrix, and classification report metrics.
   - Analyze performance on the test set to check for misclassifications, especially focusing on defect types.

6. Future Improvements:
   - Suggest further improvements such as using deeper architectures, advanced optimizers, or ensemble models.
   - Explore real-time quality inspection for deployment in production lines.

## Conclusion

Key findings from the project:

1. Model Performance:

   - The best model achieved an accuracy of 95% on the test dataset, with a high F1-score for both defective and non-defective classes.
   - The optimal hyperparameters included an AdamW optimizer, a learning rate of 0.001, and a batch size of 32.

2. Insights from Hyperparameter Tuning:

   - Hyperparameter tuning had a significant impact on model performance, especially the choice of optimizer and learning rate.
   - Data augmentation helped in improving generalization, preventing overfitting.

3. Confusion Matrix and Evaluation:

   - The confusion matrix showed that the model successfully identified most defective items but had some difficulties with subtle defects, which could be improved with more data or advanced models.
   - The classification report showed a balanced performance between precision and recall for both classes.

4. Model Improvements:
   - Future improvements could include experimenting with more complex architectures like ResNet or EfficientNet.
   - Additional data from different defect types and environments could improve the model's robustness.

## Technologies Used

- Python 3.x: The primary programming language used for model development.
- PyTorch: For implementing deep learning models, particularly CNNs.
- PyTorch Lightning: For simplifying training workflows and ensuring reproducibility.
- Optuna: For performing hyperparameter tuning to improve model performance.
- OpenCV & scikit-image: For image preprocessing and enhancement.
- Matplotlib & Seaborn: For data visualization and model performance evaluation.
- scikit-learn: For evaluation metrics (accuracy, precision, recall, confusion matrix).

## How to Use

### Prerequisites

- Python 3.8 or later
- pip for managing Python packages

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <module-folder>
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
