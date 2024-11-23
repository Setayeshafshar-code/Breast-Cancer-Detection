# Breast-Cancer-Detection

Breast Cancer Binary Classification Project

This project focuses on solving a binary classification problem related to breast cancer. The goal is to predict whether a tumor is malignant (M) or benign (B) using the Breast Cancer Wisconsin Dataset. Two models were tested: Logistic Regression and Decision Tree Classifier, to evaluate their performance and determine the best fit for this problem.

Features:
-Data preprocessing and visualization.
-Model training with Logistic Regression and Decision Tree Classifier.
-Model evaluation using accuracy, confusion matrix, and classification report.
-Deployment using Flask for real-time predictions.
-Containerization and deployment to the cloud.

Problem Statement:
The dataset contains features extracted from digitized images of breast mass. The task is to build a machine learning model that can accurately classify tumors into two categories:

0: Benign
1: Malignant

# Deployment
Run the following command in the terminal to build the Docker Image :

```sudo docker build -t breast_cancer_app .```

run the Docker container and map it to port ```5000```:
```sudo docker run -p 5000:5000 breast_cancer_app```

This will:
1. Start the Flask app inside the Docker container
2. Test the form submission adn ensure prediction are working.