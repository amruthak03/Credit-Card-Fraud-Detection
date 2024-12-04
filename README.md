# Credit-Card-Fraud-Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, with fraudulent transactions constituting only 0.17% of all transactions. The goal is to build a robust model pipeline to accurately identify fraudulent transactions while minimizing false positives and false negatives.

Table of Contents
- Dataset
- Problem Statement
- Approach
- Results
- Technologies Used

## Dataset
The dataset was created by the Machine Learning Group - ULB and contains:
- 284,807 transactions over two days (September 2013 by European cardholders).
- 492 fraudulent transactions (0.172% of all transactions).
- Includes anonymized features (V1, V2, ..., V28), along with Time, Amount, and the target label Class (0: Non-Fraud, 1: Fraud).

## Problem Statement
Fraudulent transactions are rare but can cause significant financial damage. The challenge is to build a model that:
- Handles the class imbalance effectively.
- Detects fraud with high precision and recall to reduce financial losses and avoid unnecessary manual interventions.

## Approach
1. Data Preprocessing
   - Scaled the Time and Amount features using StandardScaler.
   - Handled class imbalance using undersampling.
  
2. Exploratory Data Analysis (EDA)
   - Visualized transaction distribution over time for fraudulent and non-fraudulent transactions.
   - Analyzed feature correlations and performed IQR analysis for outliers.
  
3. Dimensionality Reduction
   - Constructed an autoencoder to learn a compact feature representation.
   - Visualized the reduced dimensions using t-SNE, identifying clear clusters for fraud vs. non-fraud.
  
4. Machine Learning Models
   - Trained Logistic Regression and Decision Tree Classifier on the encoded features.
   - Evaluated performance using precision, recall, F1-score, and confusion matrix.
  
## Results
Logistic Regression
- Accuracy: 94%
- Precision (Fraud): 98%
- Recall (Fraud): 91%

Decision Tree
- Accuracy: 88%
- Precision (Fraud): 86%
- Recall (Fraud): 90%
  
The autoencoder successfully captured hidden patterns, enabling the models to differentiate between fraudulent and non-fraudulent transactions effectively.

## Technologies Used
- Python: Data analysis, preprocessing, and modeling.
- Keras: Constructing the autoencoder.
- Scikit-learn: Logistic Regression, Decision Tree, scaling, and evaluation metrics.
- Seaborn & Matplotlib: Data visualization.
- t-SNE: Dimensionality reduction for visualization.
