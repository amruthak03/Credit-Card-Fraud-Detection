# Credit-Card-Fraud-Detection

This project focuses on detecting fraudulent credit card transactions and explores the root causes behind fraudulent behavior using machine learning techniques. The dataset is highly imbalanced, with fraudulent transactions constituting only 0.17% of all transactions. The goal is to build a robust model pipeline to accurately identify fraudulent transactions while minimizing false positives and false negatives. Fraud detection is crucial to prevent financial losses, safeguard customer trust, and reduce manual review costs for financial institutions.

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
- Interpret the predictions using SHAP values to identify the key factors driving fraud detection.
- Provide actionable insights for root cause analysis to improve fraud prevention systems.

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
   - Trained Logistic Regression, Random Forest, and Decision Tree Classifier on the encoded features.
   - Evaluated performance using precision, recall, F1-score, and confusion matrix.
  
5. SHAP Analysis (Feature Importance)
   - Applied SHAP (SHapley Additive exPlanations) to understand feature contributions.
  
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

Key features like V12, V2, and V4 were identified as strong indicators of fraudulent transactions, allowing for better root cause analysis.

<img width="783" alt="Screenshot 2024-12-16 at 8 01 28 PM" src="https://github.com/user-attachments/assets/b67800d1-9ab7-4178-a388-0eec6e455d34" />


## Technologies Used
- Python: Data analysis, preprocessing, and modeling.
- Keras: Constructing the autoencoder.
- Scikit-learn: Logistic Regression, Random Forest, Decision Tree, scaling, and evaluation metrics.
- Seaborn & Matplotlib: Data visualization.
- t-SNE: Dimensionality reduction for visualization.
