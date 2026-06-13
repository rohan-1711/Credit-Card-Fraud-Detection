# Credit Card Fraud Detection System

## Description
An ML-based fraud detection system that identifies fraudulent credit card transactions using Logistic Regression and Random Forest classifiers. Handles imbalanced data using SMOTE oversampling.

## Tech Stack
- Python 3
- scikit-learn (Logistic Regression, Random Forest)
- imbalanced-learn (SMOTE for class imbalance)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)

## Problem Statement
Credit card fraud is rare but costly. Standard ML models fail because:
- **Class Imbalance**: 99% legitimate, 1% fraud
- **Cost Sensitivity**: False negatives (missed fraud) are worse than false positives

## Solution
- **SMOTE Oversampling**: Creates synthetic fraud cases to balance training data
- **Multiple Models**: Trains Logistic Regression and Random Forest
- **Proper Metrics**: Uses AUC-ROC, AUC-PR, F1-score instead of just accuracy

## Dataset
**Synthetic Credit Card Dataset**
- Total transactions: 10,000
- Legitimate (Class 0): 9,900 (99%)
- Fraudulent (Class 1): 100 (1%)
- Features: 10 numerical features
- Train/Test Split: 80/20

## Pipeline
1. **Data Generation**: Create synthetic transactions
2. **Feature Scaling**: StandardScaler normalization
3. **SMOTE**: Balance training data (99% → 50-50)
4. **Model Training**: Logistic Regression & Random Forest
5. **Evaluation**: AUC-ROC, confusion matrix, classification report
6. **Visualization**: ROC curves, confusion matrix heatmap

## Model Performance

### Logistic Regression
- **Accuracy**: 99.95%
- **F1-Score**: 0.9756
- **AUC-ROC**: 1.0000 (Perfect)
- **True Positives**: 20/20 (caught all fraud)
- **False Positives**: 1 (minimal false alarms)

### Random Forest
- **Accuracy**: 99.85%
- **F1-Score**: 0.9189
- **AUC-ROC**: 0.9999 (Near perfect)
- **True Positives**: 17/20
- **False Positives**: 0 (no false alarms)

## Key Metrics Explained
- **True Positive (TP)**: Correctly detected fraud ✅
- **False Negative (FN)**: Missed fraud (dangerous) ❌
- **False Positive (FP)**: False alarm (annoying but safe) ⚠️
- **AUC-ROC**: Area under ROC curve (higher = better) — measures model discrimination ability
- **F1-Score**: Balance between precision and recall

## How to Run
```bash
python credit_card_fraud_detector.py
```

## Output Files
- `fraud_detection_evaluation.png` - ROC curves & confusion matrix
- `fraud_detection_results.csv` - Model comparison table

## Real-World Applications
- **Payment Risk Mitigation**: Real-time transaction scoring
- **Fraud Prevention**: Flag suspicious transactions for manual review
- **Cost Reduction**: Prevent chargebacks and fraud losses
- **Customer Trust**: Protect cardholders from unauthorized charges

## Learning Outcomes
✅ Handling class imbalance (SMOTE)  
✅ Training multiple ML models  
✅ Feature scaling & normalization  
✅ Proper evaluation metrics for imbalanced data  
✅ ROC curves and AUC-PR analysis  
✅ Cross-validation & model comparison  
✅ Business metrics (cost-sensitive evaluation)  

## Author
Rohan Gunjal - MIT-ADT University, Pune  
Slash Mark Internship - Task 4
