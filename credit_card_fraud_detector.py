# ============================================
# Credit Card Fraud Detection System
# Slash Mark Internship - Task 4
# Name: Rohan Gunjal
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, auc, 
                             precision_recall_curve, f1_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CREDIT CARD FRAUD DETECTION SYSTEM")
print("="*70)

# ---- STEP 1: Generate Synthetic Dataset ----
print("\n📥 Generating synthetic credit card dataset...")
np.random.seed(42)

# Generate legitimate transactions (majority class - 0)
n_legitimate = 9900
legitimate_features = np.random.randn(n_legitimate, 10)
legitimate_labels = np.zeros(n_legitimate)

# Generate fraudulent transactions (minority class - 1)
n_fraud = 100
fraud_features = np.random.randn(n_fraud, 10) + 2  # Frauds have different patterns
fraud_labels = np.ones(n_fraud)

# Combine and shuffle
X = np.vstack([legitimate_features, fraud_features])
y = np.hstack([legitimate_labels, fraud_labels])

# Shuffle
shuffle_idx = np.random.permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"✅ Total transactions: {len(y)}")
print(f"✅ Legitimate (Class 0): {sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
print(f"✅ Fraudulent (Class 1): {sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%)")

# ---- STEP 2: Split Data ----
print("\n🔀 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# ---- STEP 3: Feature Scaling ----
print("\n⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- STEP 4: Handle Class Imbalance with SMOTE ----
print("\n🔧 Handling imbalanced data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"✅ After SMOTE:")
print(f"   Legitimate (Class 0): {sum(y_train_smote==0)}")
print(f"   Fraudulent (Class 1): {sum(y_train_smote==1)}")

# ---- STEP 5: Train Models ----
print("\n🚀 Training models...")

# Model 1: Logistic Regression
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_smote, y_train_smote)

# Model 2: Random Forest
print("  Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_smote, y_train_smote)

# ---- STEP 6: Make Predictions ----
print("\n🎯 Making predictions on test set...")
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# ---- STEP 7: Evaluate Models ----
print("\n" + "="*70)
print("MODEL EVALUATION RESULTS")
print("="*70)

print("\n📊 LOGISTIC REGRESSION:")
lr_confusion = confusion_matrix(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc_roc = roc_auc_score(y_test, lr_pred_proba)
print(f"  ✅ Accuracy: {(lr_pred == y_test).sum() / len(y_test) * 100:.2f}%")
print(f"  ✅ F1-Score: {lr_f1:.4f}")
print(f"  ✅ AUC-ROC: {lr_auc_roc:.4f}")
print(f"  ✅ Confusion Matrix:")
print(f"     True Negatives:  {lr_confusion[0,0]}")
print(f"     False Positives: {lr_confusion[0,1]}")
print(f"     False Negatives: {lr_confusion[1,0]}")
print(f"     True Positives:  {lr_confusion[1,1]}")

print("\n📊 RANDOM FOREST:")
rf_confusion = confusion_matrix(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc_roc = roc_auc_score(y_test, rf_pred_proba)
print(f"  ✅ Accuracy: {(rf_pred == y_test).sum() / len(y_test) * 100:.2f}%")
print(f"  ✅ F1-Score: {rf_f1:.4f}")
print(f"  ✅ AUC-ROC: {rf_auc_roc:.4f}")
print(f"  ✅ Confusion Matrix:")
print(f"     True Negatives:  {rf_confusion[0,0]}")
print(f"     False Positives: {rf_confusion[0,1]}")
print(f"     False Negatives: {rf_confusion[1,0]}")
print(f"     True Positives:  {rf_confusion[1,1]}")

# ---- STEP 8: ROC Curves ----
print("\n📈 Generating ROC curves...")
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)

plt.figure(figsize=(12, 4))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_auc_roc:.4f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_auc_roc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid()

# Confusion Matrix - Random Forest (better model)
plt.subplot(1, 2, 2)
sns.heatmap(rf_confusion, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('fraud_detection_evaluation.png', dpi=100)
print("✅ Evaluation plots saved as 'fraud_detection_evaluation.png'")
plt.show()

# ---- STEP 9: Feature Importance ----
print("\n🔍 Feature Importance (Random Forest):")
feature_importance = rf_model.feature_importances_
for i, importance in enumerate(feature_importance[:5]):
    print(f"  Feature {i}: {importance:.4f}")

# ---- STEP 10: Classification Report ----
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT - RANDOM FOREST")
print("="*70)
print(classification_report(y_test, rf_pred, 
                          target_names=['Legitimate', 'Fraudulent']))

# ---- STEP 11: Save Results ----
results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [
        (lr_pred == y_test).sum() / len(y_test) * 100,
        (rf_pred == y_test).sum() / len(y_test) * 100
    ],
    'F1-Score': [lr_f1, rf_f1],
    'AUC-ROC': [lr_auc_roc, rf_auc_roc],
    'True Positives': [lr_confusion[1,1], rf_confusion[1,1]],
    'False Negatives': [lr_confusion[1,0], rf_confusion[1,0]]
})

results_df.to_csv('fraud_detection_results.csv', index=False)
print("\n✅ Results saved as 'fraud_detection_results.csv'")

print("\n" + "="*70)
print("✅ TASK 4 COMPLETE!")
print("="*70)