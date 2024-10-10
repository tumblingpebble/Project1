# optimized_xgboost_model.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure directories for saving models, predictions, and visualizations
os.makedirs('../models', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../predictions', exist_ok=True)

# Load processed datasets
X_train = pd.read_csv('../data/processed/X_train_ordinal.csv')
X_val = pd.read_csv('../data/processed/X_val_ordinal.csv')
X_test = pd.read_csv('../data/processed/X_test_ordinal.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')
y_val = pd.read_csv('../data/processed/y_val.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

# Safely drop 'id' and 'full_name' only if they exist in the dataframe
columns_to_drop = ['id', 'full_name']
X_train = X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], axis=1)
X_val = X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns], axis=1)
X_test = X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], axis=1)

# Train XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=800, 
    max_depth=9,
    learning_rate=0.08,
    subsample=0.7,
    colsample_bytree=0.85,
    scale_pos_weight=2, #handle class imbalance, adjust based on data 
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train.values.ravel())

# Evaluate on Training Set
y_train_pred = xgb_model.predict(X_train)
print("Classification Report for XGBoost (Training Set):")
print(classification_report(y_train, y_train_pred))

# Evaluate on Validation Set
y_val_pred = xgb_model.predict(X_val)
print("Classification Report for XGBoost (Validation Set):")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# Save the trained model
joblib.dump(xgb_model, '../models/optimized_xgb_model.pkl')

# Evaluate on Test Set
y_test_pred = xgb_model.predict(X_test)
print("Classification Report for XGBoost (Test Set):")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    xgb_model,
    X_train,
    y_train.values.ravel(),
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='accuracy',
    shuffle=True,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, 'o-', label="Training score")
plt.plot(train_sizes, test_mean, 'o-', label="Cross-validation score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.title('Learning Curve (XGBoost)')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.savefig('../visualizations/optimized_learning_curve_xgb.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (XGBoost)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/optimized_confusion_matrix_xgb.png')

# ROC Curve
y_test_prob = xgb_model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (XGBoost)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('../visualizations/optimized_roc_curve_xgb.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='XGBoost')
plt.title('Precision-Recall Curve (XGBoost)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/optimized_precision_recall_xgb.png')
