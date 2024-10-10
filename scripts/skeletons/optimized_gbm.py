import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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

# Optimized GBM with hyperparameters to be tested
gbm_model = GradientBoostingClassifier(
    n_estimators=350,  # Adjust as needed for other combinations
    learning_rate=0.08,  # Adjust as needed for other combinations
    max_depth=4,  # Adjust as needed for other combinations
    min_samples_split=3,  # Adjust as needed for other combinations
    min_samples_leaf=1,  # Adjust as needed for other combinations
    subsample=0.6,  # Adjust as needed for other combinations
    random_state=42
)
gbm_model.fit(X_train, y_train.values.ravel())

# Evaluate on Training Set
y_train_pred = gbm_model.predict(X_train)
print("Classification Report for Optimized GBM (Training Set):")
print(classification_report(y_train, y_train_pred))

# Evaluate on Validation Set
y_val_pred = gbm_model.predict(X_val)
print("Classification Report for Optimized GBM (Validation Set):")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# Save the trained model
joblib.dump(gbm_model, '../models/optimized_gbm_model.pkl')

# Evaluate on Test Set
y_test_pred = gbm_model.predict(X_test)
print("Classification Report for Optimized GBM (Test Set):")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    gbm_model,
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
plt.title('Learning Curve (Optimized GBM)')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.savefig('../visualizations/optimized_learning_curve_gbm.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (Optimized GBM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/optimized_confusion_matrix_gbm.png')

# ROC Curve
y_test_prob = gbm_model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Optimized GBM)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('../visualizations/optimized_roc_curve_gbm.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='Optimized GBM')
plt.title('Precision-Recall Curve (Optimized GBM)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/optimized_precision_recall_gbm.png')
