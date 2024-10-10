# logistic_regression.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

# Load the processed datasets
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

# Logistic Regression with class weights and regularization
logreg_model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.01, random_state=42)
logreg_model.fit(X_train, y_train.values.ravel())

# Evaluate on Training Set
y_train_pred = logreg_model.predict(X_train)
print("Classification Report for Logistic Regression (Training Set):")
print(classification_report(y_train, y_train_pred))

# Evaluate on Validation Set
y_val_pred = logreg_model.predict(X_val)
print("Classification Report for Logistic Regression (Validation Set):")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# Save the trained model
joblib.dump(logreg_model, '../models/logreg_model.pkl')

# Evaluate on Test Set
y_test_pred = logreg_model.predict(X_test)
print("Classification Report for Logistic Regression (Test Set):")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Save predictions and probabilities for later use in visualizations
pd.DataFrame(y_test_pred, columns=['Predictions']).to_csv('../predictions/logreg_test_predictions.csv', index=False)
y_test_prob = logreg_model.predict_proba(X_test)
pd.DataFrame(y_test_prob, columns=['Prob_Class_0', 'Prob_Class_1']).to_csv('../predictions/logreg_test_probabilities.csv', index=False)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    logreg_model,
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
plt.title('Learning Curve (Logistic Regression)')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.savefig('../visualizations/learning_curve_logreg.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/confusion_matrix_logreg.png')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Logistic Regression)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('../visualizations/roc_curve_logreg.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='Logistic Regression')
plt.title('Precision-Recall Curve (Logistic Regression)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/precision_recall_logreg.png')
