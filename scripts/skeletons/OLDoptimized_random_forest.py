# optimized_random_forest.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV, learning_curve
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

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [500, 700],  # Lower the number to prevent overfitting
    'max_depth': [25, 27],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
grid_search.fit(X_train, y_train.values.ravel())

# Extract the best parameters
best_params = grid_search.best_params_
print(f"Best parameters from GridSearchCV: {best_params}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use the best model
optimized_rf_model = grid_search.best_estimator_

# Evaluate on the Validation Set
y_val_pred = optimized_rf_model.predict(X_val)
print("Classification Report for Optimized Random Forest (Validation Set):")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# Save the trained model
joblib.dump(optimized_rf_model, '../models/optimized_rf_model.pkl')

# Evaluate on Test Set
y_test_pred = optimized_rf_model.predict(X_test)
print("Classification Report for Optimized Random Forest (Test Set):")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Save predictions and probabilities for later use in visualizations
pd.DataFrame(y_test_pred, columns=['Predictions']).to_csv('../predictions/optimized_rf_test_predictions.csv', index=False)
y_test_prob = optimized_rf_model.predict_proba(X_test)
pd.DataFrame(y_test_prob, columns=['Prob_Class_0', 'Prob_Class_1']).to_csv('../predictions/optimized_rf_test_probabilities.csv', index=False)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    optimized_rf_model, X_train, y_train.values.ravel(),
    cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, 'o-', label="Training score")
plt.plot(train_sizes, test_mean, 'o-', label="Cross-validation score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.title('Learning Curve (Optimized Random Forest)')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.savefig('../visualizations/optimized_learning_curve_rf.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (Optimized Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/optimized_confusion_matrix_rf.png')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Optimized Random Forest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('../visualizations/optimized_curve_rf.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='Optimized Random Forest')
plt.title('Precision-Recall Curve (Optimized Random Forest)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/optimized_precision_recall_rf.png')
