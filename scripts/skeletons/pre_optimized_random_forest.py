# optimized_random_forest.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.stats import randint

# New imports for logging and system monitoring
import logging
import psutil
import time

# Uncomment the following line if you wish to implement dimensionality reduction
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Set up logging
# -----------------------------
logging.basicConfig(
    filename='optimized_random_forest.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Log the start of the script
logging.info('Starting optimized_random_forest.py script')

# -----------------------------
# Ensure directories for saving models, predictions, and visualizations
# -----------------------------
os.makedirs('../models', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../predictions', exist_ok=True)

# -----------------------------
# Load processed datasets
# -----------------------------
logging.info('Loading datasets')

# For Random Forest, which is a tree-based model, we can use ordinal encoded data
X_train = pd.read_csv('../data/processed/X_train_ordinal.csv')
X_val = pd.read_csv('../data/processed/X_val_ordinal.csv')
X_test = pd.read_csv('../data/processed/X_test_ordinal.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')
y_val = pd.read_csv('../data/processed/y_val.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

# -----------------------------
# Safely drop 'id' and 'full_name' only if they exist in the dataframe
# -----------------------------
columns_to_drop = ['id', 'full_name']
X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], axis=1, inplace=True)
X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns], axis=1, inplace=True)
X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], axis=1, inplace=True)

# -----------------------------
# Optional: Apply Dimensionality Reduction
# -----------------------------
# If you wish to use dimensionality reduction (e.g., for high-dimensional data), uncomment the following code

logging.info('Applying TruncatedSVD for dimensionality reduction')

# Since the ordinal data is likely low-dimensional, TruncatedSVD may not be necessary
# If you have high-dimensional data (e.g., after one-hot encoding), you can apply TruncatedSVD

# Load one-hot encoded datasets instead of ordinal
X_train = pd.read_csv('../data/processed/X_train_onehot.csv')
X_val = pd.read_csv('../data/processed/X_val_onehot.csv')
X_test = pd.read_csv('../data/processed/X_test_onehot.csv')

# Drop columns before SVD
X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], axis=1, inplace=True)
X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns], axis=1, inplace=True)
X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], axis=1, inplace=True)

# Apply TruncatedSVD
n_components = 10  # Adjust the number of components as needed
svd = TruncatedSVD(n_components=n_components, random_state=42)

# Fit on X_train and transform
X_train = svd.fit_transform(X_train)
X_val = svd.transform(X_val)
X_test = svd.transform(X_test)

logging.info(f'TruncatedSVD applied with n_components={n_components}')


# Note: Ensure that you comment or uncomment the above section appropriately

# -----------------------------
# Set up RandomForestClassifier
# -----------------------------
rf_model = RandomForestClassifier(random_state=42)

# -----------------------------
# Define hyperparameter distributions
# -----------------------------
# Fix the 'max_features' parameter to avoid the 'auto' error
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2', None]  # Removed 'auto' to fix parameter error
}

# Optionally, you can set up HalvingRandomSearchCV
# Uncomment the following code to use HalvingRandomSearchCV

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

logging.info('Starting hyperparameter tuning using HalvingRandomSearchCV')

search = HalvingRandomSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    factor=2,
    resource='n_estimators',
    max_resources=1000,
    min_resources=100,
    random_state=42,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

"""
# Otherwise, proceed with RandomizedSearchCV
logging.info('Starting hyperparameter tuning using RandomizedSearchCV')

search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=100,  # Adjust n_iter based on your computational resources
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
"""
# -----------------------------
# Start monitoring system resources
# -----------------------------
cpu_usage = psutil.cpu_percent(interval=None)
memory_usage = psutil.virtual_memory().percent
logging.info(f'Initial CPU usage: {cpu_usage}%')
logging.info(f'Initial Memory usage: {memory_usage}%')

# -----------------------------
# Fit the search on the training data
# -----------------------------
start_time = time.time()
search.fit(X_train, y_train.values.ravel())
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f'Hyperparameter tuning completed in {elapsed_time:.2f} seconds')

# Log system resource usage after fitting
cpu_usage = psutil.cpu_percent(interval=None)
memory_usage = psutil.virtual_memory().percent
logging.info(f'CPU usage after tuning: {cpu_usage}%')
logging.info(f'Memory usage after tuning: {memory_usage}%')

# -----------------------------
# Extract the best parameters
# -----------------------------
best_params = search.best_params_
logging.info(f'Best parameters: {best_params}')
print(f"Best parameters from RandomizedSearchCV: {best_params}")

best_score = search.best_score_
logging.info(f'Best cross-validation score: {best_score:.4f}')
print(f"Best cross-validation score: {best_score:.4f}")

# Use the best model
optimized_rf_model = search.best_estimator_

# -----------------------------
# Evaluate on the Validation Set
# -----------------------------
logging.info('Evaluating model on the validation set')
y_val_pred = optimized_rf_model.predict(X_val)
val_report = classification_report(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
logging.info('Validation Set Classification Report:\n' + val_report)
logging.info('Validation Set Confusion Matrix:\n' + str(val_conf_matrix))
print("Classification Report for Optimized Random Forest (Validation Set):")
print(val_report)
print("Confusion Matrix (Validation Set):")
print(val_conf_matrix)

# -----------------------------
# Save the trained model
# -----------------------------
model_filename = '../models/optimized_rf_model.pkl'
joblib.dump(optimized_rf_model, model_filename)
logging.info(f'Model saved to {model_filename}')

# -----------------------------
# Evaluate on Test Set
# -----------------------------
logging.info('Evaluating model on the test set')
y_test_pred = optimized_rf_model.predict(X_test)
test_report = classification_report(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
logging.info('Test Set Classification Report:\n' + test_report)
logging.info('Test Set Confusion Matrix:\n' + str(test_conf_matrix))
print("Classification Report for Optimized Random Forest (Test Set):")
print(test_report)
print("Confusion Matrix (Test Set):")
print(test_conf_matrix)

# -----------------------------
# Save predictions and probabilities for later use in visualizations
# -----------------------------
predictions_filename = '../predictions/optimized_rf_test_predictions.csv'
probabilities_filename = '../predictions/optimized_rf_test_probabilities.csv'
pd.DataFrame(y_test_pred, columns=['Predictions']).to_csv(predictions_filename, index=False)
y_test_prob = optimized_rf_model.predict_proba(X_test)
pd.DataFrame(y_test_prob, columns=['Prob_Class_0', 'Prob_Class_1']).to_csv(probabilities_filename, index=False)
logging.info(f'Test set predictions saved to {predictions_filename}')
logging.info(f'Test set probabilities saved to {probabilities_filename}')

# -----------------------------
# Learning Curve
# -----------------------------
logging.info('Generating learning curve')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_sizes, train_scores, test_scores = learning_curve(
    optimized_rf_model, X_train, y_train.values.ravel(),
    cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
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
learning_curve_filename = '../visualizations/optimized_learning_curve_rf.png'
plt.savefig(learning_curve_filename)
plt.close()
logging.info(f'Learning curve saved to {learning_curve_filename}')

# -----------------------------
# Confusion Matrix Heatmap
# -----------------------------
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (Optimized Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
confusion_matrix_filename = '../visualizations/optimized_confusion_matrix_rf.png'
plt.savefig(confusion_matrix_filename)
logging.info(f'Confusion matrix heatmap saved to {confusion_matrix_filename}')

# -----------------------------
# ROC Curve
# -----------------------------
y_test_prob = optimized_rf_model.predict_proba(X_test)
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
roc_curve_filename = '../visualizations/optimized_roc_curve_rf.png'
plt.savefig(roc_curve_filename)
logging.info(f'ROC curve saved to {roc_curve_filename}')

# -----------------------------
# Precision-Recall Curve
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='Optimized Random Forest')
plt.title('Precision-Recall Curve (Optimized Random Forest)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
pr_curve_filename = '../visualizations/optimized_precision_recall_rf.png'
plt.savefig(pr_curve_filename)
logging.info(f'Precision-Recall curve saved to {pr_curve_filename}')

# -----------------------------
# Log the completion of the script
# -----------------------------
logging.info('optimized_random_forest.py script completed')
