# random_forest.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print(f"SHAP version: {shap.__version__}")

# Ensure directories for saving models, predictions, and visualizations
os.makedirs('../models', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../predictions', exist_ok=True)

# Load processed datasets (Ordinal encoded data for Random Forest)
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


# Train Random Forest with limited depth
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=27,  # Adjust as necessary
    random_state=42
)
rf_model.fit(X_train, y_train.values.ravel())

# Evaluate on Training Set
y_train_pred = rf_model.predict(X_train)
print("Classification Report for Random Forest (Training Set):")
print(classification_report(y_train, y_train_pred))

# Evaluate on Validation Set
y_val_pred = rf_model.predict(X_val)
print("Classification Report for Random Forest (Validation Set):")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# Save the trained model
joblib.dump(rf_model, '../models/rf_model.pkl')

# Evaluate on Test Set
y_test_pred = rf_model.predict(X_test)
print("Classification Report for Random Forest (Test Set):")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Save predictions and probabilities for later use in visualizations
pd.DataFrame(y_test_pred, columns=['Predictions']).to_csv('../predictions/random_forest_test_predictions.csv', index=False)
y_test_prob = rf_model.predict_proba(X_test)
pd.DataFrame(y_test_prob, columns=['Prob_Class_0', 'Prob_Class_1']).to_csv('../predictions/random_forest_test_probabilities.csv', index=False)

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Generate SHAP values
shap_values = explainer.shap_values(X_test)

# Since shap_values is a 3D array, extract the SHAP values for class 1
shap_values_class1 = shap_values[:, :, 1]  # Shape: (1500, 7)

# Now check the shapes
print(f"Shape of shap_values_class1: {shap_values_class1.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Ensure the shapes match before plotting
if shap_values_class1.shape == X_test.shape:
    # SHAP summary plot
    shap.summary_plot(shap_values_class1, X_test, feature_names=X_test.columns)
    plt.savefig('../visualizations/shap_summary_rf.png', bbox_inches='tight')
    plt.close()
else:
    print("Mismatch between SHAP values and feature matrix shape.")

# Learning Curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf_model,
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
plt.title('Learning Curve (Random Forest)')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.savefig('../visualizations/learning_curve_rf.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/confusion_matrix_rf.png')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (Random Forest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('../visualizations/roc_curve_rf.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.title('Precision-Recall Curve (Random Forest)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/precision_recall_rf.png')