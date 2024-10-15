import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../predictions', exist_ok=True)

# Load the imputed dataset
data = pd.read_csv("KNN_ad_click_dataset.csv")

# Drop irrelevant columns
data.drop("id", inplace=True, axis=1)
data.drop("full_name", inplace=True, axis=1)

# Prepare features (X) and target (y)
y = data["click"]
X = data.drop("click", axis=1)

# Encode categorical features
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1275, random_state=0)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model on training set
y_train_pred = rf_model.predict(X_train)
print("Random Forest - Training Set")
print(classification_report(y_train, y_train_pred))

# Evaluate model on validation set
y_val_pred = rf_model.predict(X_val)
print("\nRandom Forest - Validation Set")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_val, y_val_pred))

# Evaluate model on test set
y_test_pred = rf_model.predict(X_test)
print("\nRandom Forest - Test Set")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Save the trained model
import joblib
joblib.dump(rf_model, '../models/rf_model_ian.pkl')

# Save predictions and probabilities for later use in visualizations
pd.DataFrame(y_test_pred, columns=['Predictions']).to_csv('../predictions/random_forest_test_predictions_ian.csv', index=False)
y_test_prob = rf_model.predict_proba(X_test)
pd.DataFrame(y_test_prob, columns=['Prob_Class_0', 'Prob_Class_1']).to_csv('../predictions/random_forest_test_probabilities_groupmate.csv', index=False)

# Initialize the SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf_model)

# Generate SHAP values
shap_values = explainer.shap_values(X_test)

# If shap_values is a list, extract the correct index for class 1
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]  # For class 1
else:
    shap_values_class1 = shap_values

# Ensure that the shapes match before proceeding to the SHAP summary plot
print(f"Shape of shap_values_class1: {shap_values_class1.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Check if the shapes match
if shap_values_class1.shape == X_test.shape:
    # SHAP summary plot
    shap.summary_plot(shap_values_class1, X_test, feature_names=X_test.columns)
    plt.savefig('../visualizations/shap_summary_rf_groupmate.png', bbox_inches='tight')
    plt.close()
else:
    print("Mismatch between SHAP values and feature matrix shape.")

# Learning Curve
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
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
plt.savefig('../visualizations/learning_curve_rf_ian.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/confusion_matrix_rf_ian.png')

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
plt.savefig('../visualizations/roc_curve_rf_ian.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob[:, 1])
plt.figure()
plt.plot(recall, precision, marker='.', label='Random Forest')
plt.title('Precision-Recall Curve (Random Forest)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/precision_recall_rf_ian.png')

print("All visualizations and model outputs saved.")
