import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Check LightGBM version
print(f"LightGBM version: {lgb.__version__}")

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


# Define LightGBM classifier with tuned hyperparameters
gbm_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='binary_logloss',
    n_estimators=200,
    max_depth=7,
    min_data_in_leaf=50,
    min_split_gain=0.1,
    learning_rate=0.05,
    num_leaves=31
)

# Using eval_set for validation and enabling early stopping using callbacks
callbacks = [lgb.early_stopping(stopping_rounds=10)]
gbm_model.fit(
    X_train,
    y_train.values.ravel(),
    eval_set=[(X_val, y_val.values.ravel())],
    eval_metric='logloss',
    callbacks=callbacks
)

# Save the model
joblib.dump(gbm_model, '../models/lightgbm_tuned_model.pkl')

# Adjust prediction threshold based on validation data
y_val_prob = gbm_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
f1_scores = 2 * precision * recall / (precision + recall)
optimal_idx = np.argmax(f1_scores)
best_threshold = thresholds[optimal_idx]
print(f"Best threshold: {best_threshold}")

# Evaluate on Test Set using best threshold
y_test_prob = gbm_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)
print("Classification Report for LightGBM (Test Set):")
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
plt.title('Learning Curve (LightGBM)')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.savefig('../visualizations/learning_curve_lightgbm.png')
plt.close()

# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix Heatmap (LightGBM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('../visualizations/confusion_matrix_lightgbm.png')
plt.close()

# ROC Curve
y_test_prob = gbm_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve (LightGBM)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('../visualizations/roc_curve_lightgbm.png')
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
plt.figure()
plt.plot(recall, precision, marker='.', label='LightGBM')
plt.title('Precision-Recall Curve (LightGBM)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.legend()
plt.savefig('../visualizations/precision_recall_lightgbm.png')
plt.close()

# Plot feature importance instead of tree visualization to avoid Graphviz dependency
def save_feature_importance(model, output_dir='../visualizations/feature_importance'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=10, importance_type='split')
    plt.title('Feature Importance (Top 10)')
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

# Save the feature importance visualization
save_feature_importance(gbm_model)