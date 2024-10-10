# optimized_model_training.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import warnings
import logging
import psutil
import time
import traceback

from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import optuna  # For hyperparameter optimization

# Setup logging
logging.basicConfig(
    filename="optimized_model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Ensure directories for saving models, predictions, and visualizations
os.makedirs("../models", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../predictions", exist_ok=True)

# Define model constructors for initialization after hyperparameter optimization
model_constructors = {
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "CatBoost": CatBoostClassifier,
    "GradientBoosting": GradientBoostingClassifier
}

# Load processed datasets
def load_data():
    logging.info("Loading datasets")
    X_train = pd.read_csv("../data/processed/X_train.csv")
    X_val = pd.read_csv("../data/processed/X_val.csv")
    X_test = pd.read_csv("../data/processed/X_test.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
    y_val = pd.read_csv("../data/processed/y_val.csv").squeeze()
    y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

    logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Hyperparameter optimization function
def hyperparameter_optimization(model_name, X_train, y_train):
    logging.info(f"Starting hyperparameter optimization for {model_name}")

    def objective(trial):
        if model_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'n_jobs': -1,
            }
            clf = RandomForestClassifier(random_state=42, **params)
        elif model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'eval_metric': 'logloss',
                'n_jobs': -1,
            }
            clf = XGBClassifier(random_state=42, **params)
        elif model_name == "CatBoost":
            physical_cores = psutil.cpu_count(logical=False)
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'verbose': 0,
                'thread_count': physical_cores,
            }
            clf = CatBoostClassifier(random_state=42, **params)
        elif model_name == "GradientBoosting":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            clf = GradientBoostingClassifier(random_state=42, **params)
        else:
            logging.error(f"Model {model_name} not recognized.")
            return 0.0

        # Apply SMOTE within the objective to prevent data leakage
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=1800)
    logging.info(f"Best parameters for {model_name}: {study.best_params}")
    logging.info(f"Best cross-validation score for {model_name}: {study.best_value}")

    return study.best_params

# Train and save models, predictions, and generate visualizations
def train_and_evaluate(model_name, model_constructor, X_train, X_val, X_test, y_train, y_val, y_test):
    try:
        logging.info(f"Starting training for {model_name}")

        # Hyperparameter optimization
        best_params = hyperparameter_optimization(model_name, X_train, y_train)

        # Adjust thread_count for CatBoost
        if model_name == "CatBoost":
            physical_cores = psutil.cpu_count(logical=False)
            best_params['thread_count'] = physical_cores

        # Apply SMOTE for handling class imbalance
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"Data after SMOTE: {X_train_res.shape}")

        # Initialize model with best parameters
        model = model_constructor(random_state=42, **best_params)

        start_time = time.time()

        # Train model
        model.fit(X_train_res, y_train_res)
        end_time = time.time()
        logging.info(f"Training completed for {model_name} in {end_time - start_time:.2f} seconds")

        # Save model
        model_filename = f"../models/{model_name}_optimized_model.pkl"
        joblib.dump(model, model_filename)
        logging.info(f"Model saved to {model_filename}")

        # Evaluate model on validation and test sets
        logging.info(f"Evaluating {model_name} on validation set")
        y_val_pred = model.predict(X_val)
        val_report = classification_report(y_val, y_val_pred)
        val_cm = confusion_matrix(y_val, y_val_pred)
        logging.info(f"Validation classification report for {model_name}:\n{val_report}")
        logging.info(f"Validation confusion matrix for {model_name}:\n{val_cm}")

        logging.info(f"Evaluating {model_name} on test set")
        y_test_pred = model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        logging.info(f"Test classification report for {model_name}:\n{test_report}")
        logging.info(f"Test confusion matrix for {model_name}:\n{test_cm}")

        # Save test predictions
        predictions_filename = f"../predictions/{model_name}_optimized_test_predictions.csv"
        pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred}).to_csv(predictions_filename, index=False)
        logging.info(f"Test predictions saved to {predictions_filename}")

        # Generate visualizations with unique filenames to avoid overwriting
        model_suffix = f"{model_name}_optimized"

        generate_learning_curve(model, X_train_res, y_train_res, model_suffix)
        generate_confusion_matrix_heatmap(test_cm, model_suffix)
        generate_roc_curve(y_test, model, X_test, model_suffix)

        # Generate precision-recall curve and adjust threshold
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test)[:, 1]
            generate_precision_recall_curve(y_test, y_test_prob, model_suffix)

        # Generate feature importances if available
        if hasattr(model, "feature_importances_"):
            generate_feature_importances(model, X_train.columns, model_suffix)

    except Exception as e:
        logging.error(f"An error occurred while training {model_name}: {str(e)}")
        logging.error(traceback.format_exc())

# Generate learning curve
def generate_learning_curve(model, X_train, y_train, model_name):
    logging.info(f"Generating learning curve for {model_name}")
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=3,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy"
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, test_mean, label="Cross-validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve for {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"../visualizations/{model_name}_learning_curve.png")
    plt.close()

# Generate confusion matrix heatmap
def generate_confusion_matrix_heatmap(cm, model_name):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f"../visualizations/{model_name}_confusion_matrix.png")
    plt.close()

# Generate ROC curve
def generate_roc_curve(y_test, model, X_test, model_name):
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_test_prob = model.decision_function(X_test)
    else:
        logging.warning(f"{model_name} does not have predict_proba or decision_function method.")
        return
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"../visualizations/{model_name}_roc_curve.png")
    plt.close()

# Generate precision-recall curve and adjust threshold
def generate_precision_recall_curve(y_test, y_test_prob, model_name):
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    if len(f1_scores) > 0:
        best_threshold = thresholds[np.argmax(f1_scores)]
    else:
        best_threshold = 0.5  # Default threshold

    logging.info(f"Best threshold based on F1 score: {best_threshold}")

    # Apply adjusted threshold
    y_test_pred_adjusted = (y_test_prob >= best_threshold).astype(int)
    adjusted_report = classification_report(y_test, y_test_pred_adjusted)
    logging.info(f"Classification report with adjusted threshold for {model_name}:\n{adjusted_report}")

    # Plot Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.title(f"Precision-Recall Curve for {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.legend()
    pr_curve_filename = f"../visualizations/{model_name}_precision_recall_curve.png"
    plt.savefig(pr_curve_filename)
    plt.close()
    logging.info(f"Precision-Recall curve saved to {pr_curve_filename}")

# Generate feature importances
def generate_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importances for {model_name}")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"../visualizations/{model_name}_feature_importances.png")
    plt.close()
    logging.info(f"Feature importances saved for {model_name}")

# Main function to load data, train, and evaluate models
def main():
    logging.info("Starting optimized model training script")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Combine training and validation sets if desired
    # X_train_combined = pd.concat([X_train, X_val], axis=0)
    # y_train_combined = pd.concat([y_train, y_val], axis=0)

    # Train and evaluate each model
    for model_name, model_constructor in model_constructors.items():
        train_and_evaluate(model_name, model_constructor, X_train, X_val, X_test, y_train, y_val, y_test)

    logging.info("All models have been trained and evaluated")

if __name__ == "__main__":
    main()
    logging.info("Script ended successfully")
