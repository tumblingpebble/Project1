# base_model_training.py

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

from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    filename="base_model_training.log",
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

# Define base models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
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

    # Drop unnecessary columns
    columns_to_drop = ["id"]
    X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], inplace=True)
    X_val.drop(columns=[col for col in columns_to_drop if col in X_val.columns], inplace=True)
    X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], inplace=True)

    logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train and save models, predictions, and generate visualizations
def train_and_evaluate(model_name, model, X_train, X_val, X_test, y_train, y_val, y_test):
    try:
        logging.info(f"Starting training for {model_name}")
        start_time = time.time()

        # Apply SMOTE for handling class imbalance
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"Data after SMOTE: {X_train_res.shape}")

        # Train model
        model.fit(X_train_res, y_train_res)
        end_time = time.time()
        logging.info(f"Training completed for {model_name} in {end_time - start_time:.2f} seconds")

        # Save model
        model_filename = f"../models/{model_name}_model.pkl"
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
        predictions_filename = f"../predictions/{model_name}_test_predictions.csv"
        pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred}).to_csv(predictions_filename, index=False)
        logging.info(f"Test predictions saved to {predictions_filename}")

        # Generate visualizations
        generate_learning_curve(model, X_train_res, y_train_res, model_name)
        generate_confusion_matrix_heatmap(test_cm, model_name)
        generate_roc_curve(y_test, model, X_test, model_name)

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
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        n_jobs=-1
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

# Main function to load data, train, and evaluate models
def main():
    logging.info("Starting base model training script")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Combine training and validation sets for training (optional)
    # X_train_combined = pd.concat([X_train, X_val], axis=0)
    # y_train_combined = pd.concat([y_train, y_val], axis=0)

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate(model_name, model, X_train, X_val, X_test, y_train, y_val, y_test)

    logging.info("All models have been trained and evaluated")

if __name__ == "__main__":
    main()
    logging.info("Script ended successfully")
