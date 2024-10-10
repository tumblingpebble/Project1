# script_name: base_model_training.py

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
from imblearn.pipeline import Pipeline as ImbPipeline
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
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# Utility functions for system monitoring
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

# Define base models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# Load processed datasets
def load_data():
    logging.info("Loading datasets")
    X_train = pd.read_csv("../data/processed/X_train_ordinal.csv")
    X_test = pd.read_csv("../data/processed/X_test_ordinal.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

    # Drop unnecessary columns
    columns_to_drop = ["id", "full_name"]
    X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], inplace=True)
    X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], inplace=True)

    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Train and save models, predictions, and generate visualizations
def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
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

        # Evaluate model on test set
        logging.info(f"Evaluating {model_name}")
        y_test_pred = model.predict(X_test)
        report = classification_report(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)
        logging.info(f"Classification report for {model_name}:\n{report}")
        logging.info(f"Confusion matrix for {model_name}:\n{cm}")

        # Save predictions
        predictions_filename = f"../predictions/{model_name}_test_predictions.csv"
        pd.DataFrame(y_test_pred, columns=["Predictions"]).to_csv(predictions_filename, index=False)
        logging.info(f"Test predictions saved to {predictions_filename}")

        # Generate learning curve
        generate_learning_curve(model, X_train_res, y_train_res, model_name)

        # Generate confusion matrix heatmap
        generate_confusion_matrix_heatmap(cm, model_name)

        # Generate ROC curve
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test)[:, 1]
            generate_roc_curve(y_test, y_test_prob, model_name)

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
        scoring="accuracy"
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, test_mean, label="Validation score")
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
def generate_roc_curve(y_test, y_test_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"../visualizations/{model_name}_roc_curve.png")
    plt.close()

# Main function to load data, train, and evaluate models
def main():
    logging.info("Starting base model training script")
    X_train, X_test, y_train, y_test = load_data()

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test)

    logging.info("All models have been trained and evaluated")

if __name__ == "__main__":
    main()
    logging.info("Script ended successfully")
