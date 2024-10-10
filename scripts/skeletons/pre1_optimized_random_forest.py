# optimized_random_forest.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    RandomizedSearchCV,
    HalvingRandomSearchCV,
    cross_val_score,  # Added this line
    learning_curve,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline  # For handling pipelines with SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from scipy.stats import randint, uniform
from sklearn.decomposition import TruncatedSVD

# New imports for logging and system monitoring
import logging
import psutil
import time
import traceback

# Imports for other models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE  # For handling class imbalance
import optuna  # For Bayesian optimization

# -----------------------------
# Set up logging
# -----------------------------
logging.basicConfig(
    filename="optimized_random_forest.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# -----------------------------
# Ensure directories for saving models, predictions, and visualizations
# -----------------------------
os.makedirs("../models", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../predictions", exist_ok=True)

# -----------------------------
# Main script execution
# -----------------------------
try:
    # Log the start of the script
    logging.info("Starting optimized_random_forest.py script")

    # -----------------------------
    # Load processed datasets
    # -----------------------------
    logging.info("Loading datasets")

    # For tree-based models, we can use ordinal encoded data
    X_train_ord = pd.read_csv("../data/processed/X_train_ordinal.csv")
    X_val_ord = pd.read_csv("../data/processed/X_val_ordinal.csv")
    X_test_ord = pd.read_csv("../data/processed/X_test_ordinal.csv")

    X_train_onehot = pd.read_csv("../data/processed/X_train_onehot.csv")
    X_val_onehot = pd.read_csv("../data/processed/X_val_onehot.csv")
    X_test_onehot = pd.read_csv("../data/processed/X_test_onehot.csv")

    y_train = pd.read_csv("../data/processed/y_train.csv")
    y_val = pd.read_csv("../data/processed/y_val.csv")
    y_test = pd.read_csv("../data/processed/y_test.csv")

    # -----------------------------
    # Define models and configurations
    # -----------------------------
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    configs = {
        "WithSVD": {
            "apply_svd": True,
            "data": "onehot",
            "n_components": 10,  # Adjust as needed
        },
        "WithoutSVD": {
            "apply_svd": False,
            "data": "ordinal",
        },
    }

    # -----------------------------
    # Iterate over models and configurations
    # -----------------------------
    for config_name, config in configs.items():
        for model_name, model in models.items():
            logging.info(f"Running {model_name} with {config_name}")

            # -----------------------------
            # Select data based on configuration
            # -----------------------------
            if config["data"] == "ordinal":
                X_train = X_train_ord.copy()
                X_val = X_val_ord.copy()
                X_test = X_test_ord.copy()
            else:
                X_train = X_train_onehot.copy()
                X_val = X_val_onehot.copy()
                X_test = X_test_onehot.copy()

            y_train_copy = y_train.copy()

            # -----------------------------
            # Safely drop 'id' and 'full_name' if they exist
            # -----------------------------
            columns_to_drop = ["id", "full_name"]
            X_train.drop(
                columns=[col for col in columns_to_drop if col in X_train.columns],
                axis=1,
                inplace=True,
            )
            X_val.drop(
                columns=[col for col in columns_to_drop if col in X_val.columns],
                axis=1,
                inplace=True,
            )
            X_test.drop(
                columns=[col for col in columns_to_drop if col in X_test.columns],
                axis=1,
                inplace=True,
            )

            # -----------------------------
            # Apply Dimensionality Reduction if specified
            # -----------------------------
            if config.get("apply_svd", False):
                logging.info("Applying TruncatedSVD for dimensionality reduction")
                n_components = config["n_components"]
                svd = TruncatedSVD(n_components=n_components, random_state=42)

                # Fit and transform
                X_train = svd.fit_transform(X_train)
                X_val = svd.transform(X_val)
                X_test = svd.transform(X_test)

                logging.info(f"TruncatedSVD applied with n_components={n_components}")

            # -----------------------------
            # Define hyperparameter optimization using Optuna
            # -----------------------------
            if model_name == "RandomForest":
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 5, 50),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'class_weight': 'balanced',
                    }
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=42)),
                        ('classifier', RandomForestClassifier(random_state=42, **params)),
                    ])
                    score = cross_val_score(
                        pipeline,
                        X_train, y_train.values.ravel(),
                        cv=3,
                        scoring='accuracy',
                        n_jobs=-1
                    ).mean()
                    return score

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=20)
                best_params = study.best_params
                logging.info(f"Best parameters from Optuna: {best_params}")
                optimized_model = RandomForestClassifier(random_state=42, **best_params)

            # Repeat similar adjustments for XGBoost, CatBoost, GradientBoosting

            # -----------------------------
            # Start monitoring system resources
            # -----------------------------
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            logging.info(f"Initial CPU usage: {cpu_usage}%")
            logging.info(f"Initial Memory usage: {memory_usage}%")

            # -----------------------------
            # Fit the optimized model on the training data with SMOTE
            # -----------------------------
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', optimized_model),
            ])
            start_time = time.time()
            pipeline.fit(X_train, y_train.values.ravel())
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Model training completed in {elapsed_time:.2f} seconds")

            # Log system resource usage after fitting
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            logging.info(f"CPU usage after training: {cpu_usage}%")
            logging.info(f"Memory usage after training: {memory_usage}%")

            # -----------------------------
            # Evaluate on the Validation Set
            # -----------------------------
            logging.info("Evaluating model on the validation set")
            y_val_pred = pipeline.predict(X_val)
            val_report = classification_report(y_val, y_val_pred)
            val_conf_matrix = confusion_matrix(y_val, y_val_pred)
            logging.info("Validation Set Classification Report:\n" + val_report)
            logging.info("Validation Set Confusion Matrix:\n" + str(val_conf_matrix))
            print(f"\nClassification Report for {model_name} (Validation Set):")
            print(val_report)
            print("Confusion Matrix (Validation Set):")
            print(val_conf_matrix)

            # -----------------------------
            # Save the trained model
            # -----------------------------
            model_filename = f"../models/{model_name}_{config_name}_model.pkl"
            joblib.dump(pipeline, model_filename)
            logging.info(f"Model saved to {model_filename}")

            # -----------------------------
            # Evaluate on Test Set
            # -----------------------------
            logging.info("Evaluating model on the test set")
            y_test_pred = pipeline.predict(X_test)
            test_report = classification_report(y_test, y_test_pred)
            test_conf_matrix = confusion_matrix(y_test, y_test_pred)
            logging.info("Test Set Classification Report:\n" + test_report)
            logging.info("Test Set Confusion Matrix:\n" + str(test_conf_matrix))
            print(f"\nClassification Report for {model_name} (Test Set):")
            print(test_report)
            print("Confusion Matrix (Test Set):")
            print(test_conf_matrix)

            # Rest of your code...

    # -----------------------------
    # Log the completion of the script
    # -----------------------------
    logging.info("optimized_random_forest.py script completed")

except Exception as e:
    logging.error("An error occurred: " + str(e))
    logging.error("Traceback: " + traceback.format_exc())
    print("An error occurred during execution. Please check the log file for details.")
