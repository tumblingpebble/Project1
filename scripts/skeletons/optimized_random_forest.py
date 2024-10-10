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
    cross_val_score,
    learning_curve,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from scipy.stats import randint, uniform

# New imports for logging and system monitoring
import logging
import psutil
import time
import traceback

# Imports for other models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE  # For handling class imbalance
from imblearn.combine import SMOTEENN  # SMOTEENN is in the combine module
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

    # Use ordinal encoded data
    X_train_ord = pd.read_csv("../data/processed/X_train_ordinal.csv")
    X_val_ord = pd.read_csv("../data/processed/X_val_ordinal.csv")
    X_test_ord = pd.read_csv("../data/processed/X_test_ordinal.csv")

    y_train = pd.read_csv("../data/processed/y_train.csv")
    y_val = pd.read_csv("../data/processed/y_val.csv")
    y_test = pd.read_csv("../data/processed/y_test.csv")

    # -----------------------------
    # Combine Training and Validation Data
    # -----------------------------
    X_train_full = pd.concat([X_train_ord, X_val_ord], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)

    # -----------------------------
    # Define models
    # -----------------------------
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    # -----------------------------
    # Over-sampling techniques
    # -----------------------------
    over_samplers = {
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
    }

    # -----------------------------
    # Iterate over models and over-sampling techniques
    # -----------------------------
    for sampler_name, sampler in over_samplers.items():
        for model_name, model in models.items():
            logging.info(f"Running {model_name} using {sampler_name}")

            # -----------------------------
            # Prepare data
            # -----------------------------
            X_train = X_train_full.copy()
            X_test = X_test_ord.copy()
            y_train_copy = y_train_full.copy()

            # -----------------------------
            # Safely drop 'id' and 'full_name' if they exist
            # -----------------------------
            columns_to_drop = ["id", "full_name"]
            X_train.drop(
                columns=[col for col in columns_to_drop if col in X_train.columns],
                axis=1,
                inplace=True,
            )
            X_test.drop(
                columns=[col for col in columns_to_drop if col in X_test.columns],
                axis=1,
                inplace=True,
            )

            # -----------------------------
            # Define hyperparameter optimization using Optuna
            # -----------------------------
            
            def objective(trial):
                if model_name == "RandomForest":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'max_depth': trial.suggest_int('max_depth', 5, 100),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'class_weight': 'balanced',
                        'n_jobs': -1,
                    }
                    clf = RandomForestClassifier(random_state=42, **params)
                elif model_name == "XGBoost":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                        'tree_method': 'hist',  # Use CPU-based tree method
                        'n_jobs': 1,  # Limit to 1 job to prevent resource contention
                    }
                    clf = XGBClassifier(random_state=42, **params)
                elif model_name == "CatBoost":
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 1000),
                        'depth': trial.suggest_int('depth', 3, 16),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                        'task_type': 'GPU',
                        'devices': '0',
                    }
                    clf = CatBoostClassifier(random_state=42, verbose=0, **params)
                elif model_name == "GradientBoosting":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    }
                    clf = GradientBoostingClassifier(random_state=42, **params)
                else:
                    return 0  # Assign a default score if model is not recognized

                pipeline = ImbPipeline([
                    ('sampler', sampler),
                    ('classifier', clf),
                ])

                # Set n_jobs based on the model
                if model_name == "CatBoost" and params.get('task_type') == 'GPU':
                    cv_n_jobs = 1  # Prevent multiple GPU tasks running in parallel
                else:
                    cv_n_jobs = -1  # Use all available cores

                score = cross_val_score(
                    pipeline,
                    X_train, y_train_copy.values.ravel(),
                    cv=5,
                    scoring='accuracy',
                    n_jobs=cv_n_jobs
                ).mean()
                return score

            # -----------------------------
            # Start monitoring system resources
            # -----------------------------
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            logging.info(f"Initial CPU usage: {cpu_usage}%")
            logging.info(f"Initial Memory usage: {memory_usage}%")

            # -----------------------------
            # Perform Hyperparameter Optimization
            # -----------------------------
            logging.info(f"Starting Optuna optimization for {model_name} with {sampler_name}")
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, timeout=600)  # Adjust trials and timeout as needed

            logging.info(f"Best trial for {model_name} with {sampler_name}: {study.best_trial.params}")
            best_params = study.best_trial.params

            # -----------------------------
            # Define the optimized classifier with best parameters
            # -----------------------------
            if model_name == "RandomForest":
                best_clf = RandomForestClassifier(random_state=42, **best_params)
            elif model_name == "XGBoost":
                best_clf = XGBClassifier(random_state=42, **best_params)
            elif model_name == "CatBoost":
                best_clf = CatBoostClassifier(random_state=42, verbose=0, **best_params)
            elif model_name == "GradientBoosting":
                best_clf = GradientBoostingClassifier(random_state=42, **best_params)
            else:
                logging.error(f"Model {model_name} is not recognized.")
                continue  # Skip to next iteration if model is not recognized

            # -----------------------------
            # Create the pipeline with the optimized classifier
            # -----------------------------
            pipeline = ImbPipeline([
                ('sampler', sampler),
                ('classifier', best_clf),
            ])

            # -----------------------------
            # Fit the optimized model on the training data with over-sampling
            # -----------------------------
            start_time = time.time()
            pipeline.fit(X_train, y_train_copy.values.ravel())
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f"Model training completed in {elapsed_time:.2f} seconds")

            # Log system resource usage after fitting
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            logging.info(f"CPU usage after training: {cpu_usage}%")
            logging.info(f"Memory usage after training: {memory_usage}%")

            # -----------------------------
            # Evaluate on Test Set
            # -----------------------------
            logging.info("Evaluating model on the test set")
            y_test_pred = pipeline.predict(X_test)
            test_report = classification_report(y_test, y_test_pred)
            test_conf_matrix = confusion_matrix(y_test, y_test_pred)
            logging.info("Test Set Classification Report:\n" + test_report)
            logging.info("Test Set Confusion Matrix:\n" + str(test_conf_matrix))
            print(f"\nClassification Report for {model_name} (Test Set) using {sampler_name}:")
            print(test_report)
            print("Confusion Matrix (Test Set):")
            print(test_conf_matrix)

            # -----------------------------
            # Save the trained model
            # -----------------------------
            model_filename = f"../models/{model_name}_{sampler_name}_model.pkl"
            joblib.dump(pipeline, model_filename)
            logging.info(f"Model saved to {model_filename}")

            # -----------------------------
            # Save predictions and probabilities
            # -----------------------------
            predictions_filename = f"../predictions/{model_name}_{sampler_name}_test_predictions.csv"
            probabilities_filename = f"../predictions/{model_name}_{sampler_name}_test_probabilities.csv"
            pd.DataFrame(y_test_pred, columns=["Predictions"]).to_csv(
                predictions_filename, index=False
            )
            y_test_prob = pipeline.predict_proba(X_test)
            pd.DataFrame(y_test_prob).to_csv(probabilities_filename, index=False)
            logging.info(f"Test set predictions saved to {predictions_filename}")
            logging.info(f"Test set probabilities saved to {probabilities_filename}")

            # -----------------------------
            # Learning Curve
            # -----------------------------
            logging.info("Generating learning curve")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_sizes, train_scores, test_scores = learning_curve(
                pipeline,
                X_train,
                y_train_copy.values.ravel(),
                cv=cv,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="accuracy",
            )

            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            plt.figure()
            plt.plot(train_sizes, train_mean, "o-", label="Training score")
            plt.plot(train_sizes, test_mean, "o-", label="Cross-validation score")
            plt.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.1,
            )
            plt.fill_between(
                train_sizes,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.1,
            )
            plt.title(f"Learning Curve ({model_name} - {sampler_name})")
            plt.xlabel("Training examples")
            plt.ylabel("Accuracy")
            plt.grid()
            plt.legend(loc="best")
            learning_curve_filename = f"../visualizations/{model_name}_{sampler_name}_learning_curve.png"
            plt.savefig(learning_curve_filename)
            plt.close()
            logging.info(f"Learning curve saved to {learning_curve_filename}")

            # -----------------------------
            # Confusion Matrix Heatmap
            # -----------------------------
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            plt.figure()
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[0, 1],
                yticklabels=[0, 1],
            )
            plt.title(f"Confusion Matrix Heatmap ({model_name} - {sampler_name})")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            confusion_matrix_filename = (
                f"../visualizations/{model_name}_{sampler_name}_confusion_matrix.png"
            )
            plt.savefig(confusion_matrix_filename)
            plt.close()  # Ensure the plot is closed
            logging.info(f"Confusion matrix heatmap saved to {confusion_matrix_filename}")

            # -----------------------------
            # ROC Curve
            # -----------------------------
            y_test_prob = pipeline.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"ROC Curve ({model_name} - {sampler_name})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid()
            roc_curve_filename = f"../visualizations/{model_name}_{sampler_name}_roc_curve.png"
            plt.savefig(roc_curve_filename)
            plt.close()  # Ensure the plot is closed
            logging.info(f"ROC curve saved to {roc_curve_filename}")

            # -----------------------------
            # Precision-Recall Curve and Threshold Adjustment
            # -----------------------------
            precision, recall, thresholds = precision_recall_curve(
                y_test, y_test_prob[:, 1]
            )
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_threshold = thresholds[np.argmax(f1_scores)]
            logging.info(f"Best threshold based on F1 score: {best_threshold}")
            print(f"Best threshold based on F1 score: {best_threshold}")

            # Apply adjusted threshold
            y_test_pred_adjusted = (y_test_prob[:, 1] >= best_threshold).astype(int)
            adjusted_report = classification_report(y_test, y_test_pred_adjusted)
            logging.info("Classification Report with Adjusted Threshold:\n" + adjusted_report)
            print("\nClassification Report with Adjusted Threshold:")
            print(adjusted_report)

            # Plot Precision-Recall Curve
            plt.figure()
            plt.plot(recall, precision, marker=".", label=model_name)
            plt.title(f"Precision-Recall Curve ({model_name} - {sampler_name})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid()
            plt.legend()
            pr_curve_filename = (
                f"../visualizations/{model_name}_{sampler_name}_precision_recall_curve.png"
            )
            plt.savefig(pr_curve_filename)
            plt.close()  # Ensure the plot is closed
            logging.info(f"Precision-Recall curve saved to {pr_curve_filename}")

            # -----------------------------
            # Feature Importance
            # -----------------------------
            # Access the classifier from the pipeline
            classifier = pipeline.named_steps['classifier']
            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = X_train.columns

                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importances ({model_name} - {sampler_name})")
                plt.bar(range(len(importances)), importances[indices], align="center")
                plt.xticks(
                    range(len(importances)),
                    [feature_names[i] for i in indices],
                    rotation=90,
                )
                plt.tight_layout()
                fi_filename = f"../visualizations/{model_name}_{sampler_name}_feature_importances.png"
                plt.savefig(fi_filename)
                plt.close()
                logging.info(f"Feature importances plot saved to {fi_filename}")
            else:
                logging.info(f"Model {model_name} does not have feature_importances_ attribute.")

    # -----------------------------
    # Log the completion of the script
    # -----------------------------
    logging.info("optimized_random_forest.py script completed")

except Exception as e:
    logging.error("An error occurred: " + str(e))
    logging.error("Traceback: " + traceback.format_exc())
    print("An error occurred during execution. Please check the log file for details.")
