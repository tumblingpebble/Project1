# one_trial_script.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score,
    learning_curve,  # Ensure this is imported
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
from scipy.stats import randint, uniform

# New imports for logging and system monitoring
import logging
import psutil
import time
import traceback

# Imports for models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE  # For handling class imbalance
from imblearn.combine import SMOTEENN  # SMOTEENN is in the combine module

# -----------------------------
# Set up logging
# -----------------------------
logging.basicConfig(
    filename="one_trial.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# -----------------------------
# Ensure directories for saving models, predictions, and visualizations
# -----------------------------
os.makedirs("../models", exist_ok=True)
os.makedirs("../visualizations", exist_ok=True)
os.makedirs("../predictions", exist_ok=True)

# -----------------------------
# Utility functions for system monitoring
# -----------------------------
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    return psutil.virtual_memory().percent

def get_gpu_memory_available():
    """Function to check available GPU memory using NVIDIA-SMI."""
    try:
        import subprocess
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        free_memory = int(result.strip().split('\n')[0])  # Assuming single GPU
        return free_memory  # in MiB
    except Exception as e:
        logging.warning("Could not retrieve GPU memory info: " + str(e))
        return None

# -----------------------------
# Main execution
# -----------------------------
def main():
    try:
        logging.info("Starting one_trial_script.py")

        # -----------------------------
        # Load processed datasets
        # -----------------------------
        logging.info("Loading datasets")

        # Use ordinal encoded data
        X_train_ord = pd.read_csv("../data/processed/X_train_ordinal.csv")
        X_val_ord = pd.read_csv("../data/processed/X_val_ordinal.csv")
        X_test_ord = pd.read_csv("../data/processed/X_test_ordinal.csv")

        y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
        y_val = pd.read_csv("../data/processed/y_val.csv").squeeze()
        y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

        # -----------------------------
        # Combine Training and Validation Data
        # -----------------------------
        X_train_full = pd.concat([X_train_ord, X_val_ord], ignore_index=True)
        y_train_full = pd.concat([y_train, y_val], ignore_index=True)

        # -----------------------------
        # Define model and over-sampler
        # -----------------------------
        model_name = "CatBoost"  # Change as needed: "RandomForest", "XGBoost", "CatBoost", "GradientBoosting"
        sampler_name = "SMOTE"    # Change as needed: "SMOTE", "ADASYN", "BorderlineSMOTE", "SMOTEENN"

        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }

        over_samplers = {
            "SMOTE": SMOTE(random_state=42),
            "ADASYN": ADASYN(random_state=42),
            "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
            "SMOTEENN": SMOTEENN(random_state=42),
        }

        if sampler_name not in over_samplers:
            logging.error(f"Over-sampler {sampler_name} is not recognized.")
            return

        if model_name not in models:
            logging.error(f"Model {model_name} is not recognized.")
            return

        sampler = over_samplers[sampler_name]
        model = models[model_name]

        # -----------------------------
        # Prepare data
        # -----------------------------
        X_train = X_train_full.copy()
        X_test = X_test_ord.copy()
        y_train_copy = y_train_full.copy()

        # Safely drop 'id' and 'full_name' if they exist
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
        # Define hyperparameters for the single trial
        # -----------------------------
        hyperparams = {
            'iterations': 300,        # For CatBoost
            'depth': 10,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            # Add other hyperparameters as needed
        }

        # Adjust hyperparameters based on the model
        if model_name == "RandomForest":
            clf = RandomForestClassifier(random_state=42, **hyperparams)
        elif model_name == "XGBoost":
            clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **hyperparams)
        elif model_name == "CatBoost":
            # Check available GPU memory
            gpu_free_mem = get_gpu_memory_available()
            if gpu_free_mem is not None and gpu_free_mem < 8000:  # Less than ~8GB free
                logging.warning(f"Low GPU memory ({gpu_free_mem} MiB). Switching CatBoost to CPU.")
                hyperparams['task_type'] = 'CPU'
                hyperparams['devices'] = ''
            else:
                hyperparams['task_type'] = 'GPU'
                hyperparams['devices'] = '0'

            # Limit threads to prevent GPU overuse
            hyperparams['thread_count'] = 2
            hyperparams['early_stopping_rounds'] = 50

            clf = CatBoostClassifier(random_state=42, verbose=0, **hyperparams)
        elif model_name == "GradientBoosting":
            clf = GradientBoostingClassifier(random_state=42, **hyperparams)
        else:
            logging.error(f"Model {model_name} is not recognized.")
            return

        # -----------------------------
        # Create the pipeline
        # -----------------------------
        pipeline = ImbPipeline([
            ('sampler', sampler),
            ('classifier', clf),
        ])

        # -----------------------------
        # Start monitoring system resources
        # -----------------------------
        cpu_usage = get_cpu_usage()
        memory_usage = get_memory_usage()
        logging.info(f"Initial CPU usage: {cpu_usage}%")
        logging.info(f"Initial Memory usage: {memory_usage}%")

        # -----------------------------
        # Fit the model
        # -----------------------------
        start_time = time.time()
        try:
            pipeline.fit(X_train, y_train_copy)
        except Exception as e:
            logging.error(f"Exception during model fitting: {e}")
            logging.error(traceback.format_exc())
            return
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Model training completed in {elapsed_time:.2f} seconds")

        # Log system resource usage after fitting
        cpu_usage = get_cpu_usage()
        memory_usage = get_memory_usage()
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

        # Add logging to verify CatBoost's task_type
        classifier = pipeline.named_steps['classifier']
        if isinstance(classifier, CatBoostClassifier):
            task_type = classifier.get_param('task_type')
            devices = classifier.get_param('devices')
            logging.info(f"CatBoost task_type: {classifier.get_param('task_type')}")
            print(f"CatBoost task_type: {classifier.get_param('task_type')}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X_train,
            y_train_copy,
            cv=cv,
            n_jobs=1,  # Set to 1 to prevent parallel processes
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
        if len(f1_scores) > 0:
            best_threshold = thresholds[np.argmax(f1_scores)]
        else:
            best_threshold = 0.5  # Default threshold
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

        logging.info("one_trial_script.py completed successfully.")

    except Exception as e:
        logging.error("An error occurred: " + str(e))
        logging.error("Traceback: " + traceback.format_exc())
        print("An error occurred during execution. Please check the log file for details.")

if __name__ == "__main__":
    main()
