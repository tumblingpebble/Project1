# script_name: optimized_model_training2.py

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import psutil
import time
import traceback
import subprocess
import datetime

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
    f1_score,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import optuna  # For hyperparameter optimization

# Setup logging
logging.basicConfig(
    filename="optimized_model_training2.log",
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
    X_train = pd.read_csv("../data/processed/X_train_ordinal.csv")
    X_test = pd.read_csv("../data/processed/X_test_ordinal.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

    # Drop unnecessary columns
    columns_to_drop = ["id", "full_name", "name"]
    X_train.drop(columns=[col for col in columns_to_drop if col in X_train.columns], inplace=True)
    X_test.drop(columns=[col for col in columns_to_drop if col in X_test.columns], inplace=True)

    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Hyperparameter optimization function with refined search space
def hyperparameter_optimization(model_name, X_train, y_train):
    logging.info(f"Starting hyperparameter optimization for {model_name}")

    # Best parameters from previous optimization
    best_params_previous = {
        "RandomForest": {'n_estimators': 494, 'max_depth': 19, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': None},
        "XGBoost": {'n_estimators': 448, 'max_depth': 16, 'learning_rate': 0.404, 'subsample': 0.601, 'colsample_bytree': 0.845, 'gamma': 0.032, 'reg_alpha': 1.927, 'reg_lambda': 4.378},
        "CatBoost": {'iterations': 643, 'depth': 9, 'learning_rate': 0.141, 'l2_leaf_reg': 6.983},
        "GradientBoosting": {'n_estimators': 96, 'max_depth': 13, 'learning_rate': 0.090, 'subsample': 0.922, 'max_features': None}
    }

    # Define the refined hyperparameter search spaces based on the best parameters
    search_spaces = {}

    if model_name == "RandomForest":
        bp = best_params_previous["RandomForest"]
        search_spaces["RandomForest"] = {
            'n_estimators': {'type': 'int', 'low': max(bp['n_estimators'] - 50, 100), 'high': bp['n_estimators'] + 50},
            'max_depth': {'type': 'int', 'low': max(bp['max_depth'] - 2, 1), 'high': bp['max_depth'] + 2},
            'min_samples_split': {'type': 'int', 'low': max(bp['min_samples_split'] - 1, 2), 'high': bp['min_samples_split'] + 1},
            'min_samples_leaf': {'type': 'int', 'low': max(bp['min_samples_leaf'], 1), 'high': bp['min_samples_leaf'] + 1},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
        }
    elif model_name == "XGBoost":
        bp = best_params_previous["XGBoost"]
        search_spaces["XGBoost"] = {
            'n_estimators': {'type': 'int', 'low': max(int(bp['n_estimators'] * 0.8), 50), 'high': int(bp['n_estimators'] * 1.2)},
            'max_depth': {'type': 'int', 'low': max(bp['max_depth'] - 2, 1), 'high': bp['max_depth'] + 2},
            'learning_rate': {'type': 'float', 'low': max(bp['learning_rate'] * 0.8, 0.005), 'high': min(bp['learning_rate'] * 1.2, 0.5)},
            'subsample': {'type': 'float', 'low': max(bp['subsample'] - 0.1, 0.5), 'high': min(bp['subsample'] + 0.1, 1.0)},
            'colsample_bytree': {'type': 'float', 'low': max(bp['colsample_bytree'] - 0.1, 0.5), 'high': min(bp['colsample_bytree'] + 0.1, 1.0)},
            'gamma': {'type': 'float', 'low': max(bp['gamma'] - 0.05, 0), 'high': bp['gamma'] + 0.05},
            'reg_alpha': {'type': 'float', 'low': max(bp['reg_alpha'] - 0.5, 0), 'high': bp['reg_alpha'] + 0.5},
            'reg_lambda': {'type': 'float', 'low': max(bp['reg_lambda'] - 0.5, 0), 'high': bp['reg_lambda'] + 0.5},
        }
    elif model_name == "CatBoost":
        bp = best_params_previous["CatBoost"]
        search_spaces["CatBoost"] = {
            'iterations': {'type': 'int', 'low': max(bp['iterations'] - 100, 100), 'high': bp['iterations'] + 100},
            'depth': {'type': 'int', 'low': max(bp['depth'] - 1, 1), 'high': bp['depth'] + 1},
            'learning_rate': {'type': 'float', 'low': max(bp['learning_rate'] * 0.8, 0.005), 'high': min(bp['learning_rate'] * 1.2, 0.5)},
            'l2_leaf_reg': {'type': 'float', 'low': max(bp['l2_leaf_reg'] - 1, 1), 'high': bp['l2_leaf_reg'] + 1},
        }
    elif model_name == "GradientBoosting":
        bp = best_params_previous["GradientBoosting"]
        search_spaces["GradientBoosting"] = {
            'n_estimators': {'type': 'int', 'low': max(bp['n_estimators'] - 50, 50), 'high': bp['n_estimators'] + 50},
            'max_depth': {'type': 'int', 'low': max(bp['max_depth'] - 2, 1), 'high': bp['max_depth'] + 2},
            'learning_rate': {'type': 'float', 'low': max(bp['learning_rate'] * 0.8, 0.005), 'high': min(bp['learning_rate'] * 1.2, 0.5)},
            'subsample': {'type': 'float', 'low': max(bp['subsample'] - 0.1, 0.5), 'high': min(bp['subsample'] + 0.1, 1.0)},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
        }
    else:
        logging.error(f"Model {model_name} not recognized.")
        return None

    search_space = search_spaces[model_name]

    # Function to get hyperparameters for the trial
    def get_params(trial, model_name, search_space):
        params = {}
        for param_name, param_info in search_space.items():
            if param_info['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_info['low'], param_info['high']
                )
            elif param_info['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_info['low'], param_info['high']
                )
            elif param_info['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_info['choices']
                )
        # Add any fixed parameters
        if model_name == "RandomForest":
            params['class_weight'] = 'balanced'
            params['n_jobs'] = -1  # Use all available cores
        elif model_name == "XGBoost":
            params['tree_method'] = 'hist'  # Use CPU
            params['n_jobs'] = -1
            params['eval_metric'] = 'logloss'
        elif model_name == "CatBoost":
            params['task_type'] = 'CPU'
            params['verbose'] = 0
            params['thread_count'] = -1
        elif model_name == "GradientBoosting":
            pass  # No additional fixed parameters
        else:
            logging.error(f"Model {model_name} not recognized.")
            params = None
        return params

    # Objective function for Optuna
    def objective(trial):
        try:
            params = get_params(trial, model_name, search_space)
            if not params:
                return 0.0

            # Initialize classifier with params
            if model_name == "RandomForest":
                clf = RandomForestClassifier(random_state=42, **params)
            elif model_name == "XGBoost":
                clf = XGBClassifier(random_state=42, **params)
            elif model_name == "CatBoost":
                clf = CatBoostClassifier(random_state=42, **params)
            elif model_name == "GradientBoosting":
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

            # Report intermediate value for pruning
            trial.report(scores.mean(), step=0)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return scores.mean()
        except Exception as e:
            logging.error(f"Error in trial: {str(e)}")
            logging.error(traceback.format_exc())
            return 0.0

    # Use Optuna's Multivariate TPE Sampler and ASHA Pruner
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        warn_independent_sampling=False  # Suppress independent sampling warnings
    )
    pruner = optuna.pruners.SuccessiveHalvingPruner()

    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    # Number of trials
    n_trials = 30  # You can adjust this number

    start_time = time.time()

    logging.info(f"Starting optimization with {n_trials} trials for {model_name}")

    # Optimize
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    end_time = time.time()
    logging.info(f"Hyperparameter optimization for {model_name} completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Best parameters for {model_name}: {study.best_params}")
    logging.info(f"Best cross-validation score for {model_name}: {study.best_value}")

    return study.best_params

# Function to generate learning curve
def generate_learning_curve(model, X, y, model_name):
    logging.info(f"Generating learning curve for {model_name}")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(f"../visualizations/{model_name}_learning_curve.png")
    plt.close()

# Function to generate confusion matrix heatmap
def generate_confusion_matrix_heatmap(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"../visualizations/{model_name}_confusion_matrix.png")
    plt.close()

# Function to generate ROC curve
def generate_roc_curve(y_test, y_test_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"../visualizations/{model_name}_roc_curve.png")
    plt.close()

# Function to generate precision-recall curve and adjust threshold
def generate_precision_recall_curve(y_test, y_test_prob, model_name):
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    best_threshold = thresholds[np.argmax(f1_scores)]
    logging.info(f"Best threshold based on F1 score: {best_threshold}")
    y_pred_adjusted = (y_test_prob >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred_adjusted)
    logging.info(f"Classification report with adjusted threshold for {model_name}:\n{report}")

    plt.figure()
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f"../visualizations/{model_name}_precision_recall_curve.png")
    plt.close()

# Function to generate feature importances
def generate_feature_importances(model, feature_names, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {model_name}")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"../visualizations/{model_name}_feature_importances.png")
    plt.close()

def train_and_evaluate(model_name, model_constructor, X_train, X_test, y_train, y_test):
    try:
        logging.info(f"Starting training for {model_name}")

        # Hyperparameter optimization
        best_params = hyperparameter_optimization(model_name, X_train, y_train)

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

        # Save model with timestamp to prevent overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"../models/{model_name}_optimized_model_{timestamp}.pkl"
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
        predictions_filename = f"../predictions/{model_name}_optimized_test_predictions_{timestamp}.csv"
        pd.DataFrame(y_test_pred, columns=["Predictions"]).to_csv(predictions_filename, index=False)
        logging.info(f"Test predictions saved to {predictions_filename}")

        # Generate learning curve
        generate_learning_curve(model, X_train_res, y_train_res, model_name + "_optimized_" + timestamp)

        # Generate confusion matrix heatmap
        generate_confusion_matrix_heatmap(cm, model_name + "_optimized_" + timestamp)

        # Generate ROC curve
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test)[:, 1]
            generate_roc_curve(y_test, y_test_prob, model_name + "_optimized_" + timestamp)

            # Generate precision-recall curve and adjust threshold
            generate_precision_recall_curve(y_test, y_test_prob, model_name + "_optimized_" + timestamp)

        # Generate feature importances if available
        if hasattr(model, "feature_importances_"):
            generate_feature_importances(model, X_train.columns, model_name + "_optimized_" + timestamp)

    except Exception as e:
        logging.error(f"An error occurred while training {model_name}: {str(e)}")
        logging.error(traceback.format_exc())

# Main function to load data, train, and evaluate models
def main():
    logging.info("Starting optimized model training script")
    X_train, X_test, y_train, y_test = load_data()

    # Train and evaluate models
    model_items = list(model_constructors.items())

    for model_name, model_constructor in model_items:
        train_and_evaluate(model_name, model_constructor, X_train, X_test, y_train, y_test)

    logging.info("All models have been trained and evaluated")

if __name__ == "__main__":
    main()
    logging.info("Script ended successfully")
