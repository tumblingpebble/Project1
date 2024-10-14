Project Overview
The primary goal of this project is to build machine learning models that can accurately predict ad clicks. The project follows these main steps:

Data Preprocessing and Imputation: Handling missing values using advanced imputation techniques.
Data Visualization: Comparing distributions before and after data cleaning.
Data Splitting: Creating training, validation, and test sets with proper encoding.
Model Training: Training base models and optimizing them using hyperparameter tuning.
Model Evaluation: Evaluating model performance using various metrics and visualizations.
Dataset Description
The dataset used in this project is ad_click_dataset.csv, which contains the following features:

id: Unique identifier for each user.
full_name: User's full name.
age: Age of the user.
gender: Gender of the user.
device_type: Type of device used.
ad_position: Position of the ad on the page.
browsing_history: User's browsing history category.
time_of_day: Time when the ad was displayed.
click: Target variable indicating whether the ad was clicked (1) or not (0).

Requirements
Python 3.8 or higher
See requirements.txt for a full list of packages.

Usage

1. Data Preprocessing and Imputation
Script: data_imputation.py

This script performs data preprocessing, including handling missing values using a custom imputer that combines IterativeImputer with random sampling for certain categorical variables.

Run: python3 scripts/data_imputation.py

2. Data Visualization
Script: plot_compare.py

This script generates visual comparisons between the original and cleaned datasets, including histograms, boxplots, KDE plots, and count plots for both numerical and categorical variables.

Run: python3 scripts/plot_compare.py

3. Train Test Split
Script: train_test_split.py

This script split the data into training, validation, and test sets with stratification to maintain class distribution.  Also handles encoding for categorical variables.

Run: python3 scripts/train_test_split.py

4. Base Model Training
Script: base_model_training.py

This script trains base models using default hyperparameters and evaluates them.

Run: python3 scripts/base_model_training.py

5. Optimized Model Training
Script: optimized_model_training.py

This script performs hyperparameter optimization using Optuna and retrains the models with the best-found parameters.

Run: python scripts/optimized_model_training.py

Acknowledgments

Contributors: Jonathan Aguilar, Sakshi Srivastava, Ian-Bailey Soemarna

Dataset Source: https://www.kaggle.com/datasets/marius2303/ad-click-prediction-dataset/data

Libraries and Tools:
Pandas
NumPy
Scikit-Learn
Matplotlib
Seaborn
XGBoost
CatBoost
Imbalanced-Learn
Optuna
