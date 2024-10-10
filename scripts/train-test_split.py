# train-test_split.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import os

# Ensure the directory exists for saving processed data
os.makedirs('../data/processed', exist_ok=True)

# Load the dataset
df_final = pd.read_csv('../data/raw/ad_click_dataset_imputedMICE3.csv')

# Drop the 'full_name' column since it won't be useful for classification
df_final = df_final.drop(columns=['full_name'])

# Define features and target
X = df_final.drop(columns=['click'])  # All features except target
y = df_final['click']  # Target variable

# Handle categorical data
categorical_cols = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']

# OneHot encoding for Logistic Regression
onehot_encoder = OneHotEncoder(sparse_output=False)
X_onehot = pd.DataFrame(onehot_encoder.fit_transform(X[categorical_cols]), columns=onehot_encoder.get_feature_names_out())

# Add any remaining numeric columns (e.g., age)
X_onehot = pd.concat([X_onehot, X.drop(columns=categorical_cols)], axis=1)

# Ordinal encoding for Random Forest
ordinal_encoder = OrdinalEncoder()
X_ordinal = X.copy()
X_ordinal[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])

# Stratified train-test split to maintain class distribution
X_train_ordinal, X_temp_ordinal, y_train, y_temp = train_test_split(X_ordinal, y, test_size=0.3, random_state=42, stratify=y)
X_val_ordinal, X_test_ordinal, y_val, y_test = train_test_split(X_temp_ordinal, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Repeat for OneHot encoded data
X_train_onehot, X_temp_onehot, y_train_onehot, y_temp_onehot = train_test_split(X_onehot, y, test_size=0.3, random_state=42, stratify=y)
X_val_onehot, X_test_onehot, y_val_onehot, y_test_onehot = train_test_split(X_temp_onehot, y_temp_onehot, test_size=0.5, random_state=42, stratify=y_temp_onehot)

# Save processed datasets
X_train_ordinal.to_csv('../data/processed/X_train_ordinal.csv', index=False)
X_val_ordinal.to_csv('../data/processed/X_val_ordinal.csv', index=False)
X_test_ordinal.to_csv('../data/processed/X_test_ordinal.csv', index=False)
X_train_onehot.to_csv('../data/processed/X_train_onehot.csv', index=False)
X_val_onehot.to_csv('../data/processed/X_val_onehot.csv', index=False)
X_test_onehot.to_csv('../data/processed/X_test_onehot.csv', index=False)

# Save target values
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_val.to_csv('../data/processed/y_val.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

print(f"Training Set (OneHot): {X_train_onehot.shape}, Validation Set (OneHot): {X_val_onehot.shape}, Test Set (OneHot): {X_test_onehot.shape}")
print(f"Training Set (Ordinal): {X_train_ordinal.shape}, Validation Set (Ordinal): {X_val_ordinal.shape}, Test Set (Ordinal): {X_test_ordinal.shape}")
print("Data successfully saved in /processed directory.")
