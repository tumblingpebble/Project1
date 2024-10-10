# mod_data_imputation.py

import pandas as pd
import numpy as np
import time
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(0)

# Read the dataset
df = pd.read_csv('../data/raw/ad_click_dataset.csv')

# Identify and aggregate recurring users based on 'full_name'
df_aggregated = df.groupby('full_name').agg({
    'age': 'mean',  # Average age for recurring users
    'gender': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,  # Mode of gender
    'device_type': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'ad_position': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'browsing_history': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'time_of_day': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'click': 'max'  # Use max to indicate if the user ever clicked
}).reset_index()

# Optional: Check for missing values after aggregation
missing_values = df_aggregated.isnull().sum()
print("Missing values after aggregation:")
print(missing_values)

# Drop 'full_name' as it is not needed for imputation
df_impute = df_aggregated.drop(['full_name'], axis=1)

# Identify categorical and numerical columns
categorical_columns = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']
numerical_columns = ['age', 'click']

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
df_impute[categorical_columns] = ordinal_encoder.fit_transform(df_impute[categorical_columns])

# Calculate category probabilities for 'gender' and 'browsing_history'
category_probabilities = {}
for col in ['gender', 'browsing_history']:
    value_counts = df_impute[col].value_counts(normalize=True)
    category_probabilities[col] = value_counts

# Define Custom Imputer Class
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, category_probabilities):
        self.category_probabilities = category_probabilities
        self.numerical_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=0),
            max_iter=20,
            random_state=0,
            verbose=0
        )
        self.categorical_imputer = IterativeImputer(
            estimator=RandomForestClassifier(n_estimators=100, random_state=0),
            max_iter=20,
            random_state=0,
            initial_strategy='most_frequent',
            verbose=0
        )
    
    def fit(self, X, y=None):
        self.numerical_imputer.fit(X[numerical_columns])
        self.categorical_imputer.fit(X[categorical_columns])
        return self
    
    def transform(self, X):
        X_num = pd.DataFrame(
            self.numerical_imputer.transform(X[numerical_columns]),
            columns=numerical_columns
        )
        X_cat = pd.DataFrame(
            self.categorical_imputer.transform(X[categorical_columns]),
            columns=categorical_columns
        )
        
        X_imputed = pd.concat([X_num, X_cat], axis=1)
        X_imputed[categorical_columns] = X_imputed[categorical_columns].round().astype(int)
        
        for col in ['gender', 'browsing_history']:
            missing_mask = X[col].isnull()
            n_missing = missing_mask.sum()
            if n_missing > 0:
                imputed_values = np.random.choice(
                    self.category_probabilities[col].index.astype(int),
                    size=n_missing,
                    p=self.category_probabilities[col].values
                )
                X_imputed.loc[missing_mask, col] = imputed_values
        
        return X_imputed

# Initialize the custom imputer with category probabilities
imputer = CustomImputer(category_probabilities=category_probabilities)
start_time = time.time()
df_imputed = imputer.fit_transform(df_impute)
end_time = time.time()
print(f"Imputation took {end_time - start_time:.2f} seconds.")

# Round age and decode categorical variables
df_imputed['age'] = df_imputed['age'].round().astype(int)
df_imputed[categorical_columns] = ordinal_encoder.inverse_transform(df_imputed[categorical_columns])

# Optionally, save 'full_name' for further use
df_imputed = pd.concat([df_aggregated[['full_name']], df_imputed], axis=1)

# Save the imputed data
df_imputed.to_csv('ad_click_dataset_imputed_aggregated_new.csv', index=False)

# Create directory for visualizations if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Compare distributions before and after imputation
def compare_distributions(df_before, df_after, column):
    counts_before = df_before[column].value_counts(normalize=True).sort_index()
    counts_after = df_after[column].value_counts(normalize=True).sort_index()
    comparison = pd.DataFrame({'Before': counts_before, 'After': counts_after})
    print(f"Distribution of '{column}' before and after imputation:")
    print(comparison)
    print("\n")

# Decode original data for comparison
df_impute_decoded = df_impute.copy()
df_impute_decoded[categorical_columns] = ordinal_encoder.inverse_transform(df_impute_decoded[categorical_columns])

# Compare distributions
for col in categorical_columns:
    compare_distributions(df_impute_decoded, df_imputed, col)

# Plot Age Distribution After Imputation
plt.figure()
df_imputed['age'].hist(bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution After Imputation')
plt.savefig('visualizations/age_distribution_after_imputation_new.png')  # New filename
plt.close()
