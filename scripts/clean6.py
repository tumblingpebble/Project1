import pandas as pd
import numpy as np
import time
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Read the dataset
df = pd.read_csv('../data/raw/ad_click_dataset.csv')

# Drop 'id' and 'full_name' as they are not needed for imputation
df_impute = df.drop(['id', 'full_name'], axis=1)

# Identify categorical and numerical columns
categorical_columns = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']
numerical_columns = ['age', 'click']

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

# Encode categorical variables
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
            verbose=2  # verbose output
        )
        self.categorical_imputer = IterativeImputer(
            estimator=RandomForestClassifier(n_estimators=100, random_state=0),
            max_iter=20,
            random_state=0,
            initial_strategy='most_frequent',
            verbose=2  # verbose output
        )
    
    def fit(self, X, y=None):
        # Fit both imputers on the entire dataset
        self.numerical_imputer.fit(X)
        self.categorical_imputer.fit(X)
        return self
    
    def transform(self, X):
        # Impute numerical variables
        X_num = pd.DataFrame(
            self.numerical_imputer.transform(X),
            columns=X.columns
        )
        # Impute categorical variables
        X_cat = pd.DataFrame(
            self.categorical_imputer.transform(X),
            columns=X.columns
        )
        
        # Combine the imputed data
        X_imputed = X_num.copy()
        X_imputed[categorical_columns] = X_cat[categorical_columns]
        
        # Round categorical columns and convert to integers
        X_imputed[categorical_columns] = X_imputed[categorical_columns].round().astype(int)
        
        # Overwrite 'gender' and 'browsing_history' with random sampling
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

# Measure time taken for imputation
start_time = time.time()

# Perform imputation
df_imputed = imputer.fit_transform(df_impute)

end_time = time.time()
print(f"Imputation took {end_time - start_time:.2f} seconds.")

# Round the 'age' column to integers
df_imputed['age'] = df_imputed['age'].round().astype(int)

# Decode categorical variables
df_imputed[categorical_columns] = ordinal_encoder.inverse_transform(df_imputed[categorical_columns])

# Concatenate 'id' and 'full_name' back to the DataFrame
df_final = pd.concat([df[['id', 'full_name']], df_imputed], axis=1)

# Display the final DataFrame
print(df_final)

# , save to a new CSV file
df_final.to_csv('../data/raw/ad_click_dataset_imputedMICE3.csv', index=False)

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
for col in ['gender', 'browsing_history', 'device_type', 'ad_position']:
    compare_distributions(df_impute_decoded, df_final, col)

# Check the distribution of the 'age' column
df_final['age'].hist(bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution After Imputation')
plt.show()
plt.savefig('age_distribution_after_imputation3.png')
