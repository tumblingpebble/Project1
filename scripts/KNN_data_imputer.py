import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.impute import KNNImputer

df = read_csv("ad_click_dataset.csv")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

imputer = KNNImputer(n_neighbors=5)

imputed_numerical = imputer.fit_transform(df[numerical_cols])

imputed_numerical_df = pd.DataFrame(imputed_numerical, columns=numerical_cols)

for col in categorical_cols:
    most_frequent_value = df[col].mode()[0]
    df[col].fillna(most_frequent_value, inplace=True)

cleaned_df = pd.concat([imputed_numerical_df, df[categorical_cols]], axis=1)

cleaned_df.to_csv('KNN_ad_click_dataset.csv', index=False)

improved_df = read_csv('KNN_ad_click_dataset.csv')

# histogram for ages BEFORE imputation
plt.figure(figsize=(10,6))
plt.hist(df['age'], bins=10, color='blue', edgecolor='black')
plt.title('Ages of Users (original)')
plt.xlabel('Ages')
plt.ylabel('frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()

# histogram for ages AFTER imputation
plt.figure(figsize=(10,6))
plt.hist(improved_df['age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Ages of Users (after cleaning)')
plt.xlabel('Ages')
plt.ylabel('frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()

# bar graph function
def bar_plot(csv_file, column_name, title):
    data = read_csv(csv_file)
    data.columns = data.columns.str.strip()
    counts = data[column_name].value_counts()
    plt.figure(figsize=(10,6))
    counts.plot(kind='bar', color=['lightblue', 'salmon', 'gray', 'navy', 'lightgreen'])
    plt.xlabel(column_name.capitalize())
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=0)
    plt.show()

# bar graphs for gender before and after imputation
bar_plot('ad_click_dataset.csv', 'gender', 'Frequency of Gender (original)')
bar_plot('KNN_ad_click_dataset.csv', 'gender', 'Frequency of Gender (imputed)')

# bar graphs for device type before and after imputation
bar_plot('ad_click_dataset.csv', 'device_type', 'Frequency of Device Types (original)')
bar_plot('KNN_ad_click_dataset.csv', 'device_type', 'Frequency of Device Types (imputed)')

# bar graphs for browsing history before and after imputation
bar_plot('ad_click_dataset.csv', 'browsing_history', 'Browsing History (original)')
bar_plot('KNN_ad_click_dataset.csv', 'browsing_history', 'Browsing History (imputed)')

# bar graphs for time of day before and after imputation
bar_plot('ad_click_dataset.csv', 'time_of_day', 'Time of Day (original)')
bar_plot('KNN_ad_click_dataset.csv', 'time_of_day', 'Time of Day (imputed)')

# bar graphs for ad position before and after imputation
bar_plot('ad_click_dataset.csv', 'ad_position', 'Ad Positions (original)')
bar_plot('KNN_ad_click_dataset.csv', 'ad_position', 'Ad Positions (imputed)')