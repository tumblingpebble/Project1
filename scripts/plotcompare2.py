#plotcompare2.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the datasets
df_original = pd.read_csv('../data/raw/ad_click_dataset.csv')
df_cleaned = pd.read_csv('../data/raw/ad_click_dataset_imputedMICE3.csv')

# Define a list of columns to visualize (you can customize this)
columns_to_visualize = ['age', 'gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']  # Update based on your actual feature names

# Create a directory for saving the images (if it doesn't exist)

if not os.path.exists('../visualizations'):
    os.makedirs('../visualizations')

# Function to plot histograms for numeric columns
def plot_histograms(df_original, df_cleaned, column_name):
    if pd.api.types.is_numeric_dtype(df_original[column_name]):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        axs[0].hist(df_original[column_name].dropna(), bins=20, color='blue', alpha=0.7)
        axs[0].set_title(f'{column_name} Distribution (Original)')
        axs[0].set_xlabel(column_name)
        axs[0].set_ylabel('Frequency')
        
        axs[1].hist(df_cleaned[column_name].dropna(), bins=20, color='green', alpha=0.7)
        axs[1].set_title(f'{column_name} Distribution (After Cleaning)')
        axs[1].set_xlabel(column_name)
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'../visualizations/{column_name}_histogram_comparison.png')
        plt.close()

# Function to plot boxplots for numeric columns
def plot_boxplots(df_original, df_cleaned, column_name):
    if pd.api.types.is_numeric_dtype(df_original[column_name]):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].boxplot(df_original[column_name].dropna(), patch_artist=True)
        axs[0].set_title(f'{column_name} Boxplot (Original)')
        axs[0].set_ylabel(column_name)

        axs[1].boxplot(df_cleaned[column_name].dropna(), patch_artist=True)
        axs[1].set_title(f'{column_name} Boxplot (After Cleaning)')
        axs[1].set_ylabel(column_name)

        plt.tight_layout()
        plt.savefig(f'../visualizations/{column_name}_boxplot_comparison.png')
        plt.close()

# Function to plot KDE plots for numeric columns
def plot_kde(df_original, df_cleaned, column_name):
    if pd.api.types.is_numeric_dtype(df_original[column_name]):
        plt.figure(figsize=(10, 6))

        sns.kdeplot(df_original[column_name].dropna(), label='Original', color='blue')
        sns.kdeplot(df_cleaned[column_name].dropna(), label='After Cleaning', color='green')

        plt.title(f'KDE Plot for {column_name} (Original vs Cleaned)')
        plt.xlabel(column_name)
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'../visualizations/{column_name}_kde_comparison.png')
        plt.close()

# Function to compare summary statistics for numeric columns
def compare_summary_stats(df_original, df_cleaned, column_name):
    if pd.api.types.is_numeric_dtype(df_original[column_name]):
        original_stats = df_original[column_name].describe()
        cleaned_stats = df_cleaned[column_name].describe()

        summary_df = pd.DataFrame({
            'Original': original_stats,
            'After Cleaning': cleaned_stats
        })

        summary_df.to_csv(f'../visualizations/{column_name}_summary_stats_comparison.csv')
        print(f"Summary statistics for {column_name} saved as CSV.")

# Function to plot count plots for categorical columns with consistent color palettes and data type handling
def plot_countplots(df_original, df_cleaned, column_name):
    if pd.api.types.is_categorical_dtype(df_original[column_name]) or df_original[column_name].dtype == 'object':
        # Convert both original and cleaned data columns to strings to avoid type comparison issues
        df_original[column_name] = df_original[column_name].astype(str)
        df_cleaned[column_name] = df_cleaned[column_name].astype(str)

        # Get the unique categories from both datasets to ensure consistent colors
        categories = sorted(list(set(df_original[column_name].unique()) | set(df_cleaned[column_name].unique())))

        # Create a palette based on the unique categories
        palette = sns.color_palette("Set2", len(categories))
        color_mapping = dict(zip(categories, palette))

        # Plotting both the original and cleaned data
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        sns.countplot(x=df_original[column_name], ax=axs[0], palette=color_mapping)
        axs[0].set_title(f'{column_name} Count Plot (Original)')
        
        sns.countplot(x=df_cleaned[column_name], ax=axs[1], palette=color_mapping)
        axs[1].set_title(f'{column_name} Count Plot (After Cleaning)')

        plt.tight_layout()
        plt.savefig(f'../visualizations/{column_name}_countplot_comparison.png')
        plt.close()


# Iterate over the columns and generate visualizations
for column in df_original.columns:
    print(f"Processing visualizations for {column}...")
    plot_histograms(df_original, df_cleaned, column)
    plot_boxplots(df_original, df_cleaned, column)
    plot_kde(df_original, df_cleaned, column)
    compare_summary_stats(df_original, df_cleaned, column)
    plot_countplots(df_original, df_cleaned, column)

print("All visualizations saved in the 'visualizations' directory.")
