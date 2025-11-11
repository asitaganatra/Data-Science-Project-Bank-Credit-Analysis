import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

# --- Setup ---
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # Default size for plots

# --- Load Data ---
print("Loading 'cleaned_bank_credit_data.csv'...")
try:
    df = pd.read_csv('cleaned_bank_credit_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'cleaned_bank_credit_data.csv' not found.")
    print("Please make sure the file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- 1. Numerical Analysis ---
print("\n--- 1. Numerical Data Summary ---")
# Get descriptive statistics for numerical columns
numerical_stats = df[['no_of_accounts', 'credit_limit', 'amount_outstanding']].describe()
print(numerical_stats)

print("\nGenerating histograms for numerical data...")

# Plot for 'no_of_accounts'
plt.figure(figsize=(10, 6))
sns.histplot(df['no_of_accounts'], kde=True, bins=50)
plt.title('Distribution of Number of Accounts')
plt.xlabel('Number of Accounts')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('accounts_distribution_histogram.png')
print("Saved 'accounts_distribution_histogram.png'")

# Plot for 'credit_limit'
plt.figure(figsize=(10, 6))
sns.histplot(df['credit_limit'], kde=True, bins=50, color='green')
plt.title('Distribution of Credit Limit')
plt.xlabel('Credit Limit')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('credit_limit_distribution_histogram.png')
print("Saved 'credit_limit_distribution_histogram.png'")

# Plot for 'amount_outstanding'
plt.figure(figsize=(10, 6))
sns.histplot(df['amount_outstanding'], kde=True, bins=50, color='red')
plt.title('Distribution of Amount Outstanding')
plt.xlabel('Amount Outstanding')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('amount_outstanding_distribution_histogram.png')
print("Saved 'amount_outstanding_distribution_histogram.png'")

# --- 2. Categorical Analysis ---
print("\n--- 2. Categorical Data Summary ---")
# Get descriptive statistics for categorical columns
categorical_stats = df.describe(include='object')
print(categorical_stats)

print("\nGenerating count plots for categorical data...")

# Plot for 'region'
plt.figure(figsize=(10, 6))
sns.countplot(y=df['region'], order=df['region'].value_counts().index, palette='viridis')
plt.title('Count of Entries by Region')
plt.xlabel('Count')
plt.ylabel('Region')
plt.tight_layout()
plt.savefig('region_count_plot.png')
print("Saved 'region_count_plot.png'")

# Plot for 'bank_group'
plt.figure(figsize=(10, 6))
sns.countplot(y=df['bank_group'], order=df['bank_group'].value_counts().index, palette='plasma')
plt.title('Count of Entries by Bank Group')
plt.xlabel('Count')
plt.ylabel('Bank Group')
plt.tight_layout()
plt.savefig('bank_group_count_plot.png')
print("Saved 'bank_group_count_plot.png'")

# Plot for 'population_group'
plt.figure(figsize=(10, 6))
sns.countplot(y=df['population_group'], order=df['population_group'].value_counts().index, palette='inferno')
plt.title('Count of Entries by Population Group')
plt.xlabel('Count')
plt.ylabel('Population Group')
plt.tight_layout()
plt.savefig('population_group_count_plot.png')
print("Saved 'population_group_count_plot.png'")

# --- 3. Relationship & Trend Analysis ---
print("\n--- 3. Relationship & Trend Analysis ---")

# Bar plot: Sum of Amount Outstanding by Region
print("Generating bar plot for Amount Outstanding by Region...")
plt.figure(figsize=(10, 6))
region_data = df.groupby('region')['amount_outstanding'].sum().sort_values(ascending=False)
sns.barplot(x=region_data.values, y=region_data.index, palette='viridis')
plt.title('Total Amount Outstanding by Region')
plt.xlabel('Total Amount Outstanding (in Crores)')
plt.ylabel('Region')
plt.tight_layout()
plt.savefig('amount_by_region_barplot.png')
print("Saved 'amount_by_region_barplot.png'")

# Bar plot: Sum of Amount Outstanding by Bank Group
print("Generating bar plot for Amount Outstanding by Bank Group...")
plt.figure(figsize=(10, 6))
bank_data = df.groupby('bank_group')['amount_outstanding'].sum().sort_values(ascending=False)
sns.barplot(x=bank_data.values, y=bank_data.index, palette='plasma')
plt.title('Total Amount Outstanding by Bank Group')
plt.xlabel('Total Amount Outstanding (in Crores)')
plt.ylabel('Bank Group')
plt.tight_layout()
plt.savefig('amount_by_bank_group_barplot.png')
print("Saved 'amount_by_bank_group_barplot.png'")

# Line plot: Sum of Amount Outstanding over Years
print("Generating line plot for Amount Outstanding over Time...")
plt.figure(figsize=(10, 6))
year_data = df.groupby('year')['amount_outstanding'].sum()
sns.lineplot(x=year_data.index, y=year_data.values, marker='o')
plt.title('Total Amount Outstanding Over Years')
plt.xlabel('Year')
plt.ylabel('Total Amount Outstanding (in Crores)')
# Ensure x-axis shows years as integers if possible
plt.xticks(ticks=year_data.index.unique(), rotation=45) 
plt.tight_layout()
plt.savefig('amount_over_time_lineplot.png')
print("Saved 'amount_over_time_lineplot.png'")

# Scatter plot: Credit Limit vs. Amount Outstanding
print("Generating scatter plot for Credit Limit vs. Amount Outstanding...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='credit_limit', y='amount_outstanding', alpha=0.5)
plt.title('Credit Limit vs. Amount Outstanding')
plt.xlabel('Credit Limit')
plt.ylabel('Amount Outstanding')
plt.tight_layout()
plt.savefig('limit_vs_outstanding_scatterplot.png')
print("Saved 'limit_vs_outstanding_scatterplot.png'")

# --- 4. Correlation Analysis ---
print("\n--- 4. Correlation Matrix ---")
numerical_cols = ['no_of_accounts', 'credit_limit', 'amount_outstanding']
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)

# Heatmap for Correlation Matrix
print("Generating correlation heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved 'correlation_heatmap.png'")

print("\n--- EDA analysis complete. All plots saved. ---")