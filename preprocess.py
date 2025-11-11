import pandas as pd
import re

def clean_col_name(col_name):
    """
    Cleans up column names.
    - Extracts text from within parentheses (e.g., "Year (year)" -> "year")
    - If no parentheses, converts to lowercase and replaces spaces with underscores.
    """
    match = re.search(r'\((.*?)\)', col_name)
    if match:
        return match.group(1)
    return col_name.strip().replace(' ', '_').lower()

# --- Configuration ---
original_file_path = 'Credit by Scheduled Commercial Banks_Sample_Data.csv'
cleaned_file_path = 'cleaned_bank_credit_data.csv'

# --- 1. Load Data ---
print(f"Loading data from '{original_file_path}'...")
try:
    df = pd.read_csv(original_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at '{original_file_path}'")
    print("Please make sure the file is in the same directory.")
    exit()

# --- 2. Rename Columns ---
print("Renaming columns...")
original_columns = df.columns.tolist()
new_columns = [clean_col_name(col) for col in original_columns]
df.columns = new_columns
print("Columns renamed:")
print(f"From: {original_columns}")
print(f"To:   {new_columns}")

# --- 3. Check for Duplicates ---
duplicate_rows = df.duplicated().sum()
print(f"\nFound {duplicate_rows} duplicate rows.")
# You could remove them by uncommenting the next line:
# if duplicate_rows > 0:
#     df = df.drop_duplicates()
#     print("Duplicate rows removed.")

# --- 4. Check for Missing Values ---
print("\nChecking for missing values (nulls):")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values were found (listed above).")
    # Add your strategy for handling missing values here, e.g.:
    # df = df.dropna()  # To drop rows with any missing values

# --- 5. Save Cleaned Data ---
try:
    df.to_csv(cleaned_file_path, index=False)
    print(f"\nSuccessfully preprocessed and saved data to '{cleaned_file_path}'")
except Exception as e:
    print(f"\nAn error occurred while saving the file: {e}")