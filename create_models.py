import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("--- Starting Model Creation Script ---")

# --- Load Data ---
try:
    df = pd.read_csv('cleaned_bank_credit_data.csv')
    print("Loaded 'cleaned_bank_credit_data.csv'")
except FileNotFoundError:
    print("ERROR: 'cleaned_bank_credit_data.csv' not found.")
    print("Please make sure the cleaned data file is in this folder before running.")
    exit()

# Define the log-transform function
log_transformer = FunctionTransformer(np.log1p, validate=False)

# ==========================================================
# --- MODEL 1: Predict 'amount_outstanding' ---
# ==========================================================
print("\n--- Building Model 1: Predict 'amount_outstanding' ---")

# --- 1. Define Features (X1) and Target (y1) ---
categorical_features_1 = ['region', 'population_group', 'bank_group', 'occupation_group']
numeric_features_scale_1 = ['year']
numeric_features_log_scale_1 = ['no_of_accounts']
all_features_1 = categorical_features_1 + numeric_features_scale_1 + numeric_features_log_scale_1

X1 = df[all_features_1]
y1 = df['amount_outstanding']
y1_log = np.log1p(y1)

X1_train, X1_test, y1_train_log, y1_test_log = train_test_split(X1, y1_log, test_size=0.2, random_state=42)

# --- 2. Create Preprocessing Pipeline 1 ---
log_pipeline_1 = Pipeline([('log_transform', log_transformer), ('scaler', StandardScaler())])
scale_pipeline_1 = Pipeline([('scaler', StandardScaler())])
categorical_pipeline_1 = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor_1 = ColumnTransformer(
    transformers=[
        ('num_scale', scale_pipeline_1, numeric_features_scale_1),
        ('num_log_scale', log_pipeline_1, numeric_features_log_scale_1),
        ('cat', categorical_pipeline_1, categorical_features_1)
    ])

# --- 3. Create and Train Full Pipeline 1 (Linear Regression) ---
full_pipeline_1 = Pipeline(steps=[
    ('preprocessor', preprocessor_1),
    ('model', LinearRegression())
])

print("Training Linear Regression Model 1...")
full_pipeline_1.fit(X1_train, y1_train_log)

# --- 4. Create Random Forest Model for amount_outstanding ---
print("Training Random Forest Model for amount_outstanding...")
rf_pipeline_amount = Pipeline(steps=[
    ('preprocessor', preprocessor_1),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])
rf_pipeline_amount.fit(X1_train, y1_train_log)

# --- 5. Evaluate Models ---
y1_pred_lr = full_pipeline_1.predict(X1_test)
y1_pred_rf = rf_pipeline_amount.predict(X1_test)

mse_lr = mean_squared_error(y1_test_log, y1_pred_lr)
mse_rf = mean_squared_error(y1_test_log, y1_pred_rf)
r2_lr = r2_score(y1_test_log, y1_pred_lr)
r2_rf = r2_score(y1_test_log, y1_pred_rf)

print(f"Linear Regression - MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}")
print(f"Random Forest - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")

# --- 6. Save Models ---
model_filename_1 = 'bank_credit_model.joblib'
rf_model_filename_1 = 'rf_amount_model.joblib'

joblib.dump(full_pipeline_1, model_filename_1)
joblib.dump(rf_pipeline_amount, rf_model_filename_1)
print(f"SUCCESS: Linear Regression Model 1 saved as '{model_filename_1}'")
print(f"SUCCESS: Random Forest Model 1 saved as '{rf_model_filename_1}'")

# ==========================================================
# --- MODEL 2: Predict 'no_of_accounts' ---
# ==========================================================
print("\n--- Building Model 2: Predict 'no_of_accounts' ---")

# --- 1. Define Features (X2) and Target (y2) ---
categorical_features_2 = ['region', 'population_group', 'bank_group', 'occupation_group']
numeric_features_scale_2 = ['year']
numeric_features_log_scale_2 = ['credit_limit']
all_features_2 = categorical_features_2 + numeric_features_scale_2 + numeric_features_log_scale_2

X2 = df[all_features_2]
y2 = df['no_of_accounts']
y2_log = np.log1p(y2)

X2_train, X2_test, y2_train_log, y2_test_log = train_test_split(X2, y2_log, test_size=0.2, random_state=42)

# --- 2. Create Preprocessing Pipeline 2 ---
log_pipeline_2 = Pipeline([('log_transform', log_transformer), ('scaler', StandardScaler())])
scale_pipeline_2 = Pipeline([('scaler', StandardScaler())])
categorical_pipeline_2 = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor_2 = ColumnTransformer(
    transformers=[
        ('num_scale', scale_pipeline_2, numeric_features_scale_2),
        ('num_log_scale', log_pipeline_2, numeric_features_log_scale_2),
        ('cat', categorical_pipeline_2, categorical_features_2)
    ])

# --- 3. Create and Train Full Pipeline 2 (Linear Regression) ---
full_pipeline_2 = Pipeline(steps=[
    ('preprocessor', preprocessor_2),
    ('model', LinearRegression())
])

print("Training Linear Regression Model 2...")
full_pipeline_2.fit(X2_train, y2_train_log)

# --- 4. Create Random Forest Model for no_of_accounts ---
print("Training Random Forest Model for no_of_accounts...")
rf_pipeline_accounts = Pipeline(steps=[
    ('preprocessor', preprocessor_2),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])
rf_pipeline_accounts.fit(X2_train, y2_train_log)

# --- 5. Evaluate Models ---
y2_pred_lr = full_pipeline_2.predict(X2_test)
y2_pred_rf = rf_pipeline_accounts.predict(X2_test)

mse_lr2 = mean_squared_error(y2_test_log, y2_pred_lr)
mse_rf2 = mean_squared_error(y2_test_log, y2_pred_rf)
r2_lr2 = r2_score(y2_test_log, y2_pred_lr)
r2_rf2 = r2_score(y2_test_log, y2_pred_rf)

print(f"Linear Regression - MSE: {mse_lr2:.4f}, R2: {r2_lr2:.4f}")
print(f"Random Forest - MSE: {mse_rf2:.4f}, R2: {r2_rf2:.4f}")

# --- 6. Save Models ---
model_filename_2 = 'account_prediction_model.joblib'
rf_model_filename_2 = 'rf_accounts_model.joblib'

joblib.dump(full_pipeline_2, model_filename_2)
joblib.dump(rf_pipeline_accounts, rf_model_filename_2)
print(f"SUCCESS: Linear Regression Model 2 saved as '{model_filename_2}'")
print(f"SUCCESS: Random Forest Model 2 saved as '{rf_model_filename_2}'")

print("\nAll models created successfully!")

# Add at the end of your create_models.py file:

# ==========================================================
# --- CLASSIFICATION MODELS ---
# ==========================================================
print("\n--- Building Classification Models ---")

from classification_models import train_classification_models

try:
    classifier, classification_results = train_classification_models('cleaned_bank_credit_data.csv', 'classification_models.pkl')
    print("SUCCESS: Classification models trained and saved!")
    
    # Print results summary
    print("\n=== Classification Results Summary ===")
    for task, results in classification_results.items():
        print(f"\n{task}:")
        for model_name, accuracy in results.items():
            print(f"  {model_name}: {accuracy:.4f}")
            
except Exception as e:
    print(f"ERROR in classification training: {e}")