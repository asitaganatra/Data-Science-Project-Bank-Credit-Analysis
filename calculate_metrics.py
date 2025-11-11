import pandas as pd
import numpy as np
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                           classification_report, silhouette_score)
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Load data
print("Loading data...")
df = pd.read_csv('cleaned_bank_credit_data.csv')

# 1. Regression Models Metrics
print("\n=== Regression Models Metrics ===")

# Features for each model
amount_features = ['region', 'population_group', 'bank_group', 'occupation_group', 'year', 'no_of_accounts']
account_features = ['region', 'population_group', 'bank_group', 'occupation_group', 'year', 'credit_limit']

# Load models
lr_amount = joblib.load('bank_credit_model.joblib')
rf_amount = joblib.load('rf_amount_model.joblib')
lr_accounts = joblib.load('account_prediction_model.joblib')
rf_accounts = joblib.load('rf_accounts_model.joblib')

# Prepare data
X_amount = df[amount_features]
y_amount = np.log1p(df['amount_outstanding'])
X_accounts = df[account_features]
y_accounts = np.log1p(df['no_of_accounts'])

# Split data
X_amount_train, X_amount_test, y_amount_train, y_amount_test = train_test_split(
    X_amount, y_amount, test_size=0.2, random_state=42
)
X_accounts_train, X_accounts_test, y_accounts_train, y_accounts_test = train_test_split(
    X_accounts, y_accounts, test_size=0.2, random_state=42
)

# Calculate metrics for Amount Prediction
print("\nAmount Outstanding Prediction Metrics:")
models_amount = {
    'Linear Regression': lr_amount,
    'Random Forest': rf_amount
}

for name, model in models_amount.items():
    y_pred = model.predict(X_amount_test)
    r2 = r2_score(y_amount_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_amount_test, y_pred))
    mae = mean_absolute_error(y_amount_test, y_pred)
    
    # Convert log metrics back to original scale
    rmse_orig = np.mean(np.abs(np.expm1(y_amount_test) - np.expm1(y_pred)))
    
    print(f"\n{name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE (log scale): {rmse:.4f}")
    print(f"MAE (log scale): {mae:.4f}")
    print(f"RMSE (original scale): {rmse_orig:.2f} Cr")

# Calculate metrics for Account Prediction
print("\nNumber of Accounts Prediction Metrics:")
models_accounts = {
    'Linear Regression': lr_accounts,
    'Random Forest': rf_accounts
}

for name, model in models_accounts.items():
    y_pred = model.predict(X_accounts_test)
    r2 = r2_score(y_accounts_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_accounts_test, y_pred))
    mae = mean_absolute_error(y_accounts_test, y_pred)
    
    # Convert log metrics back to original scale
    rmse_orig = np.mean(np.abs(np.expm1(y_accounts_test) - np.expm1(y_pred)))
    
    print(f"\n{name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE (log scale): {rmse:.4f}")
    print(f"MAE (log scale): {mae:.4f}")
    print(f"RMSE (original scale): {rmse_orig:.2f} accounts")

# 2. Clustering Metrics
print("\n=== Clustering Metrics ===")
import clustering

# Features for clustering
cluster_features = ['credit_limit', 'amount_outstanding', 'no_of_accounts']

algorithms = {
    'KMeans': {'algorithm': 'KMeans', 'params': {'n_clusters': 4}},
    'DBSCAN': {'algorithm': 'DBSCAN', 'params': {'eps': 0.5, 'min_samples': 5}},
    'Agglomerative': {'algorithm': 'Agglomerative', 'params': {'n_clusters': 4}}
}

for name, config in algorithms.items():
    print(f"\n{name} Clustering:")
    result = clustering.run_clustering(df, cluster_features, 
                                    algorithm=config['algorithm'],
                                    params=config['params'])
    
    print(f"Silhouette Score: {result['metrics']['silhouette']:.4f}")
    
    # Cluster sizes
    unique, counts = np.unique(result['labels'], return_counts=True)
    print("Cluster sizes:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} samples ({count/len(df)*100:.1f}%)")

# 3. Deep Learning Metrics
print("\n=== Deep Learning Metrics ===")
import deep_learning

# Autoencoder metrics
print("\nAutoencoder Metrics:")
features = ['credit_limit', 'amount_outstanding', 'no_of_accounts']
X = df[features].values

dl_model = deep_learning.DeepLearningModel()
dl_model.create_autoencoder(input_dim=len(features), encoding_dim=2)
dl_model.train(X, epochs=50)

final_loss = dl_model.history.history['loss'][-1]
final_val_loss = dl_model.history.history['val_loss'][-1]

print(f"Final Training Loss: {final_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Classification DNN metrics
print("\nDeep Neural Network Classification Metrics:")
# Prepare data for bank_group classification
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['bank_group'])
X = df[features].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train DNN classifier
dl_model = deep_learning.DeepLearningModel()
dl_model.create_deep_classifier(input_dim=len(features), num_classes=len(le.classes_))
dl_model.train(X_train, y_train, epochs=50)

# Get predictions
y_pred = np.argmax(dl_model.predict(X_test), axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))