import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
print("Loading data...")
df = pd.read_csv('cleaned_bank_credit_data.csv')

def evaluate_classifier(model, X, y, target_names):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Detailed classification report
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

# Features for classification
features = ['credit_limit', 'amount_outstanding', 'no_of_accounts', 'year']

print("\n=== Bank Group Classification ===")
# Load models for bank_group classification
rf_bank = joblib.load('bank_group_rf_classifier.joblib')
lr_bank = joblib.load('bank_group_lr_classifier.joblib')

# Prepare data
le_bank = LabelEncoder()
X = df[features]
y_bank = le_bank.fit_transform(df['bank_group'])
bank_groups = list(le_bank.classes_)

print("\nRandom Forest - Bank Group Classification:")
evaluate_classifier(rf_bank, X, y_bank, bank_groups)

print("\nLogistic Regression - Bank Group Classification:")
evaluate_classifier(lr_bank, X, y_bank, bank_groups)

print("\n=== Region Classification ===")
# Load models for region classification
rf_region = joblib.load('region_rf_classifier.joblib')
lr_region = joblib.load('region_lr_classifier.joblib')

# Prepare data
le_region = LabelEncoder()
y_region = le_region.fit_transform(df['region'])
regions = list(le_region.classes_)

print("\nRandom Forest - Region Classification:")
evaluate_classifier(rf_region, X, y_region, regions)

print("\nLogistic Regression - Region Classification:")
evaluate_classifier(lr_region, X, y_region, regions)