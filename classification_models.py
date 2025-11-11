import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class BankCreditClassifier:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.scalers = {}
        
    def prepare_classification_data(self, df, target_column):
        """Prepare data for classification tasks"""
        # Features to use
        # define base features and ensure the target column is not included as a feature
        base_features = ['year', 'region', 'population_group', 'bank_group', 
                         'occupation_group', 'no_of_accounts', 'credit_limit', 'amount_outstanding']
        feature_columns = [c for c in base_features if c != target_column]
        
        # Create a copy
        data = df[feature_columns + [target_column]].copy()
        
        # Encode categorical features
        categorical_cols = ['region', 'population_group', 'bank_group', 'occupation_group']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
        
        # Encode target variable
        if target_column not in self.label_encoders:
            self.label_encoders[target_column] = LabelEncoder()
        data[target_column] = self.label_encoders[target_column].fit_transform(data[target_column].astype(str))
        
        # Handle missing values
        data = data.fillna(0)
        
        X = data[feature_columns]
        y = data[target_column]
        
        return X, y
    
    def train_classification_models(self, df, target_column):
        """Train multiple classification models"""
        print(f"Training classification models for target: {target_column}")
        
        X, y = self.prepare_classification_data(df, target_column)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Models to train
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and results
            self.models[f"{target_column}_{name}"] = pipeline
            results[name] = accuracy
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred, 
                                      target_names=self.label_encoders[target_column].classes_))
            print("---")
        
        return results
    
    def predict(self, input_features, target_column, model_name='Random Forest'):
        """Make classification prediction"""
        model_key = f"{target_column}_{model_name}"
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found. Please train the model first.")
        
        # Prepare input features
        input_df = self.prepare_input_features(input_features)

        # Ensure we do not pass the target column as a feature to the trained pipeline.
        # During training the target column was excluded from the feature set, but
        # `prepare_input_features` returns all base columns (for convenience). Drop
        # the target column here so the input feature set matches what the model
        # was trained on.
        if target_column in input_df.columns:
            input_df = input_df.drop(columns=[target_column])

        # Align input columns to the feature order used during training.
        # The training used `base_features` (defined in prepare_classification_data)
        # excluding the target column. Reindex the input DataFrame to that exact
        # column ordering and fill any missing columns with 0. This ensures
        # compatibility with legacy pipelines that expect the original training
        # feature layout and prevents sklearn feature-name mismatch errors.
        base_features = ['year', 'region', 'population_group', 'bank_group',
                         'occupation_group', 'no_of_accounts', 'credit_limit', 'amount_outstanding']
        expected_features = [c for c in base_features if c != target_column]
        # Reindex to expected features order and fill missing with 0
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # Make prediction
        prediction_encoded = self.models[model_key].predict(input_df)[0]
        
        # Convert back to original label
        prediction_original = self.label_encoders[target_column].inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        probabilities = self.models[model_key].predict_proba(input_df)[0]
        
        return {
            'prediction': prediction_original,
            'confidence': np.max(probabilities),
            'all_probabilities': dict(zip(
                self.label_encoders[target_column].classes_, 
                probabilities
            ))
        }
    
    def prepare_input_features(self, input_dict):
        """Prepare user input for prediction"""
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical variables
        for col in ['region', 'population_group', 'bank_group', 'occupation_group']:
            if col in input_df.columns and col in self.label_encoders:
                try:
                    input_df[col] = self.label_encoders[col].transform([input_dict[col]])[0]
                except ValueError:
                    # Handle unseen labels
                    input_df[col] = 0
        
        # Ensure all required columns are present
        required_columns = ['year', 'region', 'population_group', 'bank_group', 
                           'occupation_group', 'no_of_accounts', 'credit_limit', 'amount_outstanding']
        
        for col in required_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        return input_df[required_columns]
    
    def save_models(self, filepath):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filepath)
        print(f"Classification models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.label_encoders = model_data['label_encoders']
            print(f"Classification models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Training function
def train_classification_models(csv_file_path, model_save_path='classification_models.pkl'):
    """Train all classification models"""
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Initialize classifier
    classifier = BankCreditClassifier()
    
    # Define classification tasks
    classification_tasks = [
        'bank_group',
        'region', 
        'population_group',
        'occupation_group'
    ]
    
    # Train models for each task
    all_results = {}
    for task in classification_tasks:
        print(f"\n=== Training models for {task} ===")
        results = classifier.train_classification_models(df, task)
        all_results[task] = results
    
    # Save all models
    classifier.save_models(model_save_path)
    
    return classifier, all_results

if __name__ == "__main__":
    # Train all classification models
    classifier, results = train_classification_models('cleaned_bank_credit_data.csv')
    
    # Example prediction
    test_input = {
        'year': 2024,
        'region': 'Central Region',
        'population_group': 'Metropolitan',
        'bank_group': 'Foreign Banks',
        'occupation_group': 'Agriculture',
        'no_of_accounts': 100,
        'credit_limit': 50.0,
        'amount_outstanding': 35.0
    }
    
    prediction = classifier.predict(test_input, 'bank_group')
    print(f"\nExample Prediction: {prediction}")