# Bank Credit Analysis Project

A comprehensive data analysis and machine learning application for analyzing Indian bank credit data. This project provides various analytical tools including exploratory data analysis, predictive modeling, classification, clustering, and deep learning capabilities.

## Features

1. **Main Dashboard**
   - Overview of key banking metrics
   - Interactive filters for year, region, and bank groups
   - Visualizations of amount outstanding by bank group and state

2. **Exploratory Data Analysis (EDA)**
   - Detailed statistical analysis
   - Distribution visualizations
   - Correlation analysis
   - Trend analysis over time

3. **Predictive Modeling**
   - Loan amount prediction
   - Number of accounts prediction
   - Choice between Linear Regression and Random Forest models
   - Interactive prediction interface

4. **Classification Analysis**
   - Bank group classification
   - Region classification
   - Support for both Random Forest and Logistic Regression
   - Model performance metrics

5. **Clustering Analysis**
   - Multiple clustering algorithms (KMeans, DBSCAN, Agglomerative)
   - Interactive feature selection
   - Cluster visualization using PCA
   - Silhouette score evaluation

6. **Deep Learning Analysis**
   - Autoencoder for dimensionality reduction
   - Deep Neural Network for classification
   - Interactive training process
   - Real-time visualization of training progress

## Setup Instructions

1. **Environment Setup**
   ```bash
   python -m venv env
   .\env\Scripts\Activate.ps1  # For Windows
   pip install -r requirements.txt
   ```

2. **Required Files**
   - cleaned_bank_credit_data.csv
   - Trained models (.joblib files)
   - classification_models.pkl

3. **Running the Application**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
DS_Project/
│
├── app.py                         # Main Streamlit application
├── create_models.py              # Script to create and train models
├── classification_models.py       # Classification model implementations
├── clustering.py                 # Clustering implementations
├── deep_learning.py             # Deep learning model implementations
├── eda.py                       # Exploratory data analysis scripts
├── preprocess.py                # Data preprocessing utilities
│
├── data/
│   └── cleaned_bank_credit_data.csv
│
└── models/
    ├── bank_credit_model.joblib
    ├── account_prediction_model.joblib
    ├── rf_amount_model.joblib
    ├── rf_accounts_model.joblib
    └── classification_models.pkl
```

## Usage Guide

### Main Dashboard
- Use the sidebar filters to select specific time periods, regions, or bank groups
- View aggregate statistics and visualizations
- Export filtered data if needed

### Making Predictions
1. Select the model type (Linear Regression or Random Forest)
2. Choose what to predict (Loan Amount or Number of Accounts)
3. Input the required features
4. Click "Get Prediction" to see results

### Clustering Analysis
1. Select features for clustering
2. Choose clustering algorithm and parameters
3. Run clustering to see results
4. View PCA visualization and cluster statistics

### Deep Learning Analysis
1. Choose between Autoencoder or Deep Neural Network
2. Select features and set parameters
3. Train the model and view progress
4. Make predictions using the trained model

## Model Performance

The project includes various model types with different performance characteristics:

- **Linear Regression Models**: Good for understanding feature relationships
- **Random Forest Models**: Better for capturing non-linear patterns
- **Clustering Models**: Useful for discovering natural groupings
- **Deep Learning Models**: Best for complex patterns and feature learning

## Maintenance

Regular maintenance tasks:
1. Update data regularly for new predictions
2. Retrain models periodically with new data
3. Monitor model performance
4. Update dependencies as needed

## Troubleshooting

Common issues and solutions:

1. **Missing Files**
   - Check if all required data and model files are present
   - Run create_models.py to regenerate models

2. **Model Errors**
   - Ensure input data matches expected format
   - Check if models are properly loaded

3. **Visualization Issues**
   - Verify data is properly filtered
   - Check for missing values

## Future Improvements

Potential enhancements:
1. Add more advanced deep learning models
2. Implement real-time data updates
3. Add more interactive visualizations
4. Enhance model explainability
5. Add batch prediction capabilities
6. Implement model versioning

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.