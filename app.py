import streamlit as st
import pandas as pd
from PIL import Image
import joblib
import os
import numpy as np
import sklearn
from datetime import datetime
import clustering

# Set the page configuration (this should be the first st command)
st.set_page_config(
    page_title="Bank Credit Analysis",
    page_icon="💰",
    layout="wide"
)

# --- Define a function to load the data ---
@st.cache_data
def load_data(filepath):
    """Loads the cleaned data CSV."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found.")
        st.info("Please make sure 'cleaned_bank_credit_data.csv' is in the same folder as 'app.py'")
        return None

# --- Define functions to load models ---
@st.cache_resource
def load_model(filepath):
    """Loads a saved joblib model."""
    try:
        model = joblib.load(filepath)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{filepath}' not found.")
        st.info("Please make sure your .joblib model files are in the same folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading model {filepath}: {e}")
        return None

# --- Load the Data & Models ---
df = load_data('cleaned_bank_credit_data.csv')
model_amount = load_model('bank_credit_model.joblib')
model_accounts = load_model('account_prediction_model.joblib')
rf_model_amount = load_model('rf_amount_model.joblib')
rf_model_accounts = load_model('rf_accounts_model.joblib')

# --- Diagnostic helper (toggleable in sidebar) ---
def _show_diagnostics():
    """Return a dict of diagnostics about expected files and model load status."""
    expected_files = [
        'cleaned_bank_credit_data.csv',
        'bank_credit_model.joblib',
        'account_prediction_model.joblib',
        'rf_amount_model.joblib',
        'rf_accounts_model.joblib',
        'bank_group_rf_classifier.joblib',
        'bank_group_lr_classifier.joblib',
        'region_rf_classifier.joblib',
        'region_lr_classifier.joblib'
    ]

    files_status = {f: os.path.exists(os.path.join(os.getcwd(), f)) for f in expected_files}

    models_status = {
        'model_amount_loaded': model_amount is not None,
        'model_accounts_loaded': model_accounts is not None,
        'rf_model_amount_loaded': rf_model_amount is not None,
        'rf_model_accounts_loaded': rf_model_accounts is not None
    }

    data_loaded = df is not None

    # file timestamps
    file_timestamps = {}
    for f in expected_files:
        path = os.path.join(os.getcwd(), f)
        if os.path.exists(path):
            file_timestamps[f] = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        else:
            file_timestamps[f] = None

    # Prepare simple evaluation metrics if data and models are available
    model_metrics = {}
    try:
        if df is not None:
            # Regression amount model metrics
            try:
                feat_amt = ['region', 'population_group', 'bank_group', 'occupation_group', 'year', 'no_of_accounts']
                y_amt = np.log1p(df['amount_outstanding'])
                X_amt = df[feat_amt]
                if model_amount is not None:
                    r2_amt = float(model_amount.score(X_amt, y_amt))
                else:
                    r2_amt = None
                if rf_model_amount is not None:
                    r2_rf_amt = float(rf_model_amount.score(X_amt, y_amt))
                else:
                    r2_rf_amt = None
                model_metrics['amount_outstanding'] = {'lr_r2_log': r2_amt, 'rf_r2_log': r2_rf_amt}
            except Exception:
                model_metrics['amount_outstanding'] = {'error': 'could not compute metrics for amount model'}

            # Regression accounts model metrics
            try:
                feat_acc = ['region', 'population_group', 'bank_group', 'occupation_group', 'year', 'credit_limit']
                y_acc = np.log1p(df['no_of_accounts'])
                X_acc = df[feat_acc]
                if model_accounts is not None:
                    r2_acc = float(model_accounts.score(X_acc, y_acc))
                else:
                    r2_acc = None
                if rf_model_accounts is not None:
                    r2_rf_acc = float(rf_model_accounts.score(X_acc, y_acc))
                else:
                    r2_rf_acc = None
                model_metrics['no_of_accounts'] = {'lr_r2_log': r2_acc, 'rf_r2_log': r2_rf_acc}
            except Exception:
                model_metrics['no_of_accounts'] = {'error': 'could not compute metrics for accounts model'}

            # Classification model metrics if bundled classifier exists
            try:
                import classification_models as cls_mod
                if isinstance(classification_models, cls_mod.BankCreditClassifier):
                    cls = classification_models
                    cls_metrics = {}
                    tasks = ['bank_group', 'region', 'population_group', 'occupation_group']
                    from sklearn.model_selection import train_test_split
                    for task in tasks:
                        try:
                            Xc, yc = cls.prepare_classification_data(df, task)
                            X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
                            for model_name in ['Random Forest', 'Logistic Regression']:
                                key = f"{task}_{model_name}"
                                mdl = cls.models.get(key)
                                if mdl is not None:
                                    acc = float(mdl.score(X_test, y_test))
                                else:
                                    acc = None
                                cls_metrics.setdefault(task, {})[model_name] = acc
                        except Exception:
                            cls_metrics.setdefault(task, {})['error'] = 'failed to compute'
                    model_metrics['classification'] = cls_metrics
            except Exception:
                # No bundled classifier or failed to compute
                pass

    except Exception:
        model_metrics['error'] = 'exception while computing metrics'

    return {
        'files': files_status,
        'file_timestamps': file_timestamps,
        'models': models_status,
        'data_loaded': data_loaded,
        'sklearn_version': sklearn.__version__,
        'model_metrics': model_metrics
    }

# Show diagnostics toggle in the sidebar (off by default)
show_diag = st.sidebar.checkbox("Show diagnostics", value=False)
if show_diag:
    diag = _show_diagnostics()
    st.sidebar.subheader("Diagnostics")
    for fname, exists in diag['files'].items():
        if exists:
            st.sidebar.success(f"Found: {fname}")
        else:
            st.sidebar.warning(f"Missing: {fname}")

    st.sidebar.markdown("---")
    for mname, loaded in diag['models'].items():
        if loaded:
            st.sidebar.success(f"Loaded: {mname}")
        else:
            st.sidebar.error(f"Not loaded: {mname}")

    if diag['data_loaded']:
        st.sidebar.success("Data loaded: cleaned_bank_credit_data.csv")
    else:
        st.sidebar.error("Data not loaded: cleaned_bank_credit_data.csv")

    st.sidebar.markdown("---")
    # sklearn version
    try:
        st.sidebar.write(f"sklearn version: {diag.get('sklearn_version')}")
    except Exception:
        pass

    # File timestamps
    st.sidebar.subheader("File timestamps")
    for fname, ts in diag.get('file_timestamps', {}).items():
        if ts:
            st.sidebar.info(f"{fname}: {ts}")
        else:
            st.sidebar.write(f"{fname}: (missing)")

    # Model evaluation metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model metrics (approx)")
    mm = diag.get('model_metrics', {})
    if not mm:
        st.sidebar.write("No model metrics available.")
    else:
        # Regression metrics
        amt = mm.get('amount_outstanding')
        if amt:
            if 'error' in amt:
                st.sidebar.write("Amount model: could not compute metrics")
            else:
                st.sidebar.write(f"Amount (LR R2 on log): {amt.get('lr_r2_log')}")
                st.sidebar.write(f"Amount (RF R2 on log): {amt.get('rf_r2_log')}")

        acc = mm.get('no_of_accounts')
        if acc:
            if 'error' in acc:
                st.sidebar.write("Accounts model: could not compute metrics")
            else:
                st.sidebar.write(f"Accounts (LR R2 on log): {acc.get('lr_r2_log')}")
                st.sidebar.write(f"Accounts (RF R2 on log): {acc.get('rf_r2_log')}")

        cls_mm = mm.get('classification')
        if cls_mm:
            st.sidebar.markdown("**Classification accuracies (test set)**")
            for task, m in cls_mm.items():
                if 'error' in m:
                    st.sidebar.write(f"{task}: error computing metrics")
                else:
                    st.sidebar.write(f"{task}:")
                    for model_name, accv in m.items():
                        st.sidebar.write(f" - {model_name}: {accv}")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Dashboard", "Exploratory Data Analysis (EDA)", "Make a Prediction", "Classification Analysis", "Clustering Analysis", "Deep Learning Analysis"])

# ======================================================================================
# --- Page 1: Main Dashboard ---
# ======================================================================================
if page == "Main Dashboard":

    st.title("💰 Indian Bank Credit Dashboard")
    st.markdown("This dashboard analyzes scheduled commercial bank credit data.")
    
    if df is not None:
        
        # --- Sidebar Filters (for this page) ---
        st.sidebar.header("Data Filters")
        
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        selected_year = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year) 
        )

        all_regions = df['region'].unique()
        selected_regions = st.sidebar.multiselect(
            "Select Region(s)",
            options=sorted(all_regions),
            default=all_regions
        )
        
        all_bank_groups = df['bank_group'].unique()
        selected_bank_groups = st.sidebar.multiselect(
            "Select Bank Group(s)",
            options=sorted(all_bank_groups),
            default=all_bank_groups
        )

        df_filtered = df[
            (df['year'] >= selected_year[0]) &
            (df['year'] <= selected_year[1]) &
            (df['region'].isin(selected_regions)) &
            (df['bank_group'].isin(selected_bank_groups))
        ]

        st.subheader("High-Level KPIs (for filtered data)")
        total_outstanding = df_filtered['amount_outstanding'].sum()
        total_accounts = df_filtered['no_of_accounts'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Amount Outstanding", f"₹ {total_outstanding/100:,.2f} Cr")
        col2.metric("Total Number of Accounts", f"{total_accounts:,.0f}")
        col3.metric("Total Districts", f"{df_filtered['district_name'].nunique()}")

        st.markdown("---")

        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.subheader("Amount Outstanding by Bank Group")
            bank_group_data = df_filtered.groupby('bank_group')['amount_outstanding'].sum().sort_values(ascending=False)
            st.bar_chart(bank_group_data)

        with fig_col2:
            st.subheader("Top 10 States by Amount Outstanding")
            state_data = df_filtered.groupby('state_name')['amount_outstanding'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(state_data)
            
        with st.expander("Show Filtered Raw Data"):
            st.dataframe(df_filtered)

    else:
        st.warning("Data could not be loaded. Dashboard cannot be displayed.")


# ======================================================================================
# --- Page 2: Exploratory Data Analysis (EDA) ---
# ======================================================================================
elif page == "Exploratory Data Analysis (EDA)":
    
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("This page shows a static analysis of the *entire* dataset.")
    st.info("If you don't see any plots, please run your `eda.py` script first to generate the image files.")

    st.subheader("Univariate Analysis: Numerical Distributions")
    col1, col2, col3 = st.columns(3)
    with col1:
        try: st.image(Image.open('accounts_distribution_histogram.png'))
        except FileNotFoundError: st.warning("File not found.")
    with col2:
        try: st.image(Image.open('credit_limit_distribution_histogram.png'))
        except FileNotFoundError: st.warning("File not found.")
    with col3:
        try: st.image(Image.open('amount_outstanding_distribution_histogram.png'))
        except FileNotFoundError: st.warning("File not found.")
            
    st.markdown("---")
    st.subheader("Univariate Analysis: Categorical Counts")
    col1, col2, col3 = st.columns(3)
    with col1:
        try: st.image(Image.open('region_count_plot.png'))
        except FileNotFoundError: st.warning("File not found.")
    with col2:
        try: st.image(Image.open('bank_group_count_plot.png'))
        except FileNotFoundError: st.warning("File not found.")
    with col3:
        try: st.image(Image.open('population_group_count_plot.png'))
        except FileNotFoundError: st.warning("File not found.")

    st.markdown("---")
    st.subheader("Bivariate Analysis: Relationships and Trends")
    col1, col2 = st.columns(2)
    with col1:
        try: st.image(Image.open('amount_by_region_barplot.png'))
        except FileNotFoundError: st.warning("File not found.")
    with col2:
        try: st.image(Image.open('amount_by_bank_group_barplot.png'))
        except FileNotFoundError: st.warning("File not found.")
    try:
        st.image(Image.open('amount_over_time_lineplot.png'))
    except FileNotFoundError: st.warning("File not found.")
        
    st.markdown("---")
    st.subheader("Correlation Analysis")
    col1, col2 = st.columns(2)
    with col1:
        try: st.image(Image.open('limit_vs_outstanding_scatterplot.png'))
        except FileNotFoundError: st.warning("File not found.")
    with col2:
        try: st.image(Image.open('correlation_heatmap.png'))
        except FileNotFoundError: st.warning("File not found.")
            

# ======================================================================================
# --- Page 3: Make a Prediction ---
# ======================================================================================
elif page == "Make a Prediction":
    
    st.title("📈 Make a New Prediction")
    st.markdown("Use our trained machine learning models to get a prediction.")

    # Only show page if data and models loaded correctly
    if df is not None and model_amount is not None and model_accounts is not None:
        
        # --- 1. Get Unique Values for Dropdowns ---
        regions = sorted(df['region'].unique())
        pop_groups = sorted(df['population_group'].unique())
        bank_groups = sorted(df['bank_group'].unique())
        occupation_groups = sorted(df['occupation_group'].unique())
        
        # --- 2. User Selects Model Type and Target ---
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ("Linear Regression", "Random Forest")
            )
        with col2:
            prediction_target = st.selectbox(
                "What do you want to predict?",
                ("Loan Amount (Outstanding)", "Number of Accounts")
            )
        
        st.markdown("---")

        # --- 3. Create Input Form ---
        with st.form("prediction_form"):
            st.subheader("Enter the features for prediction:")
            
            # --- Common Features for Both Models ---
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("Year", 
                                       min_value=df['year'].min(), 
                                       max_value=2030, 
                                       value=2024)
                region = st.selectbox("Region", options=regions)
                bank_group = st.selectbox("Bank Group", options=bank_groups)
            
            with col2:
                population_group = st.selectbox("Population Group", options=pop_groups)
                occupation_group = st.selectbox("Occupation Group", options=occupation_groups)

            # --- Dynamic Features based on Prediction Target ---
            if prediction_target == "Loan Amount (Outstanding)":
                st.markdown("##### Model-Specific Feature:")
                no_of_accounts = st.number_input("Number of Accounts", 
                                                 min_value=0, 
                                                 value=100)
            
            else: # Predict Number of Accounts
                st.markdown("##### Model-Specific Feature:")
                credit_limit = st.number_input("Credit Limit", 
                                               min_value=0.0, 
                                               value=100.0, 
                                               format="%.2f")

            # --- Submit Button ---
            submit_button = st.form_submit_button(label="Get Prediction")

        # --- 4. Handle Prediction ---
        if submit_button:
            # Create a dictionary of all inputs
            inputs = {
                'region': region,
                'population_group': population_group,
                'bank_group': bank_group,
                'occupation_group': occupation_group,
                'year': year
            }
            
            # Select the correct model based on type and target
            if prediction_target == "Loan Amount (Outstanding)":
                inputs['no_of_accounts'] = no_of_accounts
                if model_type == "Linear Regression":
                    model_to_use = model_amount
                else:
                    model_to_use = rf_model_amount
                result_prefix = "₹"
                result_suffix = " Cr. (Outstanding Amount)"
                
            else: # Predict Number of Accounts
                inputs['credit_limit'] = credit_limit
                if model_type == "Linear Regression":
                    model_to_use = model_accounts
                else:
                    model_to_use = rf_model_accounts
                result_prefix = ""
                result_suffix = " Accounts"

            # Convert inputs to a DataFrame
            input_df = pd.DataFrame([inputs])
            
            try:
                # Get the log-scaled prediction
                prediction_log = model_to_use.predict(input_df)
                
                # Convert prediction back to original scale
                prediction_orig = np.expm1(prediction_log[0])
                
                # Display the result
                st.success("Prediction complete!")
                
                # Show both predictions if both models are available
                if prediction_target == "Loan Amount (Outstanding)" and rf_model_amount is not None and model_amount is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Linear Regression Prediction",
                            value=f"₹ {np.expm1(model_amount.predict(input_df)[0]):,.2f} Cr"
                        )
                    with col2:
                        st.metric(
                            label=f"Random Forest Prediction",
                            value=f"₹ {prediction_orig:,.2f} Cr"
                        )
                elif prediction_target == "Number of Accounts" and rf_model_accounts is not None and model_accounts is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=f"Linear Regression Prediction",
                            value=f"{np.expm1(model_accounts.predict(input_df)[0]):,.0f} Accounts"
                        )
                    with col2:
                        st.metric(
                            label=f"Random Forest Prediction",
                            value=f"{prediction_orig:,.0f} Accounts"
                        )
                else:
                    st.metric(
                        label=f"Predicted {prediction_target} ({model_type})",
                        value=f"{result_prefix} {prediction_orig:,.2f}{result_suffix}"
                    )
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please make sure all required model files are available.")

    else:
        st.error("Data or models could not be loaded. The prediction page cannot be displayed.")

# ======================================================================================
# --- Page 4: Classification Analysis ---
# ======================================================================================
elif page == "Classification Analysis":
    
    st.title("🔍 Classification Analysis")
    st.markdown("Use classification models to predict categorical outcomes from your banking data.")
    
    if df is None:
        st.error("Data could not be loaded. Please make sure 'cleaned_bank_credit_data.csv' is available.")
        st.stop()
    
    # Attempt to load a bundled classification model object (preferred)
    try:
        classification_models = None
        combined_model_file = 'classification_models.pkl'

        if os.path.exists(combined_model_file):
            try:
                data = joblib.load(combined_model_file)
            except Exception as e:
                st.error(f"Failed to load {combined_model_file}: {e}")
                st.stop()

            # If the saved object is an instance of BankCreditClassifier, use it directly
            try:
                # Lazy import to avoid circulars when script is used standalone
                import classification_models as cls_mod
                if isinstance(data, cls_mod.BankCreditClassifier):
                    classification_models = data
                elif isinstance(data, dict):
                    # Build a classifier instance and populate internals
                    classifier = cls_mod.BankCreditClassifier()
                    classifier.models = data.get('models', {})
                    classifier.label_encoders = data.get('label_encoders', {})
                    classification_models = classifier
            except Exception:
                # If anything goes wrong here, fall back to using the raw dict
                classification_models = data

        # If not found, try the old per-task files (legacy support)
        if classification_models is None:
            legacy_files = ['bank_group_rf_classifier.joblib', 'bank_group_lr_classifier.joblib',
                            'region_rf_classifier.joblib', 'region_lr_classifier.joblib']
            found_any = any(os.path.exists(f) for f in legacy_files)
            if found_any:
                simple_models = {}
                for f in legacy_files:
                    try:
                        if os.path.exists(f):
                            simple_models[f] = joblib.load(f)
                    except Exception:
                        simple_models[f] = None
                classification_models = simple_models

        if classification_models is None:
            st.warning("""
            **Classification models not found.**

            The classification training artifacts are missing. To create them run:
            `python create_models.py` from the project root (this will train and save
            classification models as `classification_models.pkl`).
            """)
            st.info("""
            **To fix this issue:**
            1. Ensure `cleaned_bank_credit_data.csv` exists in the project root.
            2. Run `python create_models.py` to train and save classification models.
            3. Re-open this app once `classification_models.pkl` is created.
            """)
            st.stop()
        else:
            st.success("✅ Classification models loaded successfully!")

    except Exception as e:
        st.error(f"Error loading classification models: {e}")
        st.stop()
    
    # Classification interface
    st.subheader("Make a Classification Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        classification_task = st.selectbox(
            "What do you want to predict?",
            ("Bank Group", "Region")
        )
        
        model_choice = st.selectbox(
            "Select Classification Model",
            ("Random Forest", "Logistic Regression")
        )
    
    with col2:
        st.info("""
        **Available Classification Tasks:**
        - **Bank Group**: Predict type of bank
        - **Region**: Predict geographical region
        """)
    
    st.markdown("---")
    
    # Input form for classification
    with st.form("classification_form"):
        st.subheader("Enter Feature Values:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2024)
            region = st.selectbox("Region", options=sorted(df['region'].unique()))
            population_group = st.selectbox("Population Group", options=sorted(df['population_group'].unique()))
        
        with col2:
            occupation_group = st.selectbox("Occupation Group", options=sorted(df['occupation_group'].unique()))
            no_of_accounts = st.number_input("Number of Accounts", min_value=0, value=100)
            credit_limit = st.number_input("Credit Limit (Cr)", min_value=0.0, value=50.0, format="%.2f")
            amount_outstanding = st.number_input("Amount Outstanding (Cr)", min_value=0.0, value=35.0, format="%.2f")
        
        classify_button = st.form_submit_button("Classify")
    
    # Handle classification
    if classify_button:
        # Prepare input data
        input_data = {
            'year': year,
            'region': region,
            'population_group': population_group,
            'occupation_group': occupation_group,
            'no_of_accounts': no_of_accounts,
            'credit_limit': credit_limit,
            'amount_outstanding': amount_outstanding
        }
        
        # Map task names to model keys
        task_mapping = {
            "Bank Group": "bank_group",
            "Region": "region"
        }
        
        model_mapping = {
            "Random Forest": "rf",
            "Logistic Regression": "lr"
        }
        
        target_key = task_mapping[classification_task]
        model_key = model_mapping[model_choice]

        # If classification_models is an instance of BankCreditClassifier, use its API
        try:
            import classification_models as cls_mod
            is_classifier_instance = isinstance(classification_models, cls_mod.BankCreditClassifier)
        except Exception:
            is_classifier_instance = False

        try:
            if is_classifier_instance:
                # Use the classifier's predict method
                result = classification_models.predict(input_data, target_key, model_name=model_choice)
                pred_label = result['prediction']
                confidence = result['confidence']
                all_probs = result['all_probabilities']

                st.success("Classification Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Predicted {classification_task}", value=pred_label)
                    st.metric(label="Confidence Level", value=f"{confidence:.2%}")
                with col2:
                    st.subheader("All Probabilities:")
                    for cls_label, prob in all_probs.items():
                        st.write(f"**{cls_label}**: {prob:.2%}")

                with st.expander("View Input Summary"):
                    st.json(input_data)

            else:
                # Assume classification_models is a dict of legacy sklearn pipelines
                full_model_key = f"{target_key}_{model_key}"
                model = classification_models.get(full_model_key) if isinstance(classification_models, dict) else None
                if model is None:
                    st.error(f"Model {full_model_key} not found. Please train the classification models first.")
                    st.stop()

                # Convert input data to DataFrame for prediction
                input_df = pd.DataFrame([input_data])
                try:
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    class_labels = model.classes_
                except ValueError as ve:
                    # Provide a clearer, actionable error for feature name mismatches
                    msg = str(ve)
                    if 'Feature names' in msg or 'feature names' in msg:
                        st.error("Classification error: the saved model expects different feature names or preprocessed inputs.")
                        st.info("Possible fixes:\n - Use the bundled `classification_models.pkl` (BankCreditClassifier) which applies consistent preprocessing before prediction.\n - (Recommended) Re-train classification models by running `python classification_models.py` or `python create_models.py` from the project root so the app has compatible pipelines.\n - Ensure the model was saved as a pipeline that includes preprocessing (LabelEncoder/Scaler).")
                    else:
                        st.error(f"Classification error: {msg}")
                    st.stop()

                st.success("Classification Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Predicted {classification_task}", value=prediction)
                    confidence = np.max(probabilities)
                    st.metric(label="Confidence Level", value=f"{confidence:.2%}")
                with col2:
                    st.subheader("All Probabilities:")
                    for i, class_label in enumerate(class_labels):
                        prob = probabilities[i]
                        st.write(f"**{class_label}**: {prob:.2%}")

                with st.expander("View Input Summary"):
                    st.json(input_data)

        except Exception as e:
            st.error(f"Classification error: {str(e)}")
    
    # Model information
    st.markdown("---")
    st.subheader("About Classification Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Random Forest Classifier:**
        - Ensemble method using multiple decision trees
        - Handles non-linear relationships well
        - Good for complex patterns
        """)
    
    with col2:
        st.info("""
        **Logistic Regression:**
        - Linear model for classification
        - Fast and interpretable
        - Good for linearly separable data
        """)

# ======================================================================================
# --- Page 5: Clustering Analysis ---
# ======================================================================================
elif page == "Clustering Analysis":
    st.title("🧩 Clustering Analysis")
    st.markdown("Use unsupervised clustering to find patterns and groups in your data.")

    if df is None:
        st.error("Data could not be loaded. Please make sure 'cleaned_bank_credit_data.csv' is available.")
        st.stop()

    st.markdown("---")
    st.subheader("Choose features for clustering")
    all_columns = list(df.columns)
    default_features = ['region', 'population_group', 'bank_group', 'occupation_group', 'year', 'no_of_accounts']
    selected_features = st.multiselect("Select features (at least 2)", options=all_columns, default=[c for c in default_features if c in all_columns])

    st.subheader("Algorithm & parameters")
    algorithm = st.selectbox("Algorithm", ("KMeans", "DBSCAN", "Agglomerative"))
    params = {}
    if algorithm == 'KMeans' or algorithm == 'Agglomerative':
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=4)
        params['n_clusters'] = n_clusters
        if algorithm == 'Agglomerative':
            params['linkage'] = st.selectbox('Linkage', ('ward', 'complete', 'average', 'single'))
    elif algorithm == 'DBSCAN':
        params['eps'] = st.number_input('eps', min_value=0.01, value=0.5, step=0.01)
        params['min_samples'] = st.number_input('min_samples', min_value=1, value=5, step=1)

    run_btn = st.button('Run Clustering')

    if run_btn:
        if not selected_features or len(selected_features) < 2:
            st.error('Select at least 2 features for clustering.')
        else:
            with st.spinner('Running clustering...'):
                try:
                    result = clustering.run_clustering(df, selected_features, algorithm=algorithm, params=params)

                    labels = result['labels']
                    metrics = result['metrics']
                    X_pca = result['X_pca']

                    st.success('Clustering complete')

                    # Show silhouette
                    sil = metrics.get('silhouette')
                    if sil is not None:
                        st.metric('Silhouette Score', f"{sil:.4f}")
                    else:
                        st.info('Silhouette score not available for this clustering result')

                    # Cluster sizes
                    import pandas as _pd
                    cluster_counts = _pd.Series(labels).value_counts().sort_index()
                    st.subheader('Cluster sizes')
                    st.table(cluster_counts.reset_index().rename(columns={0: 'count', 'index': 'cluster'}))

                    # PCA scatter
                    if X_pca is not None:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=20)
                        ax.set_xlabel('PC1')
                        ax.set_ylabel('PC2')
                        ax.set_title(f'{algorithm} clusters (PCA projection)')
                        st.pyplot(fig)

                    # Show sample rows per cluster
                    st.subheader('Sample rows by cluster')
                    sample_df = df[selected_features].copy()
                    sample_df['_cluster'] = labels
                    for cl in sorted(sample_df['_cluster'].unique()):
                        st.markdown(f'**Cluster {cl}** (n={int((sample_df["_cluster"]==cl).sum())})')
                        st.dataframe(sample_df[sample_df['_cluster']==cl].head(5))

                    # Save model
                    model_file = f'clustering_{algorithm.lower()}.pkl'
                    clustering.save_clustering_model(result, model_file)
                    st.info(f'Clustering model saved to {model_file}')

                except Exception as e:
                    st.error(f'Clustering failed: {e}')

# ======================================================================================
# --- Page 6: Deep Learning Analysis ---
# ======================================================================================
elif page == "Deep Learning Analysis":
    st.title("🧠 Deep Learning Analysis")
    st.markdown("Use deep learning models to uncover complex patterns in your data.")

    if df is None:
        st.error("Data could not be loaded. Please make sure 'cleaned_bank_credit_data.csv' is available.")
        st.stop()

    import deep_learning

    st.markdown("---")
    st.subheader("Choose Analysis Type")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ("Autoencoder (Dimensionality Reduction)", "Deep Neural Network (Classification)")
    )

    if analysis_type == "Autoencoder (Dimensionality Reduction)":
        st.markdown("""
        ### Autoencoder Analysis
        Autoencoder will learn a compressed representation of your data, which can be useful for:
        - Dimensionality reduction
        - Feature learning
        - Anomaly detection
        """)

        # Feature selection
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_features = st.multiselect(
            "Select numerical features for the autoencoder",
            options=numeric_columns,
            default=['credit_limit', 'amount_outstanding', 'no_of_accounts']
        )

        if len(selected_features) > 0:
            # Parameters
            encoding_dim = st.slider(
                "Encoding dimension (compressed representation size)",
                min_value=2,
                max_value=min(32, len(selected_features)),
                value=min(8, len(selected_features))
            )

            epochs = st.slider("Number of training epochs", 10, 100, 50)
            
            if st.button("Train Autoencoder"):
                with st.spinner("Training autoencoder..."):
                    try:
                        # Prepare data
                        X = df[selected_features].values

                        # Create and train autoencoder
                        dl_model = deep_learning.DeepLearningModel()
                        dl_model.create_autoencoder(input_dim=len(selected_features), encoding_dim=encoding_dim)
                        
                        # Train
                        dl_model.train(X, epochs=epochs)
                        
                        # Get encoded representation
                        encoded_data = dl_model.get_embeddings(X)
                        
                        st.success("Training complete!")
                        
                        # Visualization of encoded data (if 2D or 3D)
                        if encoding_dim in [2, 3]:
                            import plotly.express as px
                            
                            if encoding_dim == 2:
                                fig = px.scatter(
                                    x=encoded_data[:, 0],
                                    y=encoded_data[:, 1],
                                    color=df['bank_group'],
                                    title="2D Encoded Representation"
                                )
                                st.plotly_chart(fig)
                            else:  # 3D
                                fig = px.scatter_3d(
                                    x=encoded_data[:, 0],
                                    y=encoded_data[:, 1],
                                    z=encoded_data[:, 2],
                                    color=df['bank_group'],
                                    title="3D Encoded Representation"
                                )
                                st.plotly_chart(fig)
                        
                        # Save model
                        dl_model.save_model('autoencoder_model.h5')
                        st.info("Model saved as 'autoencoder_model.h5'")
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

    else:  # Deep Neural Network
        st.markdown("""
        ### Deep Neural Network Classification
        Train a deep neural network to classify banking data with high accuracy.
        """)

        # Target selection
        target_col = st.selectbox(
            "Select target for classification",
            ("bank_group", "region", "population_group", "occupation_group")
        )

        # Feature selection
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_features = st.multiselect(
            "Select features for classification",
            options=[col for col in df.columns if col != target_col],
            default=numeric_columns
        )

        if len(selected_features) > 0:
            epochs = st.slider("Number of training epochs", 10, 100, 50)
            
            if st.button("Train Deep Neural Network"):
                with st.spinner("Training neural network..."):
                    try:
                        from sklearn.preprocessing import LabelEncoder
                        
                        # Prepare data
                        X = df[selected_features]
                        le = LabelEncoder()
                        y = le.fit_transform(df[target_col])
                        
                        # Create and train model
                        dl_model = deep_learning.DeepLearningModel()
                        dl_model.create_deep_classifier(
                            input_dim=len(selected_features),
                            num_classes=len(le.classes_)
                        )
                        
                        # Train
                        dl_model.train(X, y, epochs=epochs)
                        
                        st.success("Training complete!")
                        
                        # Plot training history
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=dl_model.history.history['loss'],
                            name='Training Loss'
                        ))
                        fig.add_trace(go.Scatter(
                            y=dl_model.history.history['val_loss'],
                            name='Validation Loss'
                        ))
                        fig.update_layout(title='Training History',
                                        xaxis_title='Epoch',
                                        yaxis_title='Loss')
                        st.plotly_chart(fig)
                        
                        # Save model
                        dl_model.save_model('deep_classifier_model.h5')
                        st.info("Model saved as 'deep_classifier_model.h5'")
                        
                        # Add prediction interface
                        st.subheader("Make predictions with trained model")
                        
                        # Create input fields for each feature
                        input_data = {}
                        for feature in selected_features:
                            if df[feature].dtype in ['int64', 'float64']:
                                input_data[feature] = st.number_input(
                                    f"Enter {feature}",
                                    value=float(df[feature].mean())
                                )
                            else:
                                input_data[feature] = st.selectbox(
                                    f"Select {feature}",
                                    options=df[feature].unique()
                                )
                        
                        if st.button("Predict"):
                            # Prepare input for prediction
                            input_df = pd.DataFrame([input_data])
                            
                            # Make prediction
                            pred_probs = dl_model.predict(input_df)
                            pred_class = le.inverse_transform([pred_probs.argmax()])[0]
                            
                            # Show prediction
                            st.subheader("Prediction Results")
                            st.metric("Predicted Class", pred_class)
                            
                            # Show probabilities
                            st.subheader("Class Probabilities")
                            probs_df = pd.DataFrame({
                                'Class': le.classes_,
                                'Probability': pred_probs[0]
                            }).sort_values('Probability', ascending=False)
                            st.dataframe(probs_df)
                            
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")