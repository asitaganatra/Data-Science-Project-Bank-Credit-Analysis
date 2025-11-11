# Deployment Guide - Bank Credit Analysis

## Local Deployment (Recommended for Testing)

### Prerequisites
- Python 3.10 or higher
- Git installed
- Virtual environment set up

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/asitaganatra/Data-Science-Project-Bank-Credit-Analysis.git
   cd Data-Science-Project-Bank-Credit-Analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv env
   # On Windows:
   .\env\Scripts\Activate.ps1
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

The app will open at `http://localhost:8501`

---

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Repository pushed to GitHub
- Streamlit Cloud account (free)

### Steps

1. **Fork or create a repository on GitHub**
   - Push your code to GitHub (ensure `app.py`, `requirements.txt`, and all data files are committed)
   - Make sure the repository is public or you have access to it

2. **Go to Streamlit Cloud**
   - Visit https://streamlit.io/cloud
   - Click "New app"

3. **Configure deployment**
   - Repository: Select your GitHub repo
   - Branch: Select `main` (or your branch)
   - Main file path: `app.py`
   - App URL: Name your app (e.g., `bank-credit-analysis`)

4. **Deploy**
   - Click "Deploy"
   - Streamlit will install dependencies from `requirements.txt` and run your app
   - Your app will be live at `https://[app-name].streamlit.app`

### Important Notes for Streamlit Cloud

- **Data Files**: All CSV and model files (`.joblib`, `.pkl`, `.h5`) must be committed to your GitHub repository
- **Large Files**: If files exceed GitHub's limits, consider using:
  - Streamlit Secrets for API keys and configurations
  - Remote data sources (S3, databases, etc.)
  - Git LFS for large model files

- **Requirements**: Ensure `requirements.txt` contains all dependencies:
  ```
  streamlit>=1.24.0
  pandas>=1.5.0
  numpy>=1.23.0
  scikit-learn>=1.2.2
  tensorflow-cpu>=2.13.0
  joblib>=1.3.0
  plotly>=5.13.0
  matplotlib>=3.7.0
  pillow>=9.5.0
  keras>=2.13.0
  ```

---

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Hub account (optional, for sharing images)

### Steps

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build Docker image**
   ```bash
   docker build -t bank-credit-app:latest .
   ```

3. **Run Docker container**
   ```bash
   docker run -p 8501:8501 bank-credit-app:latest
   ```

The app will be available at `http://localhost:8501`

---

## AWS/Heroku Deployment

### AWS Elastic Beanstalk

1. Install AWS CLI and EB CLI
2. Create `.ebextensions/` with proper configuration
3. Deploy using: `eb create` and `eb deploy`

### Heroku

1. Create `Procfile` with: `web: streamlit run app.py --server.port=$PORT`
2. Push to Heroku: `git push heroku main`

---

## Troubleshooting

### Missing Data/Model Files
- Ensure all `.csv`, `.joblib`, `.pkl`, and `.h5` files are in the repository
- Check that files are not in `.gitignore`
- If files are too large, use Git LFS

### Dependencies Issues
- Update `requirements.txt` with all required packages
- Test locally first: `pip install -r requirements.txt`
- Check Python version compatibility

### Memory Issues on Cloud
- TensorFlow models can be memory-intensive
- Consider using `tensorflow-cpu` instead of `tensorflow`
- Optimize model sizes or use model compression

### Slow Performance
- Enable caching with `@st.cache_data` and `@st.cache_resource`
- Use `st.write()` sparingly; prefer specific display functions
- Optimize data loading for large datasets

---

## File Structure

Ensure your repository has this structure for successful deployment:

```
Data-Science-Project-Bank-Credit-Analysis/
├── app.py
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets_template.toml
├── cleaned_bank_credit_data.csv
├── bank_credit_model.joblib
├── account_prediction_model.joblib
├── rf_amount_model.joblib
├── rf_accounts_model.joblib
├── classification_models.pkl
├── clustering.py
├── deep_learning.py
├── classification_models.py
├── eda.py
├── preprocess.py
└── README.md
```

---

## Environment Variables

### Streamlit Cloud Secrets (`.streamlit/secrets.toml`)

If you need to add secrets for production:

```toml
# Database connections (if applicable)
database_url = "your_database_url"
api_key = "your_api_key"
```

Access in code:
```python
import streamlit as st
db_url = st.secrets["database_url"]
```

---

## Performance Optimization Tips

1. **Use caching effectively**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv("data.csv")
   ```

2. **Lazy load heavy modules**
   ```python
   if user_selects_deep_learning:
       import tensorflow as tf
   ```

3. **Optimize model predictions**
   - Batch predictions when possible
   - Use GPU-accelerated libraries

4. **Monitor resource usage**
   - Check Streamlit Cloud logs
   - Monitor memory and CPU usage locally

---

## Getting Help

- **Streamlit Documentation**: https://docs.streamlit.io
- **Streamlit Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Create an issue in the repository
- **Project README**: See `README.md` for project-specific information
