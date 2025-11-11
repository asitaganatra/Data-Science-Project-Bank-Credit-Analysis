import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib


def _build_preprocessor(df, selected_features):
    # Determine categorical and numeric among selected features
    dtypes = df[selected_features].dtypes
    numeric_cols = [c for c in selected_features if np.issubdtype(dtypes[c], np.number)]
    categorical_cols = [c for c in selected_features if c not in numeric_cols]

    # Create OneHotEncoder in a way compatible with different sklearn versions
    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    if categorical_cols:
        # Handle sklearn API differences: older versions accept `sparse`, newer use `sparse_output`.
        try:
            from inspect import signature
            sig = signature(OneHotEncoder)
            if 'sparse' in sig.parameters:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            elif 'sparse_output' in sig.parameters:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            else:
                # fallback
                ohe = OneHotEncoder(handle_unknown='ignore')
        except Exception:
            ohe = OneHotEncoder(handle_unknown='ignore')
        transformers.append(('cat', ohe, categorical_cols))

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    else:
        preprocessor = None

    return preprocessor, numeric_cols, categorical_cols


def run_clustering(df, selected_features, algorithm='KMeans', params=None):
    """Run clustering on `df[selected_features]`.
    algorithm: 'KMeans'|'DBSCAN'|'Agglomerative'
    params: dict of algorithm-specific params

    Returns: dict with labels, model, metrics, X_transformed (preprocessed), X_pca (2D), and pipeline
    """
    if params is None:
        params = {}

    preprocessor, num_cols, cat_cols = _build_preprocessor(df, selected_features)

    # Prepare X
    X_raw = df[selected_features].copy()
    if preprocessor is not None:
        # Use pipeline to transform
        X = preprocessor.fit_transform(X_raw)
    else:
        X = X_raw.values

    # Choose algorithm
    model = None
    labels = None
    metrics = {}

    if algorithm == 'KMeans':
        n_clusters = int(params.get('n_clusters', 4))
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)

    elif algorithm == 'DBSCAN':
        eps = float(params.get('eps', 0.5))
        min_samples = int(params.get('min_samples', 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

    elif algorithm == 'Agglomerative':
        n_clusters = int(params.get('n_clusters', 4))
        linkage = params.get('linkage', 'ward')
        # ward requires euclidean and numeric data; if categorical present, fallback to 'average'
        if linkage == 'ward' and cat_cols:
            linkage = 'average'
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Compute silhouette score when possible (requires more than 1 cluster and not all same)
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
            sil = silhouette_score(X, labels)
        else:
            sil = None
    except Exception:
        sil = None

    metrics['silhouette'] = sil

    # PCA 2D for visualization
    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
    except Exception:
        X_pca = None

    result = {
        'labels': labels,
        'model': model,
        'metrics': metrics,
        'preprocessor': preprocessor,
        'X_preprocessed': X,
        'X_pca': X_pca,
        'selected_features': selected_features
    }

    return result


def save_clustering_model(result, filename):
    # Save the parts needed to re-run/predict: preprocessor and model
    data = {
        'preprocessor': result.get('preprocessor'),
        'model': result.get('model'),
        'selected_features': result.get('selected_features')
    }
    joblib.dump(data, filename)


def load_clustering_model(filename):
    return joblib.load(filename)
