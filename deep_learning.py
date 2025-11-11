import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

class DeepLearningModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def create_autoencoder(self, input_dim, encoding_dim=32):
        # Encoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoder = keras.layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoder = keras.layers.Dense(encoding_dim, activation='relu')(encoder)
        
        # Decoder
        decoder = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoder)
        decoder = keras.layers.Dense(input_dim, activation='sigmoid')(decoder)
        
        # Autoencoder
        self.model = keras.Model(input_layer, decoder)
        self.model.compile(optimizer='adam', loss='mse')
        
    def create_deep_classifier(self, input_dim, num_classes):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        self.model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    def preprocess_data(self, X):
        """Preprocess the input data using StandardScaler"""
        try:
            # Try to transform the data using the existing fit
            return self.scaler.transform(X)
        except (AttributeError, NotFittedError):
            # If the scaler hasn't been fit yet, fit and transform
            return self.scaler.fit_transform(X)

    def train(self, X, y=None, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model"""
        X_processed = self.preprocess_data(X)
        
        if y is None:  # Autoencoder training
            self.history = self.model.fit(
                X_processed, X_processed,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
        else:  # Classifier training
            self.history = self.model.fit(
                X_processed, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
    
    def get_embeddings(self, X):
        """Get the encoded representations (embeddings) from the autoencoder"""
        if not isinstance(self.model, keras.Model):
            raise ValueError("Model must be an autoencoder to get embeddings")
            
        X_processed = self.preprocess_data(X)
        # Create a new model that outputs the encoding layer
        encoder = keras.Model(self.model.input, self.model.layers[2].output)
        return encoder.predict(X_processed)
    
    def predict(self, X):
        """Make predictions using the trained model"""
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
    
    def save_model(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a saved model from disk"""
        self.model = keras.models.load_model(filepath)