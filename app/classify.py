"""
Query/Document Classification Module
Production-ready KNN classifier for SmartCampus Assistant RAG system.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class QueryClassifier:
    """Classify queries into document categories using KNN."""
    
    def __init__(self, k: int = 3):
        """Initialize classifier.
        
        Args:
            k: Number of neighbors for KNN
        """
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.feature_names = None
        self.y_labels = {0: 'FAQ', 1: 'Profile/Academic', 2: 'General'}
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
        """
        self.knn.fit(X, y)
        self.feature_names = feature_names
        self.is_trained = True
    
    def tune_k(self, X: np.ndarray, y: np.ndarray, k_range: range = range(1, 10)) -> int:
        """Find optimal k using cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            k_range: Range of k values to test
            
        Returns:
            Optimal k value
        """
        cv_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            cv_score = cross_val_score(knn, X, y, cv=3, scoring='accuracy').mean()
            cv_scores.append(cv_score)
        
        optimal_k = list(k_range)[np.argmax(cv_scores)]
        self.k = optimal_k
        self.knn = KNeighborsClassifier(n_neighbors=optimal_k)
        return optimal_k
    
    def predict(self, query_vector: np.ndarray) -> Tuple[int, str, float]:
        """Predict category for a query.
        
        Args:
            query_vector: Feature vector for query (1, n_features)
            
        Returns:
            Tuple of (label_id, label_name, confidence)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Reshape if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Predict
        label_id = self.knn.predict(query_vector)[0]
        
        # Get confidence (distance to neighbors)
        distances, indices = self.knn.kneighbors(query_vector)
        avg_distance = distances[0].mean()
        confidence = 1.0 / (1.0 + avg_distance)  # Convert distance to confidence
        
        label_name = self.y_labels.get(label_id, 'Unknown')
        
        return int(label_id), label_name, float(confidence)
    
    def predict_batch(self, X: np.ndarray) -> List[Tuple[int, str, float]]:
        """Predict categories for multiple queries.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            List of (label_id, label_name, confidence) tuples
        """
        results = []
        for i in range(X.shape[0]):
            results.append(self.predict(X[i]))
        return results
    
    def save(self, filepath: str):
        """Save classifier to file."""
        model_data = {
            'knn': self.knn,
            'k': self.k,
            'feature_names': self.feature_names,
            'y_labels': self.y_labels,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load classifier from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.knn = model_data['knn']
        self.k = model_data['k']
        self.feature_names = model_data['feature_names']
        self.y_labels = model_data['y_labels']
        self.is_trained = model_data['is_trained']
