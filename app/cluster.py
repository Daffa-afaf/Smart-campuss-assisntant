"""
Document Clustering Module
Production-ready K-Means clustering for SmartCampus Assistant RAG system.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class DocumentClusterer:
    """K-Means clustering for document grouping."""
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.feature_names = None
        self.is_trained = False
        self.cluster_labels = {}
    
    def find_optimal_k(self, X: np.ndarray, k_range: range = range(2, 10)) -> int:
        """Find optimal number of clusters using silhouette score.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            k_range: Range of k values to test
            
        Returns:
            Optimal k value
        """
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        self.n_clusters = optimal_k
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        return optimal_k
    
    def train(self, X: np.ndarray, feature_names: List[str] = None, doc_names: List[str] = None):
        """Train the clusterer.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            doc_names: List of document names
        """
        self.kmeans.fit(X)
        self.feature_names = feature_names
        self.is_trained = True
        
        # Create cluster labels mapping
        if doc_names:
            labels = self.kmeans.labels_
            for doc_name, label in zip(doc_names, labels):
                if label not in self.cluster_labels:
                    self.cluster_labels[label] = []
                self.cluster_labels[label].append(doc_name)
    
    def predict(self, query_vector: np.ndarray) -> Tuple[int, float]:
        """Predict cluster for a query.
        
        Args:
            query_vector: Feature vector for query (1, n_features)
            
        Returns:
            Tuple of (cluster_id, distance_to_center)
        """
        if not self.is_trained:
            raise ValueError("Clusterer not trained. Call train() first.")
        
        # Reshape if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Predict cluster
        cluster_id = self.kmeans.predict(query_vector)[0]
        
        # Calculate distance to cluster center
        distances = self.kmeans.transform(query_vector)
        distance = distances[0][cluster_id]
        
        return int(cluster_id), float(distance)
    
    def get_cluster_documents(self, cluster_id: int) -> List[str]:
        """Get documents in a specific cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of document names in the cluster
        """
        return self.cluster_labels.get(cluster_id, [])
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers.
        
        Returns:
            Cluster centers matrix (n_clusters, n_features)
        """
        if not self.is_trained:
            raise ValueError("Clusterer not trained. Call train() first.")
        return self.kmeans.cluster_centers_
    
    def get_top_features_per_cluster(self, top_n: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """Get top features for each cluster.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dict mapping cluster_id to list of (feature_name, value) tuples
        """
        if not self.is_trained or self.feature_names is None:
            raise ValueError("Clusterer not trained or feature names not provided.")
        
        centers = self.kmeans.cluster_centers_
        top_features = {}
        
        for cluster_id in range(self.n_clusters):
            center = centers[cluster_id]
            top_indices = np.argsort(center)[-top_n:][::-1]
            top_features[cluster_id] = [
                (self.feature_names[i], float(center[i])) 
                for i in top_indices
            ]
        
        return top_features
    
    def calculate_silhouette(self, X: np.ndarray) -> float:
        """Calculate silhouette score for current clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Silhouette score
        """
        if not self.is_trained:
            raise ValueError("Clusterer not trained. Call train() first.")
        
        labels = self.kmeans.labels_
        return silhouette_score(X, labels)
    
    def save(self, filepath: str):
        """Save clusterer to file."""
        model_data = {
            'kmeans': self.kmeans,
            'n_clusters': self.n_clusters,
            'feature_names': self.feature_names,
            'cluster_labels': self.cluster_labels,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load clusterer from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.kmeans = model_data['kmeans']
        self.n_clusters = model_data['n_clusters']
        self.feature_names = model_data['feature_names']
        self.cluster_labels = model_data['cluster_labels']
        self.is_trained = model_data['is_trained']
