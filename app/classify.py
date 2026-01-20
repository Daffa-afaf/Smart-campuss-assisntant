"""
Query/Document Classification Module
Production-ready KNN classifier for SmartCampus Assistant RAG system.
K-NN implemented from scratch with cosine similarity.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple
from collections import Counter


class KNNClassifier:
    """K-Nearest Neighbors classifier implemented from scratch.
    
    Uses cosine similarity for distance metric and weighted voting for predictions.
    """
    
    def __init__(self, k: int = 3, metric: str = 'cosine', weighted: bool = True):
        """Initialize KNN classifier.
        
        Args:
            k: Number of neighbors
            metric: Distance metric ('cosine' or 'euclidean')
            weighted: Whether to use distance-weighted voting
        """
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.y_labels = {0: 'FAQ', 1: 'Profile/Academic', 2: 'General'}
        self.is_trained = False
    
    def cosine_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Cosine similarity (0 to 1, higher is more similar)
        """
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 0.0
        
        return dot_product / (norm_x1 * norm_x2)
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def calculate_distances(self, query: np.ndarray) -> List[Tuple[int, float]]:
        """Calculate distances from query to all training samples.
        
        Args:
            query: Query vector (n_features,)
            
        Returns:
            List of (index, distance/similarity) tuples
        """
        distances = []
        
        for i, x_train in enumerate(self.X_train):
            if self.metric == 'cosine':
                # For cosine, higher is better (similarity)
                sim = self.cosine_similarity(query, x_train)
                distances.append((i, sim))
            else:  # euclidean
                # For euclidean, lower is better (distance)
                dist = self.euclidean_distance(query, x_train)
                distances.append((i, dist))
        
        return distances
    
    def get_k_neighbors(self, distances: List[Tuple[int, float]]) -> List[Tuple[int, int, float]]:
        """Get k nearest neighbors.
        
        Args:
            distances: List of (index, distance/similarity) tuples
            
        Returns:
            List of (index, label, distance/similarity) for k neighbors
        """
        # Sort by distance/similarity
        if self.metric == 'cosine':
            # For cosine similarity, sort descending (higher is better)
            sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)
        else:
            # For euclidean, sort ascending (lower is better)
            sorted_distances = sorted(distances, key=lambda x: x[1])
        
        # Get top k neighbors
        k_neighbors = []
        for idx, dist in sorted_distances[:self.k]:
            label = self.y_train[idx]
            k_neighbors.append((idx, int(label), float(dist)))
        
        return k_neighbors
    
    def vote(self, neighbors: List[Tuple[int, int, float]]) -> Tuple[int, float]:
        """Vote among neighbors to determine class.
        
        Implements weighted voting with tie-breaking:
        - If weighted=True, votes are weighted by similarity/distance
        - Tie-breaking: choose class with highest total weight, or closest neighbor
        
        Args:
            neighbors: List of (index, label, distance/similarity) tuples
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if self.weighted:
            # Weighted voting
            votes = {}
            for idx, label, dist in neighbors:
                if self.metric == 'cosine':
                    # Cosine similarity is already 0-1, use directly as weight
                    weight = dist
                else:
                    # Convert distance to weight (closer = higher weight)
                    weight = 1.0 / (1.0 + dist)
                
                votes[label] = votes.get(label, 0.0) + weight
            
            # Find class with highest total weight
            predicted_label = max(votes.items(), key=lambda x: x[1])[0]
            confidence = votes[predicted_label] / sum(votes.values())
        else:
            # Simple majority voting
            labels = [label for _, label, _ in neighbors]
            vote_counts = Counter(labels)
            
            # Get most common class
            most_common = vote_counts.most_common(2)
            predicted_label = most_common[0][0]
            
            # Tie-breaking: if tied, choose class of closest neighbor
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                predicted_label = neighbors[0][1]  # Closest neighbor
            
            confidence = vote_counts[predicted_label] / len(labels)
        
        return predicted_label, confidence
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the classifier (store training data).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.feature_names = feature_names
        self.is_trained = True
    
    def predict(self, query_vector: np.ndarray, return_neighbors: bool = False) -> Tuple:
        """Predict category for a query.
        
        Args:
            query_vector: Feature vector for query (n_features,)
            return_neighbors: If True, return neighbors info
            
        Returns:
            If return_neighbors=False: (label_id, label_name, confidence)
            If return_neighbors=True: (label_id, label_name, confidence, neighbors)
            where neighbors is list of (doc_idx, label, similarity/distance)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Flatten if needed
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()
        
        # Calculate distances to all training samples
        distances = self.calculate_distances(query_vector)
        
        # Get k nearest neighbors
        neighbors = self.get_k_neighbors(distances)
        
        # Vote among neighbors
        predicted_label, confidence = self.vote(neighbors)
        
        label_name = self.y_labels.get(predicted_label, 'Unknown')
        
        if return_neighbors:
            return int(predicted_label), label_name, float(confidence), neighbors
        else:
            return int(predicted_label), label_name, float(confidence)
    
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
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate classifier on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        predictions = []
        for x in X_test:
            pred_label, _, _ = self.predict(x)
            predictions.append(pred_label)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y_test)
        
        # Per-class metrics
        unique_labels = np.unique(y_test)
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for label in unique_labels:
            tp = np.sum((predictions == label) & (y_test == label))
            fp = np.sum((predictions == label) & (y_test != label))
            fn = np.sum((predictions != label) & (y_test == label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        return {
            'accuracy': accuracy,
            'macro_precision': np.mean(precision_per_class),
            'macro_recall': np.mean(recall_per_class),
            'macro_f1': np.mean(f1_per_class),
            'predictions': predictions
        }
    
    def save(self, filepath: str):
        """Save classifier to file."""
        model_data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'k': self.k,
            'metric': self.metric,
            'weighted': self.weighted,
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
        
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        self.k = model_data['k']
        self.metric = model_data.get('metric', 'cosine')
        self.weighted = model_data.get('weighted', True)
        self.feature_names = model_data['feature_names']
        self.y_labels = model_data['y_labels']
        self.is_trained = model_data['is_trained']


# Legacy alias for backward compatibility
QueryClassifier = KNNClassifier