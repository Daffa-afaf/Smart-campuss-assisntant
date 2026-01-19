"""
IR Evaluation Metrics Module
Production-ready evaluation metrics for information retrieval systems.
"""

import numpy as np
from typing import List, Dict


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@K.
    
    Args:
        retrieved: List of retrieved document IDs in rank order
        relevant: List of relevant document IDs
        k: Cutoff rank
        
    Returns:
        Precision at rank k
    """
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Recall@K.
    
    Args:
        retrieved: List of retrieved document IDs in rank order
        relevant: List of relevant document IDs
        k: Cutoff rank
        
    Returns:
        Recall at rank k
    """
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant) if relevant else 0.0


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Average Precision for a single query.
    
    Args:
        retrieved: List of retrieved document IDs in rank order
        relevant: List of relevant document IDs
        
    Returns:
        Average precision score
    """
    if not relevant:
        return 0.0
    
    scores = []
    hits = 0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            precision_at_i = hits / (i + 1)
            scores.append(precision_at_i)
    
    return float(np.mean(scores)) if scores else 0.0


def mean_average_precision(runs: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    """Calculate Mean Average Precision (MAP) across multiple queries.
    
    Args:
        runs: Dict of {query_id: [retrieved_doc_ids]}
        qrels: Dict of {query_id: [relevant_doc_ids]}
        
    Returns:
        MAP score
    """
    ap_list = []
    for qid, retrieved in runs.items():
        rel = qrels.get(qid, [])
        ap_list.append(average_precision(retrieved, rel))
    return float(np.mean(ap_list)) if ap_list else 0.0


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Reciprocal Rank for a single query.
    
    Args:
        retrieved: List of retrieved document IDs in rank order
        relevant: List of relevant document IDs
        
    Returns:
        Reciprocal rank (1/rank of first relevant document)
    """
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(runs: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        runs: Dict of {query_id: [retrieved_doc_ids]}
        qrels: Dict of {query_id: [relevant_doc_ids]}
        
    Returns:
        MRR score
    """
    rr_list = []
    for qid, retrieved in runs.items():
        rel = qrels.get(qid, [])
        rr_list.append(reciprocal_rank(retrieved, rel))
    return float(np.mean(rr_list)) if rr_list else 0.0


def dcg(relevances: List[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain.
    
    Args:
        relevances: List of relevance scores in rank order
        k: Cutoff rank
        
    Returns:
        DCG score
    """
    rel_k = relevances[:k]
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_k))


def ndcg_at_k(retrieved: List[str], 
              relevant: List[str], 
              gains: Dict[str, float], 
              k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.
    
    Args:
        retrieved: List of retrieved document IDs in rank order
        relevant: List of relevant document IDs
        gains: Dict mapping doc_id to graded relevance score
        k: Cutoff rank
        
    Returns:
        nDCG@K score (0-1)
    """
    # Only relevant documents get their gain, others get 0
    rels = [gains.get(doc, 0.0) if doc in relevant else 0.0 for doc in retrieved]
    ideal_rels = sorted([gains.get(doc, 0.0) for doc in relevant], reverse=True)
    
    dcg_val = dcg(rels, k)
    idcg_val = dcg(ideal_rels, k)
    
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


class IRMetrics:
    """IR Evaluation Metrics Calculator."""
    
    def __init__(self, runs: Dict[str, List[str]], qrels: Dict[str, List[str]]):
        """Initialize with retrieval runs and ground truth.
        
        Args:
            runs: Dict of {query_id: [retrieved_doc_ids]}
            qrels: Dict of {query_id: [relevant_doc_ids]}
        """
        self.runs = runs
        self.qrels = qrels
    
    def evaluate(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate retrieval performance.
        
        Args:
            k_values: List of k values for P@k, R@k, nDCG@k
            
        Returns:
            Dict with metric scores
        """
        results = {
            'map': mean_average_precision(self.runs, self.qrels),
            'mrr': mean_reciprocal_rank(self.runs, self.qrels),
            'per_query': []
        }
        
        for qid, retrieved in self.runs.items():
            relevant = self.qrels.get(qid, [])
            query_metrics = {'query_id': qid}
            
            for k in k_values:
                query_metrics[f'p@{k}'] = precision_at_k(retrieved, relevant, k)
                query_metrics[f'r@{k}'] = recall_at_k(retrieved, relevant, k)
            
            query_metrics['ap'] = average_precision(retrieved, relevant)
            query_metrics['rr'] = reciprocal_rank(retrieved, relevant)
            
            results['per_query'].append(query_metrics)
        
        # Calculate mean metrics at each k
        for k in k_values:
            results[f'mean_p@{k}'] = np.mean([q[f'p@{k}'] for q in results['per_query']])
            results[f'mean_r@{k}'] = np.mean([q[f'r@{k}'] for q in results['per_query']])
        
        return results
