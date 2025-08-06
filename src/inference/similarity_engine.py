"""
FIXED Clause Similarity Engine
Enables semantic search and matching of classified contract clauses
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional, Union
import pickle
from tqdm import tqdm
import logging
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalBERTSimilarityEngine:
    """Semantic similarity engine for classified contract clauses"""
    
    def __init__(self, embeddings_file: str = None):
        """
        Initialize the powered similarity engine
        
        Args:
            embeddings_file: Path to sembeddings file
        """
        self.config = get_config()
        
        # Setup paths
        self.embeddings_dir = Path("data/embeddings")
        self.models_dir = self.config.paths.saved_models
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # LegalBERT models for new query classification
        self.binary_model = None
        self.binary_tokenizer = None
        self.multiclass_model = None
        self.multiclass_tokenizer = None
        self.label_encoder = None
        self.clause_types = []
        
        # Storage for embeddings and metadata
        self.clause_embeddings = {}
        self.clause_metadata = {}
        self.embedding_index = {}
        self.embedding_model = None
        
        # Similarity matrices
        self.similarity_matrix = None
        self.normalized_embeddings = None
        
        # Load embeddings if provided
        if embeddings_file:
            self.load_embeddings(embeddings_file)
    
    def load_legalbert_models(self):
        """Load trained LegalBERT models for query classification"""
        logger.info("Loading LegalBERT models for query processing...")
        
        try:
            # Load binary classifier
            binary_model_dir = self.models_dir / "binary_classifier"
            self.binary_tokenizer = AutoTokenizer.from_pretrained(binary_model_dir)
            self.binary_model = AutoModelForSequenceClassification.from_pretrained(binary_model_dir)
            self.binary_model.eval()
            
            # Load multiclass classifier
            multiclass_model_dir = self.models_dir / "multiclass_classifier"
            self.multiclass_tokenizer = AutoTokenizer.from_pretrained(multiclass_model_dir)
            self.multiclass_model = AutoModelForSequenceClassification.from_pretrained(multiclass_model_dir)
            self.multiclass_model.eval()
            
            # Load label encoder and clause types
            with open(multiclass_model_dir / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(multiclass_model_dir / "clause_types.json", 'r') as f:
                self.clause_types = json.load(f)
            
            logger.info("LegalBERT models loaded for query processing")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load LegalBERT models: {e}")
            return False
    
    def load_embeddings(self, filename: str):
        """Load LegalBERT embeddings and metadata from disk"""
        logger.info(f"Loading LegalBERT embeddings from: {filename}")
        
        # Handle different filename formats
        if not filename.endswith('.pkl'):
            filename = filename.replace('.pkl', '')
        
        # Load embeddings
        embeddings_file = self.embeddings_dir / f"{filename}.pkl"
        with open(embeddings_file, 'rb') as f:
            self.clause_embeddings = pickle.load(f)
        
        # Load metadata
        metadata_file = self.embeddings_dir / f"{filename}_metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.clause_metadata = json.load(f)
        
        # Load index
        index_file = self.embeddings_dir / f"{filename}_index.json"
        with open(index_file, 'r', encoding='utf-8') as f:
            self.embedding_index = json.load(f)
        
        # Initialize embedding model for new queries
        model_name = self.clause_embeddings.get('model_name', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load LegalBERT models for query processing
        self.load_legalbert_models()
        
        # Precompute normalized embeddings for faster cosine similarity
        self._precompute_normalized_embeddings()
        
        # Check if this is LegalBERT embeddings
        extraction_method = self.clause_embeddings.get('extraction_method', 'Unknown')
        logger.info(f"Loaded {self.clause_embeddings['num_clauses']} clause embeddings")
        logger.info(f"Extraction method: {extraction_method}")
        
        if extraction_method == 'LegalBERT_Pipeline':
            logger.info("Using LegalBERT-classified clause embeddings")
        else:
            logger.warning("Using baseline embeddings - consider regenerating with LegalBERT")
        
        return True
    
    def _precompute_normalized_embeddings(self):
        """Precompute normalized embeddings for faster cosine similarity"""
        logger.info("Precomputing normalized embeddings...")
        
        embeddings = self.clause_embeddings['embeddings']
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = embeddings / (norms + 1e-8)
        
        logger.info("Normalized embeddings computed")
    
    def classify_query_with_legalbert(self, query_text: str, threshold: float = 0.3) -> Tuple[bool, str, float]:
        """
        Classify a query using LegalBERT models
        
        Args:
            query_text: Query text to classify
            threshold: Binary classification threshold
            
        Returns:
            has_answer: Whether query is relevant
            predicted_clause_type: Predicted clause type (if relevant)
            confidence: Classification confidence
        """
        if not self.binary_model or not self.multiclass_model:
            return True, "unknown", 1.0  # Fallback if models not loaded
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.binary_model.to(device)
        self.multiclass_model.to(device)
        
        with torch.no_grad():
            # Binary classification
            encoding = self.binary_tokenizer(
                query_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = self.binary_model(**encoding)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            has_answer_prob = probs[0][1].item()
            has_answer = has_answer_prob > threshold
            
            if has_answer:
                # Multiclass classification
                outputs = self.multiclass_model(**encoding)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                confidence = torch.nn.functional.softmax(outputs.logits, dim=-1).max().item()
                
                predicted_clause_type = self.clause_types[prediction] if prediction < len(self.clause_types) else 'unknown'
                
                return True, predicted_clause_type, confidence
            else:
                return False, "not_relevant", has_answer_prob
    
    def compute_similarity_matrix(self, save_matrix: bool = True):
        """Compute full similarity matrix between all  clauses"""
        logger.info("Computing similarity matrix...")
        
        if self.normalized_embeddings is None:
            self._precompute_normalized_embeddings()
        
        # Compute cosine similarity matrix
        self.similarity_matrix = np.dot(self.normalized_embeddings, self.normalized_embeddings.T)
        
        # Save matrix if requested
        if save_matrix:
            matrix_file = self.results_dir / "legalbert_similarity_matrix.npy"
            np.save(matrix_file, self.similarity_matrix)
            logger.info(f"Similarity matrix saved to: {matrix_file}")
        
        logger.info(f"Similarity matrix computed: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def find_similar_clauses(self, 
                           query_text: str, 
                           top_k: int = 10,
                           clause_type_filter: Optional[str] = None,
                           min_similarity: float = 0.0,
                           use_legalbert_classification: bool = True) -> List[Dict]:
        """
        Find similar clauses to a query text using LegalBERT pipeline
        
        Args:
            query_text: Text to find similar clauses for
            top_k: Number of top results to return
            clause_type_filter: Filter by specific clause type
            min_similarity: Minimum similarity threshold
            use_legalbert_classification: Whether to use LegalBERT for query classification
            
        Returns:
            List of similar clause dictionaries with similarity scores
        """
        logger.info(f"Finding similar clauses for: '{query_text[:50]}...'")
        
        # Classify query with LegalBERT if available
        query_classification = None
        if use_legalbert_classification and self.binary_model:
            has_answer, predicted_type, confidence = self.classify_query_with_legalbert(query_text)
            query_classification = {
                'has_answer': has_answer,
                'predicted_clause_type': predicted_type,
                'confidence': confidence
            }
            
            if not has_answer:
                logger.warning(f"LegalBERT classified query as not relevant (confidence: {confidence:.3f})")
                # Still proceed but with warning
        
        # Encode query text
        query_embedding = self.embedding_model.encode([query_text])
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(self.normalized_embeddings, query_embedding.T).flatten()
        
        # Get candidate indices
        candidate_indices = np.arange(len(similarities))
        
        # Apply clause type filter if specified
        if clause_type_filter:
            # Use predicted clause types if available (LegalBERT embeddings)
            type_key = 'predicted_clause_type_to_indices'
            if type_key in self.embedding_index:
                type_indices = self.embedding_index[type_key].get(clause_type_filter, [])
            else:
                # Fallback to ground truth types
                type_indices = self.embedding_index.get('clause_type_to_indices', {}).get(clause_type_filter, [])
            
            candidate_indices = [i for i in candidate_indices if i in type_indices]
            similarities = similarities[candidate_indices]
        
        # Apply minimum similarity filter
        valid_mask = similarities >= min_similarity
        candidate_indices = [candidate_indices[i] for i, valid in enumerate(valid_mask) if valid]
        similarities = similarities[valid_mask]
        
        # Get top k results
        if len(similarities) > 0:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                clause_idx = candidate_indices[idx] if clause_type_filter else idx
                clause_data = self.clause_metadata[str(clause_idx)].copy()
                clause_data['similarity_score'] = float(similarities[idx])
                clause_data['rank'] = i + 1
                
                # Add query classification info
                if query_classification:
                    clause_data['query_classification'] = query_classification
                
                results.append(clause_data)
            
            # Log results summary
            if query_classification:
                logger.info(f"Query classified as: {query_classification['predicted_clause_type']} "
                          f"(confidence: {query_classification['confidence']:.3f})")
            
            return results
        
        return []
    
    def find_similar_clauses_by_id(self, 
                                  clause_id: str, 
                                  top_k: int = 10,
                                  exclude_same_contract: bool = True,
                                  prefer_same_type: bool = True) -> List[Dict]:
        """
        Find similar clauses to a specific LegalBERT-classified clause by ID
        
        Args:
            clause_id: ID of the clause to find similar clauses for
            top_k: Number of top results to return
            exclude_same_contract: Whether to exclude clauses from same contract
            prefer_same_type: Whether to prefer same predicted clause type
            
        Returns:
            List of similar clause dictionaries with similarity scores
        """
        logger.info(f"Finding similar clauses for clause ID: {clause_id}")
        
        # Get clause index
        clause_idx = self.embedding_index['clause_id_to_index'].get(clause_id)
        if clause_idx is None:
            logger.error(f"Clause ID not found: {clause_id}")
            return []
        
        # Get similarities from precomputed matrix
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        similarities = self.similarity_matrix[clause_idx]
        
        # Get source clause info for filtering
        source_clause = self.clause_metadata[str(clause_idx)]
        source_contract_id = source_clause['original_id']
        source_clause_type = source_clause.get('predicted_clause_type', 
                                             source_clause.get('clause_type', 'unknown'))
        
        # Get candidate indices (exclude self)
        candidate_indices = [i for i in range(len(similarities)) if i != clause_idx]
        
        # Exclude same contract if requested
        if exclude_same_contract:
            candidate_indices = [
                i for i in candidate_indices 
                if self.clause_metadata[str(i)]['original_id'] != source_contract_id
            ]
        
        # Boost similar clause types if requested
        if prefer_same_type and source_clause_type != 'unknown':
            for i in candidate_indices:
                candidate_clause = self.clause_metadata[str(i)]
                candidate_type = candidate_clause.get('predicted_clause_type',
                                                    candidate_clause.get('clause_type', 'unknown'))
                if candidate_type == source_clause_type:
                    similarities[i] *= 1.1  # Small boost for same type
        
        # Get similarities for candidates
        candidate_similarities = similarities[candidate_indices]
        
        # Get top k results
        if len(candidate_similarities) > 0:
            top_indices = np.argsort(candidate_similarities)[::-1][:top_k]
            
            results = []
            for i, idx in enumerate(top_indices):
                clause_idx = candidate_indices[idx]
                clause_data = self.clause_metadata[str(clause_idx)].copy()
                clause_data['similarity_score'] = float(candidate_similarities[idx])
                clause_data['rank'] = i + 1
                
                # Add source clause info for comparison
                clause_data['source_clause_type'] = source_clause_type
                clause_data['type_match'] = (clause_data.get('predicted_clause_type', 
                                                           clause_data.get('clause_type', 'unknown')) == source_clause_type)
                
                results.append(clause_data)
            
            return results
        
        return []
    
    def get_clause_clusters(self, 
                          n_clusters: int = 10, 
                          clause_type_filter: Optional[str] = None,
                          use_predicted_types: bool = True) -> Dict:
        """
        Cluster LegalBERT-classified clauses based on semantic similarity
        
        Args:
            n_clusters: Number of clusters to create
            clause_type_filter: Filter by specific clause type
            use_predicted_types: Whether to use LegalBERT predicted types
            
        Returns:
            Dictionary with cluster assignments and statistics
        """
        logger.info(f"Clustering LegalBERT clauses into {n_clusters} clusters")
        
        # Get embeddings to cluster
        embeddings = self.clause_embeddings['embeddings']
        clause_indices = list(range(len(embeddings)))
        
        # Apply clause type filter if specified
        if clause_type_filter:
            if use_predicted_types and 'predicted_clause_type_to_indices' in self.embedding_index:
                type_indices = self.embedding_index['predicted_clause_type_to_indices'].get(clause_type_filter, [])
            else:
                type_indices = self.embedding_index.get('clause_type_to_indices', {}).get(clause_type_filter, [])
            
            clause_indices = [i for i in clause_indices if i in type_indices]
            embeddings = embeddings[clause_indices]
        
        if len(embeddings) < n_clusters:
            logger.warning(f"Not enough clauses ({len(embeddings)}) for {n_clusters} clusters")
            n_clusters = max(1, len(embeddings) // 2)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize results
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            original_idx = clause_indices[i]
            clause_data = self.clause_metadata[str(original_idx)].copy()
            clause_data['cluster_id'] = int(cluster_id)
            clusters[cluster_id].append(clause_data)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, clauses in clusters.items():
            # Use predicted clause types if available
            if use_predicted_types and 'predicted_clause_type' in clauses[0]:
                clause_types = [clause.get('predicted_clause_type', 'unknown') for clause in clauses]
            else:
                clause_types = [clause.get('clause_type', 'unknown') for clause in clauses]
            
            cluster_stats[cluster_id] = {
                'size': len(clauses),
                'clause_types': list(set(clause_types)),
                'most_common_type': max(set(clause_types), key=clause_types.count),
                'diversity': len(set(clause_types)) / len(clause_types) if clause_types else 0,
                'uses_predicted_types': use_predicted_types
            }
        
        results = {
            'clusters': clusters,
            'cluster_stats': cluster_stats,
            'n_clusters': n_clusters,
            'total_clauses': len(clause_indices),
            'inertia': float(kmeans.inertia_),
            'cluster_centers': kmeans.cluster_centers_,
            'legalbert_classification': use_predicted_types
        }
        
        return results
    
    def analyze_legalbert_performance(self) -> Dict:
        """Analyze LegalBERT classification performance in embeddings"""
        if 'predicted_clause_type_to_indices' not in self.embedding_index:
            return {'error': 'No LegalBERT predictions found in embeddings'}
        
        logger.info("Analyzing LegalBERT classification performance...")
        
        # Compare predicted vs ground truth
        agreements = 0
        total = 0
        type_accuracies = {}
        
        for idx_str, clause_data in self.clause_metadata.items():
            predicted = clause_data.get('predicted_clause_type', 'unknown')
            ground_truth = clause_data.get('ground_truth_clause_type', 'unknown')
            
            if predicted != 'unknown' and ground_truth != 'unknown':
                total += 1
                if predicted == ground_truth:
                    agreements += 1
                
                # Track per-type accuracy
                if ground_truth not in type_accuracies:
                    type_accuracies[ground_truth] = {'correct': 0, 'total': 0}
                
                type_accuracies[ground_truth]['total'] += 1
                if predicted == ground_truth:
                    type_accuracies[ground_truth]['correct'] += 1
        
        # Calculate accuracies
        overall_accuracy = agreements / total if total > 0 else 0
        per_type_accuracy = {}
        for clause_type, counts in type_accuracies.items():
            per_type_accuracy[clause_type] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_classified': total,
            'correct_predictions': agreements,
            'per_type_accuracy': per_type_accuracy,
            'average_type_accuracy': np.mean(list(per_type_accuracy.values())) if per_type_accuracy else 0
        }
    
    def export_similarity_results(self, 
                                query_text: str, 
                                results: List[Dict], 
                                filename: str = None) -> str:
        """Export LegalBERT similarity search results to CSV"""
        if filename is None:
            filename = f"legalbert_similarity_results_{len(results)}.csv"
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Add query info
        df['query_text'] = query_text
        
        # Reorder columns for LegalBERT results
        column_order = ['rank', 'similarity_score', 'predicted_clause_type', 'ground_truth_clause_type',
                       'question', 'answer_text', 'title', 'clause_id', 'original_id', 'query_text']
        df = df[[col for col in column_order if col in df.columns]]
        
        # Save to CSV
        output_file = self.results_dir / filename
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"LegalBERT results exported to: {output_file}")
        return str(output_file)
    
    def get_engine_stats(self) -> Dict:
        """Get comprehensive statistics about the LegalBERT similarity engine"""
        if not self.clause_embeddings:
            return {}
        
        stats = {
            'extraction_method': self.clause_embeddings.get('extraction_method', 'Unknown'),
            'total_clauses': self.clause_embeddings['num_clauses'],
            'embedding_dimension': self.clause_embeddings['embedding_dim'],
            'embedding_model': self.clause_embeddings['model_name'],
            'classification_pipeline': self.clause_embeddings.get('pipeline_version', '1.0'),
            'binary_threshold': self.clause_embeddings.get('binary_threshold', 'N/A'),
            'multiclass_accuracy': self.clause_embeddings.get('multiclass_accuracy', 'N/A'),
            'similarity_matrix_computed': self.similarity_matrix is not None,
            'legalbert_models_loaded': self.binary_model is not None,
        }
        
        # Clause type statistics
        if 'predicted_clause_type_to_indices' in self.embedding_index:
            stats['predicted_clause_types'] = len(self.embedding_index['predicted_clause_type_to_indices'])
            stats['clause_type_distribution'] = {
                clause_type: len(indices) 
                for clause_type, indices in self.embedding_index['predicted_clause_type_to_indices'].items()
            }
        
        if 'ground_truth_clause_type_to_indices' in self.embedding_index:
            stats['ground_truth_clause_types'] = len(self.embedding_index['ground_truth_clause_type_to_indices'])
        
        stats['data_splits'] = len(self.embedding_index.get('split_to_indices', {}))
        
        # Similarity statistics
        if self.similarity_matrix is not None:
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
            similarities = self.similarity_matrix[mask]
            
            stats['similarity_stats'] = {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'high_similarity_pairs': int(np.sum(similarities > 0.8))
            }
        
        # LegalBERT performance analysis
        legalbert_performance = self.analyze_legalbert_performance()
        if 'error' not in legalbert_performance:
            stats['legalbert_performance'] = legalbert_performance
        
        return stats

def main():
    """Main LegalBERT similarity engine demo"""
    print("LegalBERT Clause Similarity Engine")
    print("Using LegalBERT-classified clause embeddings")
    print("=" * 50)
    
    # Initialize engine
    engine = LegalBERTSimilarityEngine()
    
    # Try to load LegalBERT embeddings first, fallback to baseline
    embeddings_files = [
        "legalbert_clause_embeddings_all-MiniLM-L6-v2",  # LegalBERT embeddings
        "clause_embeddings_all-MiniLM-L6-v2"  # Baseline embeddings
    ]
    
    loaded = False
    for embeddings_file in embeddings_files:
        try:
            engine.load_embeddings(embeddings_file)
            loaded = True
            break
        except FileNotFoundError:
            continue
    
    if not loaded:
        print("No embeddings found. Please run embedding generation first.")
        return
    
    # Demo 1: LegalBERT-powered text similarity search
    print("\n1.Text Similarity Search")
    print("-" * 35)
    
    query_text = "The party shall indemnify and hold harmless the other party"
    results = engine.find_similar_clauses(query_text, top_k=5, use_legalbert_classification=True)
    
    print(f"Query: '{query_text}'")
    print(f"Found {len(results)} similar clauses:")
    
    for result in results:
        predicted_type = result.get('predicted_clause_type', result.get('clause_type', 'unknown'))
        print(f"  {result['rank']}. [{predicted_type}] "
              f"Similarity: {result['similarity_score']:.3f}")
        print(f"     Answer: {result['answer_text'][:100]}...")
        
        # Show query classification if available
        if 'query_classification' in result:
            qc = result['query_classification']
            print(f"     Query classified as: {qc['predicted_clause_type']} "
                  f"(confidence: {qc['confidence']:.3f})")
        print()
    
    # Demo 2: LegalBERT clause-to-clause similarity
    print("\n2.Clause-to-Clause Similarity")
    print("-" * 40)
    
    # Get a random clause ID
    clause_ids = list(engine.embedding_index['clause_id_to_index'].keys())
    if clause_ids:
        sample_clause_id = clause_ids[0]
        similar_clauses = engine.find_similar_clauses_by_id(sample_clause_id, top_k=3, prefer_same_type=True)
        
        print(f"Finding clauses similar to: {sample_clause_id}")
        for result in similar_clauses:
            predicted_type = result.get('predicted_clause_type', result.get('clause_type', 'unknown'))
            type_match = result.get('type_match', False)
            match_indicator = "✓" if type_match else " "
            
            print(f"  {result['rank']}. [{predicted_type}] {match_indicator} "
                  f"Similarity: {result['similarity_score']:.3f}")
            print(f"     Answer: {result['answer_text'][:100]}...")
            print()
    
    # Demo 3: clustering analysis
    print("\n3.  Clause Clustering")
    print("-" * 30)
    
    # Check if we have predicted clause types
    if 'predicted_clause_type_to_indices' in engine.embedding_index:
        clause_types = list(engine.embedding_index['predicted_clause_type_to_indices'].keys())
        if clause_types:
            sample_type = clause_types[0]
            cluster_results = engine.get_clause_clusters(n_clusters=3, 
                                                       clause_type_filter=sample_type, 
                                                       use_predicted_types=True)
            
            print(f"Clustering '{sample_type}' clauses (LegalBERT predictions):")
            for cluster_id, stats in cluster_results['cluster_stats'].items():
                print(f"  Cluster {cluster_id}: {stats['size']} clauses")
                print(f"    Most common type: {stats['most_common_type']}")
                print(f"    Diversity: {stats['diversity']:.2f}")
                print()
    
    # Demo 4: LegalBERT engine statistics
    print("\n4. Engine Statistics")
    print("-" * 30)
    
    stats = engine.get_engine_stats()
    print(f"Extraction Method: {stats.get('extraction_method', 'Unknown')}")
    print(f"Total clauses: {stats['total_clauses']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Embedding model: {stats['embedding_model']}")
    print(f"LegalBERT models loaded: {stats['legalbert_models_loaded']}")
    
    if 'binary_threshold' in stats:
        print(f"Binary threshold: {stats['binary_threshold']}")
        print(f"Multiclass accuracy: {stats['multiclass_accuracy']}")
    
    if 'predicted_clause_types' in stats:
        print(f"LegalBERT predicted clause types: {stats['predicted_clause_types']}")
    
    # Compute similarity matrix for stats
    if not stats['similarity_matrix_computed']:
        print("\nComputing similarity matrix...")
        engine.compute_similarity_matrix()
        stats = engine.get_engine_stats()
    
    if 'similarity_stats' in stats:
        sim_stats = stats['similarity_stats']
        print(f"\nSimilarity Statistics:")
        print(f"  Mean similarity: {sim_stats['mean_similarity']:.3f}")
        print(f"  Std similarity: {sim_stats['std_similarity']:.3f}")
        print(f"  High similarity pairs (>0.8): {sim_stats['high_similarity_pairs']}")
    
    # LegalBERT performance analysis
    if 'legalbert_performance' in stats:
        perf = stats['legalbert_performance']
        print(f"\nLegalBERT Classification Performance:")
        print(f"  Overall accuracy: {perf['overall_accuracy']:.1%}")
        print(f"  Total classified: {perf['total_classified']}")
        print(f"  Average per-type accuracy: {perf['average_type_accuracy']:.1%}")
    
    print("\nLegalBERT similarity engine complete!")

if __name__ == "__main__":
    main()