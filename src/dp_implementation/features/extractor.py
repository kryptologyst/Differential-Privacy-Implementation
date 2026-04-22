"""Feature extraction utilities for differential privacy experiments."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """Feature extraction utilities for differential privacy experiments.
    
    This class provides methods for extracting various types of features
    from different data modalities for use in privacy-preserving analysis.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize feature extractor.
        
        Args:
            random_seed: Random seed for reproducible feature extraction.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.scalers: Dict[str, StandardScaler] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
    
    def extract_statistical_features(
        self,
        data: np.ndarray,
        features: List[str] = None
    ) -> np.ndarray:
        """Extract statistical features from numerical data.
        
        Args:
            data: Input numerical data.
            features: List of statistical features to extract.
            
        Returns:
            Array of extracted features.
        """
        if features is None:
            features = ["mean", "std", "min", "max", "median", "q25", "q75"]
        
        extracted_features = []
        
        for feature in features:
            if feature == "mean":
                extracted_features.append(np.mean(data))
            elif feature == "std":
                extracted_features.append(np.std(data))
            elif feature == "min":
                extracted_features.append(np.min(data))
            elif feature == "max":
                extracted_features.append(np.max(data))
            elif feature == "median":
                extracted_features.append(np.median(data))
            elif feature == "q25":
                extracted_features.append(np.percentile(data, 25))
            elif feature == "q75":
                extracted_features.append(np.percentile(data, 75))
            elif feature == "skewness":
                extracted_features.append(self._compute_skewness(data))
            elif feature == "kurtosis":
                extracted_features.append(self._compute_kurtosis(data))
            else:
                raise ValueError(f"Unknown statistical feature: {feature}")
        
        return np.array(extracted_features)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def extract_histogram_features(
        self,
        data: np.ndarray,
        bins: int = 10,
        normalize: bool = True
    ) -> np.ndarray:
        """Extract histogram features from data.
        
        Args:
            data: Input data.
            bins: Number of histogram bins.
            normalize: Whether to normalize histogram counts.
            
        Returns:
            Array of histogram features.
        """
        hist, _ = np.histogram(data, bins=bins)
        
        if normalize:
            hist = hist / np.sum(hist)
        
        return hist
    
    def extract_text_features(
        self,
        texts: List[str],
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> np.ndarray:
        """Extract text features using TF-IDF.
        
        Args:
            texts: List of text documents.
            max_features: Maximum number of features.
            ngram_range: Range of n-grams to extract.
            
        Returns:
            Array of text features.
        """
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        features = vectorizer.fit_transform(texts).toarray()
        self.vectorizers['text'] = vectorizer
        
        return features
    
    def extract_temporal_features(
        self,
        timestamps: pd.Series,
        features: List[str] = None
    ) -> np.ndarray:
        """Extract temporal features from timestamps.
        
        Args:
            timestamps: Series of timestamps.
            features: List of temporal features to extract.
            
        Returns:
            Array of temporal features.
        """
        if features is None:
            features = ["hour", "day_of_week", "month", "is_weekend"]
        
        extracted_features = []
        
        for feature in features:
            if feature == "hour":
                extracted_features.append(timestamps.dt.hour.values)
            elif feature == "day_of_week":
                extracted_features.append(timestamps.dt.dayofweek.values)
            elif feature == "month":
                extracted_features.append(timestamps.dt.month.values)
            elif feature == "is_weekend":
                extracted_features.append((timestamps.dt.dayofweek >= 5).astype(int).values)
            elif feature == "day_of_year":
                extracted_features.append(timestamps.dt.dayofyear.values)
            elif feature == "quarter":
                extracted_features.append(timestamps.dt.quarter.values)
            else:
                raise ValueError(f"Unknown temporal feature: {feature}")
        
        return np.column_stack(extracted_features)
    
    def extract_geographical_features(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        features: List[str] = None
    ) -> np.ndarray:
        """Extract geographical features from coordinates.
        
        Args:
            latitudes: Array of latitude values.
            longitudes: Array of longitude values.
            features: List of geographical features to extract.
            
        Returns:
            Array of geographical features.
        """
        if features is None:
            features = ["distance_from_origin", "quadrant", "region"]
        
        extracted_features = []
        
        for feature in features:
            if feature == "distance_from_origin":
                # Distance from origin (0, 0)
                distances = np.sqrt(latitudes**2 + longitudes**2)
                extracted_features.append(distances)
            elif feature == "quadrant":
                # Geographical quadrant
                quadrants = np.zeros(len(latitudes))
                quadrants[(latitudes >= 0) & (longitudes >= 0)] = 1  # NE
                quadrants[(latitudes >= 0) & (longitudes < 0)] = 2   # NW
                quadrants[(latitudes < 0) & (longitudes < 0)] = 3    # SW
                quadrants[(latitudes < 0) & (longitudes >= 0)] = 4   # SE
                extracted_features.append(quadrants)
            elif feature == "region":
                # Simple region classification based on coordinates
                regions = np.zeros(len(latitudes))
                regions[(latitudes > 40) & (longitudes > -100)] = 1  # Northeast
                regions[(latitudes > 40) & (longitudes <= -100)] = 2 # Northwest
                regions[(latitudes <= 40) & (longitudes > -100)] = 3 # Southeast
                regions[(latitudes <= 40) & (longitudes <= -100)] = 4 # Southwest
                extracted_features.append(regions)
            else:
                raise ValueError(f"Unknown geographical feature: {feature}")
        
        return np.column_stack(extracted_features)
    
    def extract_network_features(
        self,
        data: pd.DataFrame,
        source_col: str,
        target_col: str,
        features: List[str] = None
    ) -> np.ndarray:
        """Extract network features from graph data.
        
        Args:
            data: DataFrame with network data.
            source_col: Column name for source nodes.
            target_col: Column name for target nodes.
            features: List of network features to extract.
            
        Returns:
            Array of network features.
        """
        if features is None:
            features = ["degree_centrality", "betweenness_centrality", "clustering_coefficient"]
        
        # Create adjacency matrix
        nodes = list(set(data[source_col].unique()) | set(data[target_col].unique()))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        n_nodes = len(nodes)
        adjacency_matrix = np.zeros((n_nodes, n_nodes))
        
        for _, row in data.iterrows():
            src_idx = node_to_idx[row[source_col]]
            tgt_idx = node_to_idx[row[target_col]]
            adjacency_matrix[src_idx, tgt_idx] = 1
            adjacency_matrix[tgt_idx, src_idx] = 1  # Undirected graph
        
        extracted_features = []
        
        for feature in features:
            if feature == "degree_centrality":
                degrees = np.sum(adjacency_matrix, axis=1)
                degree_centrality = degrees / (n_nodes - 1)
                extracted_features.append(degree_centrality)
            elif feature == "betweenness_centrality":
                # Simplified betweenness centrality
                betweenness = np.zeros(n_nodes)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            # Count shortest paths through node i
                            paths_through_i = 0
                            total_paths = 0
                            # Simplified calculation
                            if adjacency_matrix[i, j] == 1:
                                total_paths += 1
                                paths_through_i += 1
                            betweenness[i] += paths_through_i / max(total_paths, 1)
                extracted_features.append(betweenness)
            elif feature == "clustering_coefficient":
                clustering = np.zeros(n_nodes)
                for i in range(n_nodes):
                    neighbors = np.where(adjacency_matrix[i] == 1)[0]
                    if len(neighbors) >= 2:
                        edges_between_neighbors = 0
                        for j in neighbors:
                            for k in neighbors:
                                if j < k and adjacency_matrix[j, k] == 1:
                                    edges_between_neighbors += 1
                        max_possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                        clustering[i] = edges_between_neighbors / max_possible_edges
                extracted_features.append(clustering)
            else:
                raise ValueError(f"Unknown network feature: {feature}")
        
        return np.column_stack(extracted_features)
    
    def extract_sequence_features(
        self,
        sequences: List[np.ndarray],
        features: List[str] = None
    ) -> np.ndarray:
        """Extract features from sequences.
        
        Args:
            sequences: List of sequences.
            features: List of sequence features to extract.
            
        Returns:
            Array of sequence features.
        """
        if features is None:
            features = ["length", "mean", "std", "trend", "autocorrelation"]
        
        extracted_features = []
        
        for sequence in sequences:
            sequence_features = []
            
            for feature in features:
                if feature == "length":
                    sequence_features.append(len(sequence))
                elif feature == "mean":
                    sequence_features.append(np.mean(sequence))
                elif feature == "std":
                    sequence_features.append(np.std(sequence))
                elif feature == "trend":
                    # Linear trend coefficient
                    x = np.arange(len(sequence))
                    if len(sequence) > 1:
                        trend = np.polyfit(x, sequence, 1)[0]
                    else:
                        trend = 0.0
                    sequence_features.append(trend)
                elif feature == "autocorrelation":
                    # Lag-1 autocorrelation
                    if len(sequence) > 1:
                        autocorr = np.corrcoef(sequence[:-1], sequence[1:])[0, 1]
                        if np.isnan(autocorr):
                            autocorr = 0.0
                    else:
                        autocorr = 0.0
                    sequence_features.append(autocorr)
                else:
                    raise ValueError(f"Unknown sequence feature: {feature}")
            
            extracted_features.append(sequence_features)
        
        return np.array(extracted_features)
    
    def normalize_features(
        self,
        features: np.ndarray,
        method: str = "standard",
        feature_name: str = "default"
    ) -> np.ndarray:
        """Normalize extracted features.
        
        Args:
            features: Input features.
            method: Normalization method ("standard", "minmax").
            feature_name: Name for storing scaler.
            
        Returns:
            Normalized features.
        """
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_features = scaler.fit_transform(features)
        self.scalers[feature_name] = scaler
        
        return normalized_features
    
    def get_feature_importance(
        self,
        features: np.ndarray,
        target: np.ndarray,
        method: str = "mutual_info"
    ) -> np.ndarray:
        """Get feature importance scores.
        
        Args:
            features: Input features.
            target: Target values.
            method: Method for computing importance ("mutual_info", "f_score").
            
        Returns:
            Array of feature importance scores.
        """
        if method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression
            importance = mutual_info_regression(features, target)
        elif method == "f_score":
            from sklearn.feature_selection import f_regression
            _, importance = f_regression(features, target)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        return importance
    
    def select_features(
        self,
        features: np.ndarray,
        target: np.ndarray,
        k: int = 10,
        method: str = "mutual_info"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top-k features.
        
        Args:
            features: Input features.
            target: Target values.
            k: Number of features to select.
            method: Method for feature selection.
            
        Returns:
            Tuple of (selected_features, selected_indices).
        """
        importance = self.get_feature_importance(features, target, method)
        selected_indices = np.argsort(importance)[-k:]
        selected_features = features[:, selected_indices]
        
        return selected_features, selected_indices
