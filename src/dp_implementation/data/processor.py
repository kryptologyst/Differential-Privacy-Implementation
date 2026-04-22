"""Data processing utilities for differential privacy experiments."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class DataProcessor:
    """Data processor for preparing data for differential privacy experiments.
    
    This class provides utilities for data preprocessing, anonymization,
    and preparation for differential privacy mechanisms.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize data processor.
        
        Args:
            random_seed: Random seed for reproducible processing.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
    
    def anonymize_identifiers(
        self,
        data: pd.DataFrame,
        id_columns: List[str],
        method: str = "hash"
    ) -> pd.DataFrame:
        """Anonymize identifier columns.
        
        Args:
            data: Input DataFrame.
            id_columns: List of column names to anonymize.
            method: Anonymization method ("hash", "random", "k_anonymity").
            
        Returns:
            DataFrame with anonymized identifiers.
        """
        data_processed = data.copy()
        
        for col in id_columns:
            if col in data_processed.columns:
                if method == "hash":
                    data_processed[col] = data_processed[col].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8]
                    )
                elif method == "random":
                    unique_values = data_processed[col].unique()
                    mapping = {val: f"ID_{i}" for i, val in enumerate(unique_values)}
                    data_processed[col] = data_processed[col].map(mapping)
                elif method == "k_anonymity":
                    # Simple k-anonymity by grouping similar values
                    data_processed[col] = self._apply_k_anonymity(data_processed[col], k=5)
        
        return data_processed
    
    def _apply_k_anonymity(self, series: pd.Series, k: int = 5) -> pd.Series:
        """Apply k-anonymity to a series.
        
        Args:
            series: Input series.
            k: Minimum group size for k-anonymity.
            
        Returns:
            Series with k-anonymity applied.
        """
        # Simple implementation: group by value and ensure minimum group size
        value_counts = series.value_counts()
        valid_values = value_counts[value_counts >= k].index
        
        # Replace values that don't meet k-anonymity with "OTHER"
        result = series.copy()
        result[~result.isin(valid_values)] = "OTHER"
        
        return result
    
    def remove_pii(self, data: pd.DataFrame, pii_columns: List[str]) -> pd.DataFrame:
        """Remove or obfuscate PII columns.
        
        Args:
            data: Input DataFrame.
            pii_columns: List of PII column names to remove.
            
        Returns:
            DataFrame with PII removed.
        """
        data_processed = data.copy()
        
        for col in pii_columns:
            if col in data_processed.columns:
                # Remove the column entirely
                data_processed = data_processed.drop(columns=[col])
        
        return data_processed
    
    def hash_sensitive_fields(
        self,
        data: pd.DataFrame,
        sensitive_columns: List[str]
    ) -> pd.DataFrame:
        """Hash sensitive fields for privacy protection.
        
        Args:
            data: Input DataFrame.
            sensitive_columns: List of sensitive column names to hash.
            
        Returns:
            DataFrame with sensitive fields hashed.
        """
        data_processed = data.copy()
        
        for col in sensitive_columns:
            if col in data_processed.columns:
                data_processed[col] = data_processed[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()
                )
        
        return data_processed
    
    def normalize_features(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: str = "standard",
        columns: Optional[List[str]] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Normalize features.
        
        Args:
            data: Input data.
            method: Normalization method ("standard", "minmax").
            columns: List of columns to normalize (for DataFrame).
            
        Returns:
            Normalized data.
        """
        if isinstance(data, pd.DataFrame):
            return self._normalize_dataframe(data, method, columns)
        else:
            return self._normalize_array(data, method)
    
    def _normalize_dataframe(
        self,
        data: pd.DataFrame,
        method: str,
        columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Normalize DataFrame columns.
        
        Args:
            data: Input DataFrame.
            method: Normalization method.
            columns: List of columns to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        data_processed = data.copy()
        
        if columns is None:
            columns = data_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in data_processed.columns:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
                
                data_processed[col] = scaler.fit_transform(data_processed[[col]]).flatten()
                self.scalers[col] = scaler
        
        return data_processed
    
    def _normalize_array(self, data: np.ndarray, method: str) -> np.ndarray:
        """Normalize numpy array.
        
        Args:
            data: Input array.
            method: Normalization method.
            
        Returns:
            Normalized array.
        """
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return scaler.fit_transform(data)
    
    def encode_categorical(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "label"
    ) -> pd.DataFrame:
        """Encode categorical variables.
        
        Args:
            data: Input DataFrame.
            columns: List of categorical columns to encode.
            method: Encoding method ("label", "onehot").
            
        Returns:
            DataFrame with encoded categorical variables.
        """
        data_processed = data.copy()
        
        if columns is None:
            columns = data_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if col in data_processed.columns:
                if method == "label":
                    encoder = LabelEncoder()
                    data_processed[col] = encoder.fit_transform(data_processed[col])
                    self.encoders[col] = encoder
                elif method == "onehot":
                    # One-hot encoding
                    dummies = pd.get_dummies(data_processed[col], prefix=col)
                    data_processed = pd.concat([data_processed, dummies], axis=1)
                    data_processed = data_processed.drop(columns=[col])
        
        return data_processed
    
    def add_noise(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        noise_level: float = 0.1,
        noise_type: str = "gaussian"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Add noise to data for privacy protection.
        
        Args:
            data: Input data.
            noise_level: Level of noise to add.
            noise_type: Type of noise ("gaussian", "laplace").
            
        Returns:
            Data with added noise.
        """
        if isinstance(data, pd.DataFrame):
            return self._add_noise_dataframe(data, noise_level, noise_type)
        else:
            return self._add_noise_array(data, noise_level, noise_type)
    
    def _add_noise_dataframe(
        self,
        data: pd.DataFrame,
        noise_level: float,
        noise_type: str
    ) -> pd.DataFrame:
        """Add noise to DataFrame.
        
        Args:
            data: Input DataFrame.
            noise_level: Level of noise to add.
            noise_type: Type of noise.
            
        Returns:
            DataFrame with added noise.
        """
        data_processed = data.copy()
        numeric_columns = data_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if noise_type == "gaussian":
                noise = np.random.normal(0, noise_level, len(data_processed))
            elif noise_type == "laplace":
                noise = np.random.laplace(0, noise_level, len(data_processed))
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            data_processed[col] = data_processed[col] + noise
        
        return data_processed
    
    def _add_noise_array(
        self,
        data: np.ndarray,
        noise_level: float,
        noise_type: str
    ) -> np.ndarray:
        """Add noise to numpy array.
        
        Args:
            data: Input array.
            noise_level: Level of noise to add.
            noise_type: Type of noise.
            
        Returns:
            Array with added noise.
        """
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, data.shape)
        elif noise_type == "laplace":
            noise = np.random.laplace(0, noise_level, data.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return data + noise
    
    def create_entity_aware_split(
        self,
        data: pd.DataFrame,
        entity_column: str,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create entity-aware train-validation-test split.
        
        Args:
            data: Input DataFrame.
            entity_column: Column containing entity identifiers.
            test_ratio: Proportion for testing.
            val_ratio: Proportion for validation.
            
        Returns:
            Tuple of (train_data, val_data, test_data).
        """
        entities = data[entity_column].unique()
        n_entities = len(entities)
        
        # Split entities
        test_entities = np.random.choice(
            entities, size=int(n_entities * test_ratio), replace=False
        )
        remaining_entities = np.setdiff1d(entities, test_entities)
        val_entities = np.random.choice(
            remaining_entities, size=int(n_entities * val_ratio), replace=False
        )
        train_entities = np.setdiff1d(remaining_entities, val_entities)
        
        # Split data based on entities
        train_data = data[data[entity_column].isin(train_entities)]
        val_data = data[data[entity_column].isin(val_entities)]
        test_data = data[data[entity_column].isin(test_entities)]
        
        return train_data, val_data, test_data
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the data.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Dictionary with data summary.
        """
        summary = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "numeric_summary": data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
            "categorical_summary": {}
        }
        
        # Categorical summary
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            summary["categorical_summary"][col] = {
                "unique_values": data[col].nunique(),
                "most_common": data[col].value_counts().head().to_dict()
            }
        
        return summary
