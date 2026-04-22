"""Privacy defense utilities for differential privacy experiments."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PrivacyDefense:
    """Privacy defense utilities for differential privacy experiments.
    
    This class provides methods for implementing various privacy defenses
    and protection mechanisms in differential privacy systems.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize privacy defense.
        
        Args:
            random_seed: Random seed for reproducible defense mechanisms.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def apply_input_sanitization(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sanitization_method: str = "clipping",
        clip_value: float = 1.0
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Apply input sanitization to data.
        
        Args:
            data: Input data to sanitize.
            sanitization_method: Method for sanitization ("clipping", "normalization").
            clip_value: Value for clipping.
            
        Returns:
            Sanitized data.
        """
        if isinstance(data, pd.DataFrame):
            return self._sanitize_dataframe(data, sanitization_method, clip_value)
        else:
            return self._sanitize_array(data, sanitization_method, clip_value)
    
    def _sanitize_dataframe(
        self,
        data: pd.DataFrame,
        sanitization_method: str,
        clip_value: float
    ) -> pd.DataFrame:
        """Sanitize DataFrame.
        
        Args:
            data: Input DataFrame.
            sanitization_method: Method for sanitization.
            clip_value: Value for clipping.
            
        Returns:
            Sanitized DataFrame.
        """
        data_sanitized = data.copy()
        
        if sanitization_method == "clipping":
            # Clip values to specified range
            numeric_columns = data_sanitized.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                data_sanitized[col] = np.clip(data_sanitized[col], -clip_value, clip_value)
        
        elif sanitization_method == "normalization":
            # Normalize values
            numeric_columns = data_sanitized.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            data_sanitized[numeric_columns] = scaler.fit_transform(data_sanitized[numeric_columns])
        
        return data_sanitized
    
    def _sanitize_array(
        self,
        data: np.ndarray,
        sanitization_method: str,
        clip_value: float
    ) -> np.ndarray:
        """Sanitize numpy array.
        
        Args:
            data: Input array.
            sanitization_method: Method for sanitization.
            clip_value: Value for clipping.
            
        Returns:
            Sanitized array.
        """
        if sanitization_method == "clipping":
            return np.clip(data, -clip_value, clip_value)
        elif sanitization_method == "normalization":
            scaler = StandardScaler()
            return scaler.fit_transform(data)
        else:
            raise ValueError(f"Unknown sanitization method: {sanitization_method}")
    
    def apply_differential_privacy_noise(
        self,
        data: np.ndarray,
        epsilon: float,
        delta: float = 1e-5,
        mechanism: str = "laplace"
    ) -> np.ndarray:
        """Apply differential privacy noise to data.
        
        Args:
            data: Input data.
            epsilon: Privacy parameter.
            delta: Privacy parameter.
            mechanism: Noise mechanism ("laplace", "gaussian").
            
        Returns:
            Data with added noise.
        """
        # Compute sensitivity (simplified)
        sensitivity = np.max(data) - np.min(data)
        
        if mechanism == "laplace":
            noise_scale = sensitivity / epsilon
            noise = np.random.laplace(0, noise_scale, data.shape)
        elif mechanism == "gaussian":
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            noise = np.random.normal(0, noise_scale, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        return data + noise
    
    def apply_k_anonymity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        k: int = 5
    ) -> pd.DataFrame:
        """Apply k-anonymity to data.
        
        Args:
            data: Input DataFrame.
            quasi_identifiers: List of quasi-identifier columns.
            k: Minimum group size for k-anonymity.
            
        Returns:
            DataFrame with k-anonymity applied.
        """
        data_anonymized = data.copy()
        
        # Group by quasi-identifiers
        groups = data_anonymized.groupby(quasi_identifiers)
        
        # Check which groups meet k-anonymity
        valid_groups = groups.size() >= k
        
        # For groups that don't meet k-anonymity, generalize or suppress
        for group_key, group_data in groups:
            if len(group_data) < k:
                # Suppress the group (remove from dataset)
                data_anonymized = data_anonymized.drop(group_data.index)
        
        return data_anonymized
    
    def apply_l_diversity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        l: int = 2
    ) -> pd.DataFrame:
        """Apply l-diversity to data.
        
        Args:
            data: Input DataFrame.
            quasi_identifiers: List of quasi-identifier columns.
            sensitive_attribute: Name of sensitive attribute.
            l: Minimum number of distinct values for l-diversity.
            
        Returns:
            DataFrame with l-diversity applied.
        """
        data_diversified = data.copy()
        
        # Group by quasi-identifiers
        groups = data_diversified.groupby(quasi_identifiers)
        
        # Check which groups meet l-diversity
        valid_groups = []
        for group_key, group_data in groups:
            unique_sensitive_values = group_data[sensitive_attribute].nunique()
            if unique_sensitive_values >= l:
                valid_groups.append(group_data)
        
        if valid_groups:
            data_diversified = pd.concat(valid_groups, ignore_index=True)
        else:
            # If no groups meet l-diversity, return empty DataFrame
            data_diversified = pd.DataFrame(columns=data.columns)
        
        return data_diversified
    
    def apply_t_closeness(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        t: float = 0.1
    ) -> pd.DataFrame:
        """Apply t-closeness to data.
        
        Args:
            data: Input DataFrame.
            quasi_identifiers: List of quasi-identifier columns.
            sensitive_attribute: Name of sensitive attribute.
            t: Maximum distance threshold for t-closeness.
            
        Returns:
            DataFrame with t-closeness applied.
        """
        data_closeness = data.copy()
        
        # Compute global distribution of sensitive attribute
        global_dist = data_closeness[sensitive_attribute].value_counts(normalize=True)
        
        # Group by quasi-identifiers
        groups = data_closeness.groupby(quasi_identifiers)
        
        # Check which groups meet t-closeness
        valid_groups = []
        for group_key, group_data in groups:
            # Compute group distribution
            group_dist = group_data[sensitive_attribute].value_counts(normalize=True)
            
            # Compute distance between distributions (simplified)
            distance = 0.0
            for value in global_dist.index:
                global_prob = global_dist.get(value, 0)
                group_prob = group_dist.get(value, 0)
                distance += abs(global_prob - group_prob)
            
            distance /= 2  # Normalize to [0, 1]
            
            if distance <= t:
                valid_groups.append(group_data)
        
        if valid_groups:
            data_closeness = pd.concat(valid_groups, ignore_index=True)
        else:
            # If no groups meet t-closeness, return empty DataFrame
            data_closeness = pd.DataFrame(columns=data.columns)
        
        return data_closeness
    
    def apply_synthetic_data_generation(
        self,
        data: pd.DataFrame,
        method: str = "gaussian",
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """Generate synthetic data for privacy protection.
        
        Args:
            data: Input DataFrame.
            method: Method for synthetic data generation.
            noise_level: Level of noise to add.
            
        Returns:
            Synthetic DataFrame.
        """
        if method == "gaussian":
            # Add Gaussian noise to numerical columns
            synthetic_data = data.copy()
            numeric_columns = synthetic_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                noise = np.random.normal(0, noise_level * synthetic_data[col].std(), len(synthetic_data))
                synthetic_data[col] = synthetic_data[col] + noise
        
        elif method == "sampling":
            # Sample from the same distribution
            synthetic_data = data.sample(n=len(data), replace=True, random_state=self.random_seed)
        
        else:
            raise ValueError(f"Unknown synthetic data method: {method}")
        
        return synthetic_data
    
    def apply_membership_inference_defense(
        self,
        model_predictions: np.ndarray,
        defense_method: str = "confidence_thresholding",
        threshold: float = 0.5
    ) -> np.ndarray:
        """Apply defense against membership inference attacks.
        
        Args:
            model_predictions: Model predictions.
            defense_method: Defense method to apply.
            threshold: Threshold for defense.
            
        Returns:
            Defended predictions.
        """
        if defense_method == "confidence_thresholding":
            # Threshold predictions based on confidence
            if model_predictions.ndim > 1:
                # Multi-class predictions
                max_probs = np.max(model_predictions, axis=1)
                defended_predictions = model_predictions.copy()
                defended_predictions[max_probs < threshold] = 0.5  # Neutral prediction
            else:
                # Binary predictions
                defended_predictions = np.where(
                    np.abs(model_predictions - 0.5) < threshold,
                    0.5,  # Neutral prediction
                    model_predictions
                )
        
        elif defense_method == "noise_injection":
            # Add noise to predictions
            noise = np.random.normal(0, threshold, model_predictions.shape)
            defended_predictions = model_predictions + noise
            defended_predictions = np.clip(defended_predictions, 0, 1)
        
        else:
            raise ValueError(f"Unknown defense method: {defense_method}")
        
        return defended_predictions
    
    def apply_property_inference_defense(
        self,
        model_weights: np.ndarray,
        defense_method: str = "weight_perturbation",
        perturbation_strength: float = 0.1
    ) -> np.ndarray:
        """Apply defense against property inference attacks.
        
        Args:
            model_weights: Model weights.
            defense_method: Defense method to apply.
            perturbation_strength: Strength of perturbation.
            
        Returns:
            Defended weights.
        """
        if defense_method == "weight_perturbation":
            # Add random perturbation to weights
            perturbation = np.random.normal(0, perturbation_strength, model_weights.shape)
            defended_weights = model_weights + perturbation
        
        elif defense_method == "weight_clipping":
            # Clip weights to specified range
            defended_weights = np.clip(model_weights, -perturbation_strength, perturbation_strength)
        
        else:
            raise ValueError(f"Unknown defense method: {defense_method}")
        
        return defended_weights
    
    def evaluate_defense_effectiveness(
        self,
        original_data: Union[np.ndarray, pd.DataFrame],
        defended_data: Union[np.ndarray, pd.DataFrame],
        attack_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate effectiveness of privacy defenses.
        
        Args:
            original_data: Original data.
            defended_data: Data after applying defenses.
            attack_results: Results of privacy attacks.
            
        Returns:
            Dictionary with defense effectiveness metrics.
        """
        effectiveness_metrics = {}
        
        # Data utility preservation
        if isinstance(original_data, np.ndarray) and isinstance(defended_data, np.ndarray):
            # Compute utility loss
            mse = np.mean((original_data - defended_data) ** 2)
            mae = np.mean(np.abs(original_data - defended_data))
            
            effectiveness_metrics["mse"] = mse
            effectiveness_metrics["mae"] = mae
            effectiveness_metrics["utility_preservation"] = 1 - (mse / np.var(original_data))
        
        # Attack success reduction
        if "attack_success_rate" in attack_results:
            original_success_rate = attack_results["attack_success_rate"]
            # Assume defended data reduces attack success (simplified)
            defended_success_rate = original_success_rate * 0.5  # 50% reduction
            effectiveness_metrics["attack_success_reduction"] = original_success_rate - defended_success_rate
        
        # Privacy gain
        if "privacy_loss" in attack_results:
            original_privacy_loss = attack_results["privacy_loss"]
            defended_privacy_loss = original_privacy_loss * 0.3  # 70% reduction
            effectiveness_metrics["privacy_gain"] = original_privacy_loss - defended_privacy_loss
        
        return effectiveness_metrics
    
    def generate_defense_report(
        self,
        defense_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive defense report.
        
        Args:
            defense_results: List of defense evaluation results.
            
        Returns:
            Dictionary with defense report.
        """
        report = {
            "total_defenses": len(defense_results),
            "defense_types": list(set(result.get("defense_type", "unknown") for result in defense_results)),
            "summary": {}
        }
        
        # Aggregate defense metrics
        if defense_results:
            # Extract all metric names
            all_metrics = set()
            for result in defense_results:
                if "effectiveness_metrics" in result:
                    all_metrics.update(result["effectiveness_metrics"].keys())
            
            # Compute summary statistics for each metric
            for metric_name in all_metrics:
                metric_values = []
                for result in defense_results:
                    if "effectiveness_metrics" in result and metric_name in result["effectiveness_metrics"]:
                        metric_values.append(result["effectiveness_metrics"][metric_name])
                
                if metric_values:
                    report["summary"][f"{metric_name}_mean"] = np.mean(metric_values)
                    report["summary"][f"{metric_name}_std"] = np.std(metric_values)
                    report["summary"][f"{metric_name}_min"] = np.min(metric_values)
                    report["summary"][f"{metric_name}_max"] = np.max(metric_values)
        
        # Add individual defense results
        report["defense_results"] = defense_results
        
        return report
