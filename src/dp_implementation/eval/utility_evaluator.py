"""Utility evaluation utilities for differential privacy experiments."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import cross_val_score


class UtilityEvaluator:
    """Utility evaluation utilities for differential privacy experiments.
    
    This class provides methods for evaluating the utility of differentially
    private mechanisms and their impact on model performance.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize utility evaluator.
        
        Args:
            random_seed: Random seed for reproducible evaluation.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def evaluate_classification_utility(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate utility for classification tasks.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities (optional).
            
        Returns:
            Dictionary with classification utility metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }
        
        # Add ROC AUC if probabilities are provided
        if y_prob is not None:
            try:
                if y_prob.ndim > 1 and y_prob.shape[1] > 2:
                    # Multi-class ROC AUC
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                else:
                    # Binary ROC AUC
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = 0.0
        
        return metrics
    
    def evaluate_regression_utility(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate utility for regression tasks.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary with regression utility metrics.
        """
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred)
        }
        
        # Add relative metrics
        if np.mean(y_true) != 0:
            metrics["mape"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics["smape"] = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        return metrics
    
    def evaluate_statistical_utility(
        self,
        true_statistics: Dict[str, float],
        private_statistics: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate utility for statistical queries.
        
        Args:
            true_statistics: True statistical values.
            private_statistics: Differentially private statistical values.
            
        Returns:
            Dictionary with statistical utility metrics.
        """
        metrics = {}
        
        for stat_name in true_statistics:
            if stat_name in private_statistics:
                true_val = true_statistics[stat_name]
                private_val = private_statistics[stat_name]
                
                # Absolute error
                metrics[f"{stat_name}_absolute_error"] = abs(true_val - private_val)
                
                # Relative error
                if true_val != 0:
                    metrics[f"{stat_name}_relative_error"] = abs(true_val - private_val) / abs(true_val)
                else:
                    metrics[f"{stat_name}_relative_error"] = float('inf') if private_val != 0 else 0.0
                
                # Squared error
                metrics[f"{stat_name}_squared_error"] = (true_val - private_val) ** 2
        
        # Overall utility metrics
        if metrics:
            metrics["mean_absolute_error"] = np.mean([v for k, v in metrics.items() if k.endswith("_absolute_error")])
            metrics["mean_relative_error"] = np.mean([v for k, v in metrics.items() if k.endswith("_relative_error") and v != float('inf')])
            metrics["mean_squared_error"] = np.mean([v for k, v in metrics.items() if k.endswith("_squared_error")])
        
        return metrics
    
    def evaluate_utility_loss(
        self,
        baseline_utility: Dict[str, float],
        private_utility: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate utility loss due to privacy.
        
        Args:
            baseline_utility: Utility metrics without privacy.
            private_utility: Utility metrics with privacy.
            
        Returns:
            Dictionary with utility loss metrics.
        """
        utility_loss = {}
        
        for metric_name in baseline_utility:
            if metric_name in private_utility:
                baseline_val = baseline_utility[metric_name]
                private_val = private_utility[metric_name]
                
                # Absolute loss
                utility_loss[f"{metric_name}_loss"] = baseline_val - private_val
                
                # Relative loss
                if baseline_val != 0:
                    utility_loss[f"{metric_name}_relative_loss"] = (baseline_val - private_val) / baseline_val
                else:
                    utility_loss[f"{metric_name}_relative_loss"] = 0.0
        
        # Overall utility loss
        if utility_loss:
            utility_loss["mean_absolute_loss"] = np.mean([v for k, v in utility_loss.items() if k.endswith("_loss") and not k.endswith("_relative_loss")])
            utility_loss["mean_relative_loss"] = np.mean([v for k, v in utility_loss.items() if k.endswith("_relative_loss")])
        
        return utility_loss
    
    def evaluate_utility_privacy_tradeoff(
        self,
        privacy_levels: List[float],
        utility_scores: List[float],
        privacy_metric: str = "epsilon"
    ) -> Dict[str, Any]:
        """Evaluate utility-privacy tradeoff.
        
        Args:
            privacy_levels: List of privacy parameter values.
            utility_scores: List of corresponding utility scores.
            privacy_metric: Privacy metric to use ("epsilon", "delta").
            
        Returns:
            Dictionary with utility-privacy tradeoff evaluation.
        """
        if len(privacy_levels) != len(utility_scores):
            raise ValueError("Privacy levels and utility scores must have the same length")
        
        # Compute correlation
        correlation = np.corrcoef(privacy_levels, utility_scores)[0, 1]
        
        # Find optimal operating point
        if privacy_metric == "epsilon":
            # Higher epsilon = less privacy, so we want to find the point with
            # highest utility for a given privacy level
            optimal_idx = np.argmax(utility_scores)
        else:
            # For delta, lower is better, so we need to consider both metrics
            optimal_idx = np.argmax(utility_scores)
        
        # Compute efficiency metrics
        efficiency = utility_scores[optimal_idx] / privacy_levels[optimal_idx] if privacy_levels[optimal_idx] > 0 else 0
        
        # Compute utility loss at different privacy levels
        max_utility = max(utility_scores)
        utility_losses = [max_utility - score for score in utility_scores]
        
        tradeoff_evaluation = {
            "privacy_levels": privacy_levels,
            "utility_scores": utility_scores,
            "utility_losses": utility_losses,
            "correlation": correlation,
            "optimal_privacy_level": privacy_levels[optimal_idx],
            "optimal_utility": utility_scores[optimal_idx],
            "efficiency": efficiency,
            "privacy_metric": privacy_metric,
            "max_utility": max_utility,
            "min_utility": min(utility_scores)
        }
        
        return tradeoff_evaluation
    
    def evaluate_utility_consistency(
        self,
        utility_results: List[Dict[str, float]],
        metric_name: str
    ) -> Dict[str, float]:
        """Evaluate consistency of utility across multiple runs.
        
        Args:
            utility_results: List of utility results from multiple runs.
            metric_name: Name of the metric to evaluate.
            
        Returns:
            Dictionary with consistency metrics.
        """
        if not utility_results:
            return {}
        
        # Extract metric values
        metric_values = [result.get(metric_name, 0) for result in utility_results]
        
        if not metric_values:
            return {}
        
        # Compute consistency metrics
        consistency_metrics = {
            f"{metric_name}_mean": np.mean(metric_values),
            f"{metric_name}_std": np.std(metric_values),
            f"{metric_name}_min": np.min(metric_values),
            f"{metric_name}_max": np.max(metric_values),
            f"{metric_name}_cv": np.std(metric_values) / np.mean(metric_values) if np.mean(metric_values) != 0 else 0,
            f"{metric_name}_range": np.max(metric_values) - np.min(metric_values)
        }
        
        return consistency_metrics
    
    def evaluate_utility_robustness(
        self,
        baseline_utility: Dict[str, float],
        perturbed_utilities: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Evaluate robustness of utility to perturbations.
        
        Args:
            baseline_utility: Baseline utility metrics.
            perturbed_utilities: List of utility metrics under perturbations.
            
        Returns:
            Dictionary with robustness metrics.
        """
        robustness_metrics = {}
        
        for metric_name in baseline_utility:
            baseline_val = baseline_utility[metric_name]
            perturbed_vals = [result.get(metric_name, 0) for result in perturbed_utilities]
            
            if perturbed_vals:
                # Compute robustness metrics
                robustness_metrics[f"{metric_name}_mean_perturbation"] = np.mean(perturbed_vals)
                robustness_metrics[f"{metric_name}_std_perturbation"] = np.std(perturbed_vals)
                robustness_metrics[f"{metric_name}_max_deviation"] = max(abs(val - baseline_val) for val in perturbed_vals)
                robustness_metrics[f"{metric_name}_relative_deviation"] = robustness_metrics[f"{metric_name}_max_deviation"] / abs(baseline_val) if baseline_val != 0 else 0
                robustness_metrics[f"{metric_name}_robustness_score"] = 1 - (robustness_metrics[f"{metric_name}_relative_deviation"] / (1 + robustness_metrics[f"{metric_name}_relative_deviation"]))
        
        return robustness_metrics
    
    def evaluate_utility_efficiency(
        self,
        utility_scores: List[float],
        computational_costs: List[float]
    ) -> Dict[str, float]:
        """Evaluate efficiency of utility.
        
        Args:
            utility_scores: List of utility scores.
            computational_costs: List of computational costs.
            
        Returns:
            Dictionary with efficiency metrics.
        """
        if len(utility_scores) != len(computational_costs):
            raise ValueError("Utility scores and computational costs must have the same length")
        
        # Compute efficiency metrics
        efficiency_scores = [score / cost for score, cost in zip(utility_scores, computational_costs) if cost > 0]
        
        efficiency_metrics = {
            "mean_efficiency": np.mean(efficiency_scores) if efficiency_scores else 0,
            "std_efficiency": np.std(efficiency_scores) if efficiency_scores else 0,
            "max_efficiency": max(efficiency_scores) if efficiency_scores else 0,
            "min_efficiency": min(efficiency_scores) if efficiency_scores else 0,
            "efficiency_cv": np.std(efficiency_scores) / np.mean(efficiency_scores) if efficiency_scores and np.mean(efficiency_scores) != 0 else 0
        }
        
        return efficiency_metrics
    
    def generate_utility_report(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive utility report.
        
        Args:
            evaluations: List of evaluation results.
            
        Returns:
            Dictionary with comprehensive utility report.
        """
        report = {
            "total_evaluations": len(evaluations),
            "evaluation_types": list(set(eval.get("type", "unknown") for eval in evaluations)),
            "summary": {}
        }
        
        # Aggregate utility metrics
        if evaluations:
            # Extract all metric names
            all_metrics = set()
            for eval_result in evaluations:
                if "metrics" in eval_result:
                    all_metrics.update(eval_result["metrics"].keys())
            
            # Compute summary statistics for each metric
            for metric_name in all_metrics:
                metric_values = []
                for eval_result in evaluations:
                    if "metrics" in eval_result and metric_name in eval_result["metrics"]:
                        metric_values.append(eval_result["metrics"][metric_name])
                
                if metric_values:
                    report["summary"][f"{metric_name}_mean"] = np.mean(metric_values)
                    report["summary"][f"{metric_name}_std"] = np.std(metric_values)
                    report["summary"][f"{metric_name}_min"] = np.min(metric_values)
                    report["summary"][f"{metric_name}_max"] = np.max(metric_values)
        
        # Add individual evaluations
        report["evaluations"] = evaluations
        
        return report
