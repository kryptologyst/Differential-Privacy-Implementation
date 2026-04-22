"""Privacy evaluation utilities for differential privacy experiments."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

from ..utils.privacy_accountant import PrivacyAccountant


class PrivacyEvaluator:
    """Privacy evaluation utilities for differential privacy experiments.
    
    This class provides methods for evaluating the privacy guarantees
    of differential privacy mechanisms and their impact on utility.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize privacy evaluator.
        
        Args:
            random_seed: Random seed for reproducible evaluation.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.privacy_accountant = PrivacyAccountant()
    
    def evaluate_privacy_budget(
        self,
        epsilon: float,
        delta: float,
        mechanism_type: str = "laplace"
    ) -> Dict[str, Any]:
        """Evaluate privacy budget consumption.
        
        Args:
            epsilon: Privacy parameter epsilon.
            delta: Privacy parameter delta.
            mechanism_type: Type of mechanism used.
            
        Returns:
            Dictionary with privacy budget evaluation.
        """
        # Add mechanism to privacy accountant
        self.privacy_accountant.add_mechanism(epsilon, delta)
        
        # Get privacy report
        privacy_report = self.privacy_accountant.get_privacy_report()
        
        # Evaluate privacy budget
        budget_evaluation = {
            "epsilon_used": epsilon,
            "delta_used": delta,
            "mechanism_type": mechanism_type,
            "privacy_budget_remaining": 1.0 - epsilon,  # Assuming total budget of 1.0
            "privacy_budget_exhausted": epsilon >= 1.0,
            "privacy_report": privacy_report
        }
        
        return budget_evaluation
    
    def evaluate_privacy_utility_tradeoff(
        self,
        privacy_levels: List[float],
        utility_scores: List[float],
        privacy_metric: str = "epsilon"
    ) -> Dict[str, Any]:
        """Evaluate privacy-utility tradeoff.
        
        Args:
            privacy_levels: List of privacy parameter values.
            utility_scores: List of corresponding utility scores.
            privacy_metric: Privacy metric to use ("epsilon", "delta").
            
        Returns:
            Dictionary with privacy-utility tradeoff evaluation.
        """
        if len(privacy_levels) != len(utility_scores):
            raise ValueError("Privacy levels and utility scores must have the same length")
        
        # Compute correlation
        correlation = np.corrcoef(privacy_levels, utility_scores)[0, 1]
        
        # Find optimal operating point (highest utility for given privacy)
        if privacy_metric == "epsilon":
            # Higher epsilon = less privacy, so we want to find the point with
            # highest utility for a given privacy level
            optimal_idx = np.argmax(utility_scores)
        else:
            # For delta, lower is better, so we need to consider both metrics
            optimal_idx = np.argmax(utility_scores)
        
        # Compute privacy-utility efficiency
        efficiency = utility_scores[optimal_idx] / privacy_levels[optimal_idx]
        
        tradeoff_evaluation = {
            "privacy_levels": privacy_levels,
            "utility_scores": utility_scores,
            "correlation": correlation,
            "optimal_privacy_level": privacy_levels[optimal_idx],
            "optimal_utility": utility_scores[optimal_idx],
            "efficiency": efficiency,
            "privacy_metric": privacy_metric
        }
        
        return tradeoff_evaluation
    
    def evaluate_membership_inference_risk(
        self,
        model_predictions: np.ndarray,
        true_labels: np.ndarray,
        membership_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate membership inference attack risk.
        
        Args:
            model_predictions: Model predictions on data.
            true_labels: True labels for the data.
            membership_labels: Binary labels indicating membership (1) or non-membership (0).
            
        Returns:
            Dictionary with membership inference risk evaluation.
        """
        # Compute prediction confidence
        if model_predictions.ndim > 1:
            # Multi-class predictions
            prediction_confidence = np.max(model_predictions, axis=1)
        else:
            # Binary predictions
            prediction_confidence = np.abs(model_predictions - 0.5) * 2
        
        # Split by membership
        member_confidence = prediction_confidence[membership_labels == 1]
        non_member_confidence = prediction_confidence[membership_labels == 0]
        
        # Compute statistics
        member_mean_confidence = np.mean(member_confidence)
        non_member_mean_confidence = np.mean(non_member_confidence)
        confidence_gap = member_mean_confidence - non_member_mean_confidence
        
        # Compute attack success rate (simplified)
        # Higher confidence gap indicates higher risk
        risk_score = confidence_gap / (member_mean_confidence + non_member_confidence)
        
        # Compute accuracy difference
        member_accuracy = accuracy_score(
            true_labels[membership_labels == 1],
            np.argmax(model_predictions[membership_labels == 1], axis=1) if model_predictions.ndim > 1
            else (model_predictions[membership_labels == 1] > 0.5).astype(int)
        )
        non_member_accuracy = accuracy_score(
            true_labels[membership_labels == 0],
            np.argmax(model_predictions[membership_labels == 0], axis=1) if model_predictions.ndim > 1
            else (model_predictions[membership_labels == 0] > 0.5).astype(int)
        )
        accuracy_gap = member_accuracy - non_member_accuracy
        
        risk_evaluation = {
            "member_mean_confidence": member_mean_confidence,
            "non_member_mean_confidence": non_member_mean_confidence,
            "confidence_gap": confidence_gap,
            "risk_score": risk_score,
            "member_accuracy": member_accuracy,
            "non_member_accuracy": non_member_accuracy,
            "accuracy_gap": accuracy_gap,
            "high_risk": risk_score > 0.1  # Threshold for high risk
        }
        
        return risk_evaluation
    
    def evaluate_differential_privacy_guarantees(
        self,
        mechanism_results: List[Dict[str, Any]],
        epsilon: float,
        delta: float
    ) -> Dict[str, Any]:
        """Evaluate differential privacy guarantees.
        
        Args:
            mechanism_results: List of mechanism results.
            epsilon: Privacy parameter epsilon.
            delta: Privacy parameter delta.
            
        Returns:
            Dictionary with DP guarantees evaluation.
        """
        # Extract noise levels and sensitivities
        noise_levels = [result.get("noise", 0) for result in mechanism_results]
        sensitivities = [result.get("sensitivity", 0) for result in mechanism_results]
        
        # Compute noise-to-sensitivity ratio
        noise_sensitivity_ratios = [
            abs(noise) / max(sensitivity, 1e-10) for noise, sensitivity in zip(noise_levels, sensitivities)
        ]
        
        # Evaluate privacy guarantees
        avg_noise_sensitivity_ratio = np.mean(noise_sensitivity_ratios)
        noise_consistency = np.std(noise_sensitivity_ratios) / np.mean(noise_sensitivity_ratios)
        
        # Check if noise is sufficient for privacy
        theoretical_noise_scale = sensitivities[0] / epsilon if sensitivities else 0
        actual_noise_scale = np.std(noise_levels) if noise_levels else 0
        noise_adequacy = actual_noise_scale / max(theoretical_noise_scale, 1e-10)
        
        # Evaluate privacy budget consumption
        total_epsilon_used = sum(result.get("epsilon", 0) for result in mechanism_results)
        total_delta_used = sum(result.get("delta", 0) for result in mechanism_results)
        
        guarantees_evaluation = {
            "epsilon": epsilon,
            "delta": delta,
            "total_epsilon_used": total_epsilon_used,
            "total_delta_used": total_delta_used,
            "avg_noise_sensitivity_ratio": avg_noise_sensitivity_ratio,
            "noise_consistency": noise_consistency,
            "noise_adequacy": noise_adequacy,
            "privacy_budget_exhausted": total_epsilon_used >= epsilon,
            "privacy_guarantees_met": noise_adequacy >= 0.8 and total_epsilon_used <= epsilon
        }
        
        return guarantees_evaluation
    
    def evaluate_privacy_amplification(
        self,
        original_epsilon: float,
        original_delta: float,
        amplification_factor: float,
        mechanism_type: str = "subsampling"
    ) -> Dict[str, Any]:
        """Evaluate privacy amplification effects.
        
        Args:
            original_epsilon: Original privacy parameter epsilon.
            original_delta: Original privacy parameter delta.
            amplification_factor: Amplification factor.
            mechanism_type: Type of amplification mechanism.
            
        Returns:
            Dictionary with privacy amplification evaluation.
        """
        if mechanism_type == "subsampling":
            # Subsampling amplification
            amplified_epsilon = np.log(1 + amplification_factor * (np.exp(original_epsilon) - 1))
            amplified_delta = amplification_factor * original_delta
        elif mechanism_type == "shuffling":
            # Shuffling amplification (simplified)
            amplified_epsilon = original_epsilon * amplification_factor
            amplified_delta = original_delta * amplification_factor
        else:
            raise ValueError(f"Unknown amplification mechanism: {mechanism_type}")
        
        # Compute privacy improvement
        epsilon_improvement = (original_epsilon - amplified_epsilon) / original_epsilon
        delta_improvement = (original_delta - amplified_delta) / original_delta
        
        amplification_evaluation = {
            "original_epsilon": original_epsilon,
            "original_delta": original_delta,
            "amplified_epsilon": amplified_epsilon,
            "amplified_delta": amplified_delta,
            "amplification_factor": amplification_factor,
            "mechanism_type": mechanism_type,
            "epsilon_improvement": epsilon_improvement,
            "delta_improvement": delta_improvement,
            "privacy_improved": epsilon_improvement > 0 and delta_improvement > 0
        }
        
        return amplification_evaluation
    
    def evaluate_composition_privacy(
        self,
        mechanism_epsilons: List[float],
        mechanism_deltas: List[float],
        composition_type: str = "basic"
    ) -> Dict[str, Any]:
        """Evaluate privacy under composition.
        
        Args:
            mechanism_epsilons: List of epsilon values for each mechanism.
            mechanism_deltas: List of delta values for each mechanism.
            composition_type: Type of composition ("basic", "advanced").
            
        Returns:
            Dictionary with composition privacy evaluation.
        """
        if len(mechanism_epsilons) != len(mechanism_deltas):
            raise ValueError("Epsilon and delta lists must have the same length")
        
        if composition_type == "basic":
            # Basic composition
            total_epsilon = sum(mechanism_epsilons)
            total_delta = sum(mechanism_deltas)
        elif composition_type == "advanced":
            # Advanced composition (simplified)
            k = len(mechanism_epsilons)
            max_epsilon = max(mechanism_epsilons)
            total_epsilon = np.sqrt(2 * k * np.log(1 / max(mechanism_deltas))) * max_epsilon + k * max_epsilon * (np.exp(max_epsilon) - 1)
            total_delta = k * max(mechanism_deltas) + max(mechanism_deltas)
        else:
            raise ValueError(f"Unknown composition type: {composition_type}")
        
        # Evaluate composition efficiency
        basic_total_epsilon = sum(mechanism_epsilons)
        composition_efficiency = basic_total_epsilon / total_epsilon if total_epsilon > 0 else 0
        
        composition_evaluation = {
            "mechanism_epsilons": mechanism_epsilons,
            "mechanism_deltas": mechanism_deltas,
            "composition_type": composition_type,
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "basic_total_epsilon": basic_total_epsilon,
            "composition_efficiency": composition_efficiency,
            "privacy_budget_used": total_epsilon,
            "mechanism_count": len(mechanism_epsilons)
        }
        
        return composition_evaluation
    
    def generate_privacy_report(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive privacy report.
        
        Args:
            evaluations: List of evaluation results.
            
        Returns:
            Dictionary with comprehensive privacy report.
        """
        report = {
            "total_evaluations": len(evaluations),
            "evaluation_types": list(set(eval.get("type", "unknown") for eval in evaluations)),
            "summary": {}
        }
        
        # Aggregate privacy metrics
        if evaluations:
            epsilons = [eval.get("epsilon", 0) for eval in evaluations if "epsilon" in eval]
            deltas = [eval.get("delta", 0) for eval in evaluations if "delta" in eval]
            
            if epsilons:
                report["summary"]["total_epsilon"] = sum(epsilons)
                report["summary"]["avg_epsilon"] = np.mean(epsilons)
                report["summary"]["max_epsilon"] = max(epsilons)
                report["summary"]["min_epsilon"] = min(epsilons)
            
            if deltas:
                report["summary"]["total_delta"] = sum(deltas)
                report["summary"]["avg_delta"] = np.mean(deltas)
                report["summary"]["max_delta"] = max(deltas)
                report["summary"]["min_delta"] = min(deltas)
        
        # Add individual evaluations
        report["evaluations"] = evaluations
        
        return report
