#!/usr/bin/env python3
"""Script to run differential privacy experiments."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dp_implementation import (
    DataGenerator, DataProcessor, DPMechanisms, DPSGD, PATE,
    PrivacyEvaluator, UtilityEvaluator, PrivacyVisualizer, PrivacyDefense,
    PrivacyAccountant, Config
)
from dp_implementation.utils.seeding import set_deterministic_seed, get_device
from dp_implementation.models.dp_sgd import SimpleDPModel

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("experiment.log")
        ]
    )


def run_basic_mechanisms_experiment(config: Config) -> dict:
    """Run basic DP mechanisms experiment."""
    logger = logging.getLogger(__name__)
    logger.info("Running basic DP mechanisms experiment")
    
    # Generate data
    data_generator = DataGenerator(random_seed=config.get("data.synthetic.random_seed", 42))
    X, y = data_generator.generate_synthetic_classification(
        n_samples=config.get("data.synthetic.n_samples", 1000),
        n_features=config.get("data.synthetic.n_features", 10)
    )
    
    # Flatten data for mechanisms
    data = X.flatten()
    
    # Initialize DP mechanisms
    dp_mechanisms = DPMechanisms(random_seed=config.get("data.synthetic.random_seed", 42))
    
    # Test different mechanisms
    mechanisms = config.get("privacy.mechanisms", ["laplace", "gaussian"])
    epsilon = config.get("privacy.epsilon", 1.0)
    delta = config.get("privacy.delta", 1e-5)
    
    results = {}
    
    for mechanism in mechanisms:
        logger.info(f"Testing {mechanism} mechanism")
        
        if mechanism == "laplace":
            result, metadata = dp_mechanisms.laplace_mechanism(data, epsilon, "mean", delta)
        elif mechanism == "gaussian":
            result, metadata = dp_mechanisms.gaussian_mechanism(data, epsilon, delta, "mean")
        elif mechanism == "exponential":
            candidates = ["Option A", "Option B", "Option C", "Option D"]
            scores = np.array([0.8, 0.6, 0.9, 0.7])
            result, metadata = dp_mechanisms.exponential_mechanism(candidates, scores, epsilon)
        else:
            continue
        
        results[mechanism] = {
            "result": result,
            "metadata": metadata
        }
    
    return results


def run_dp_sgd_experiment(config: Config) -> dict:
    """Run DP-SGD experiment."""
    logger = logging.getLogger(__name__)
    logger.info("Running DP-SGD experiment")
    
    # Generate data
    data_generator = DataGenerator(random_seed=config.get("data.synthetic.random_seed", 42))
    X, y = data_generator.generate_synthetic_classification(
        n_samples=config.get("data.synthetic.n_samples", 1000),
        n_features=config.get("data.synthetic.n_features", 10)
    )
    
    # Split data
    X_train, X_test, y_train, y_test = data_generator.create_train_test_split(X, y, test_size=0.2)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get("model.batch_size", 32), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.get("model.batch_size", 32), shuffle=False)
    
    # Create model
    model = SimpleDPModel(
        input_dim=config.get("data.synthetic.n_features", 10),
        hidden_dim=128,
        num_classes=len(np.unique(y))
    )
    
    # Initialize DP-SGD
    dp_sgd = DPSGD(
        model=model,
        epsilon=config.get("privacy.epsilon", 1.0),
        delta=config.get("privacy.delta", 1e-5),
        noise_multiplier=config.get("model.dp_sgd.noise_multiplier", 1.1),
        max_grad_norm=config.get("model.dp_sgd.max_grad_norm", 1.0),
        learning_rate=config.get("model.learning_rate", 0.01)
    )
    
    # Train model
    history = dp_sgd.train(
        train_loader, test_loader, 
        epochs=config.get("model.epochs", 100), 
        verbose=False
    )
    
    # Get privacy report
    privacy_report = dp_sgd.get_privacy_report()
    
    return {
        "history": history,
        "privacy_report": privacy_report,
        "final_accuracy": history["val_accuracy"][-1] if "val_accuracy" in history else history["train_accuracy"][-1]
    }


def run_evaluation_experiment(config: Config) -> dict:
    """Run evaluation experiment."""
    logger = logging.getLogger(__name__)
    logger.info("Running evaluation experiment")
    
    # Initialize evaluators
    privacy_evaluator = PrivacyEvaluator(random_seed=config.get("data.synthetic.random_seed", 42))
    utility_evaluator = UtilityEvaluator(random_seed=config.get("data.synthetic.random_seed", 42))
    
    # Generate data
    data_generator = DataGenerator(random_seed=config.get("data.synthetic.random_seed", 42))
    X, y = data_generator.generate_synthetic_classification(
        n_samples=config.get("data.synthetic.n_samples", 1000),
        n_features=config.get("data.synthetic.n_features", 10)
    )
    
    # Privacy evaluation
    epsilon = config.get("privacy.epsilon", 1.0)
    delta = config.get("privacy.delta", 1e-5)
    
    privacy_budget_eval = privacy_evaluator.evaluate_privacy_budget(epsilon, delta, "laplace")
    
    # Utility evaluation
    true_statistics = {
        "mean": np.mean(X),
        "std": np.std(X),
        "min": np.min(X),
        "max": np.max(X)
    }
    
    # Simulate private statistics
    private_statistics = {
        "mean": np.mean(X) + np.random.normal(0, 0.1),
        "std": np.std(X) + np.random.normal(0, 0.05),
        "min": np.min(X) + np.random.normal(0, 0.1),
        "max": np.max(X) + np.random.normal(0, 0.1)
    }
    
    utility_metrics = utility_evaluator.evaluate_statistical_utility(true_statistics, private_statistics)
    
    return {
        "privacy_budget_eval": privacy_budget_eval,
        "utility_metrics": utility_metrics,
        "true_statistics": true_statistics,
        "private_statistics": private_statistics
    }


def run_defenses_experiment(config: Config) -> dict:
    """Run privacy defenses experiment."""
    logger = logging.getLogger(__name__)
    logger.info("Running privacy defenses experiment")
    
    # Initialize privacy defense
    privacy_defense = PrivacyDefense(random_seed=config.get("data.synthetic.random_seed", 42))
    
    # Generate data
    data_generator = DataGenerator(random_seed=config.get("data.synthetic.random_seed", 42))
    income_data = data_generator.generate_income_data(
        n_samples=config.get("data.synthetic.n_samples", 1000)
    )
    
    # Test different defenses
    results = {}
    
    # Input sanitization
    sanitization_method = config.get("defenses.input_sanitization.method", "clipping")
    clip_value = config.get("defenses.input_sanitization.clip_value", 1.0)
    
    sanitized_data = privacy_defense.apply_input_sanitization(
        income_data, sanitization_method, clip_value
    )
    
    results["input_sanitization"] = {
        "original_shape": income_data.shape,
        "sanitized_shape": sanitized_data.shape,
        "method": sanitization_method,
        "clip_value": clip_value
    }
    
    # K-anonymity
    k_value = config.get("defenses.k_anonymity.k", 5)
    quasi_identifiers = config.get("defenses.k_anonymity.quasi_identifiers", ["age", "education"])
    
    k_anonymous_data = privacy_defense.apply_k_anonymity(
        income_data, quasi_identifiers, k_value
    )
    
    results["k_anonymity"] = {
        "original_records": len(income_data),
        "k_anonymous_records": len(k_anonymous_data),
        "records_removed": len(income_data) - len(k_anonymous_data),
        "k_value": k_value,
        "quasi_identifiers": quasi_identifiers
    }
    
    # Synthetic data generation
    synthetic_method = config.get("defenses.synthetic_data.method", "gaussian")
    noise_level = config.get("defenses.synthetic_data.noise_level", 0.1)
    
    synthetic_data = privacy_defense.apply_synthetic_data_generation(
        income_data, synthetic_method, noise_level
    )
    
    results["synthetic_data"] = {
        "original_shape": income_data.shape,
        "synthetic_shape": synthetic_data.shape,
        "method": synthetic_method,
        "noise_level": noise_level
    }
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run differential privacy experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--experiment", type=str, default="all", 
                       choices=["all", "mechanisms", "dp_sgd", "evaluation", "defenses"],
                       help="Experiment to run")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set deterministic seed
    set_deterministic_seed(config.get("data.synthetic.random_seed", 42))
    
    logger.info(f"Starting experiment: {args.experiment}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    all_results = {}
    
    # Run experiments
    if args.experiment in ["all", "mechanisms"]:
        logger.info("Running basic mechanisms experiment")
        all_results["mechanisms"] = run_basic_mechanisms_experiment(config)
    
    if args.experiment in ["all", "dp_sgd"]:
        logger.info("Running DP-SGD experiment")
        all_results["dp_sgd"] = run_dp_sgd_experiment(config)
    
    if args.experiment in ["all", "evaluation"]:
        logger.info("Running evaluation experiment")
        all_results["evaluation"] = run_evaluation_experiment(config)
    
    if args.experiment in ["all", "defenses"]:
        logger.info("Running defenses experiment")
        all_results["defenses"] = run_defenses_experiment(config)
    
    # Save results
    import json
    results_file = output_dir / "experiment_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for experiment_name, results in all_results.items():
        print(f"\n{experiment_name.upper()}:")
        if experiment_name == "mechanisms":
            for mechanism, result in results.items():
                print(f"  {mechanism}: {result['result']:.4f}")
        elif experiment_name == "dp_sgd":
            print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
            print(f"  Privacy Budget Used: {results['privacy_report']['epsilon_used']:.4f}")
        elif experiment_name == "evaluation":
            print(f"  Privacy Budget: {results['privacy_budget_eval']['epsilon_used']:.4f}")
            print(f"  Mean Absolute Error: {results['utility_metrics']['mean_absolute_error']:.4f}")
        elif experiment_name == "defenses":
            print(f"  Input Sanitization: {results['input_sanitization']['method']}")
            print(f"  K-Anonymity: {results['k_anonymity']['k_anonymous_records']} records")
            print(f"  Synthetic Data: {results['synthetic_data']['method']}")
    
    print("\n" + "="*50)
    print("Experiment completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
