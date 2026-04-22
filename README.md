# Differential Privacy Implementation

A comprehensive differential privacy implementation for research and education, focusing on privacy-preserving data analysis and machine learning.

## Overview

This project provides a complete implementation of differential privacy mechanisms, including:

- **Core DP Mechanisms**: Laplace, Gaussian, Exponential, Histogram, and Quantile mechanisms
- **DP-SGD**: Differential Privacy Stochastic Gradient Descent for machine learning
- **PATE**: Private Aggregation of Teacher Ensembles
- **Privacy Defenses**: Input sanitization, k-anonymity, l-diversity, t-closeness
- **Evaluation Tools**: Privacy accounting, utility evaluation, and comprehensive metrics
- **Interactive Demo**: Streamlit-based demonstration interface

## Features

### Privacy Mechanisms
- **Laplace Mechanism**: For numerical queries with calibrated noise
- **Gaussian Mechanism**: For numerical queries with (ε, δ)-differential privacy
- **Exponential Mechanism**: For non-numerical queries with utility-based selection
- **Histogram Mechanism**: For frequency counts with privacy protection
- **Quantile Mechanism**: For statistical quantiles with noise calibration

### Machine Learning
- **DP-SGD**: Privacy-preserving stochastic gradient descent
- **PATE**: Teacher-student framework for private model training
- **Privacy Accounting**: Track privacy budget consumption
- **Utility Evaluation**: Measure impact of privacy on model performance

### Data Processing
- **Synthetic Data Generation**: Create privacy-safe datasets for experimentation
- **Data Anonymization**: Remove or obfuscate personally identifiable information
- **Feature Engineering**: Extract privacy-preserving features
- **Entity-Aware Splits**: Prevent data leakage in evaluation

### Privacy Defenses
- **Input Sanitization**: Clean and normalize input data
- **K-Anonymity**: Ensure minimum group sizes
- **L-Diversity**: Maintain attribute diversity within groups
- **T-Closeness**: Control distribution distance
- **Synthetic Data**: Generate privacy-preserving synthetic datasets

### Evaluation and Visualization
- **Privacy Metrics**: Track ε, δ consumption and privacy loss
- **Utility Metrics**: Measure accuracy, precision, recall, and other performance indicators
- **Trade-off Analysis**: Visualize privacy-utility trade-offs
- **Interactive Dashboards**: Streamlit-based exploration interface

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Differential-Privacy-Implementation.git
cd Differential-Privacy-Implementation

# Install the package
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### Development Installation

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Quick Start

### Basic Usage

```python
import numpy as np
from dp_implementation import DPMechanisms, DataGenerator

# Generate synthetic data
data_generator = DataGenerator(random_seed=42)
X, y = data_generator.generate_synthetic_classification(n_samples=1000, n_features=10)

# Initialize DP mechanisms
dp_mechanisms = DPMechanisms(random_seed=42)

# Apply Laplace mechanism
data = X.flatten()
result, metadata = dp_mechanisms.laplace_mechanism(data, epsilon=1.0, function="mean")

print(f"True mean: {metadata['true_result']:.4f}")
print(f"Private mean: {result:.4f}")
print(f"Noise added: {metadata['noise']:.4f}")
```

### DP-SGD Example

```python
import torch
from dp_implementation import DPSGD
from dp_implementation.models.dp_sgd import SimpleDPModel

# Create model
model = SimpleDPModel(input_dim=10, hidden_dim=128, num_classes=2)

# Initialize DP-SGD
dp_sgd = DPSGD(
    model=model,
    epsilon=1.0,
    delta=1e-5,
    noise_multiplier=1.1,
    max_grad_norm=1.0
)

# Train with privacy
history = dp_sgd.train(train_loader, val_loader, epochs=100)

# Get privacy report
privacy_report = dp_sgd.get_privacy_report()
print(f"Privacy budget used: {privacy_report['epsilon_used']:.4f}")
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
differential-privacy-implementation/
├── src/dp_implementation/          # Main package
│   ├── data/                      # Data generation and processing
│   ├── features/                  # Feature extraction
│   ├── models/                    # DP mechanisms and ML models
│   ├── defenses/                  # Privacy defense mechanisms
│   ├── eval/                      # Evaluation and metrics
│   ├── viz/                       # Visualization utilities
│   └── utils/                     # Configuration and utilities
├── configs/                       # Configuration files
├── scripts/                       # Experiment scripts
├── tests/                         # Unit tests
├── demo/                          # Interactive demo
├── assets/                        # Generated plots and results
├── notebooks/                     # Jupyter notebooks
├── pyproject.toml                 # Package configuration
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── DISCLAIMER.md                  # Important limitations
```

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
data:
  synthetic:
    n_samples: 1000
    n_features: 10
    random_seed: 42

privacy:
  epsilon: 1.0
  delta: 1e-5
  mechanisms: ["laplace", "gaussian"]

model:
  learning_rate: 0.01
  batch_size: 32
  epochs: 100
```

## Running Experiments

### Command Line Interface

```bash
# Run all experiments
python scripts/run_experiment.py --config configs/default.yaml

# Run specific experiment
python scripts/run_experiment.py --experiment dp_sgd --config configs/default.yaml

# Run with custom parameters
python scripts/run_experiment.py --experiment mechanisms --log-level DEBUG
```

### Programmatic Usage

```python
from dp_implementation import Config
from scripts.run_experiment import run_basic_mechanisms_experiment

# Load configuration
config = Config("configs/default.yaml")

# Run experiment
results = run_basic_mechanisms_experiment(config)
print(results)
```

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_mechanisms.py

# Run with coverage
pytest --cov=src/dp_implementation tests/
```

## Privacy Mechanisms

### Laplace Mechanism
Adds calibrated Laplace noise to numerical queries:

```python
result, metadata = dp_mechanisms.laplace_mechanism(
    data, epsilon=1.0, function="mean"
)
```

### Gaussian Mechanism
Adds calibrated Gaussian noise for (ε, δ)-differential privacy:

```python
result, metadata = dp_mechanisms.gaussian_mechanism(
    data, epsilon=1.0, delta=1e-5, function="mean"
)
```

### Exponential Mechanism
Selects outcomes based on utility scores:

```python
candidates = ["Option A", "Option B", "Option C"]
scores = np.array([0.8, 0.6, 0.9])
result, metadata = dp_mechanisms.exponential_mechanism(
    candidates, scores, epsilon=1.0
)
```

### Histogram Mechanism
Protects frequency counts in histograms:

```python
result, metadata = dp_mechanisms.histogram_mechanism(
    data, bins=10, epsilon=1.0
)
```

### Quantile Mechanism
Protects statistical quantiles:

```python
result, metadata = dp_mechanisms.quantile_mechanism(
    data, quantile=0.5, epsilon=1.0, delta=1e-5
)
```

## Privacy Defenses

### Input Sanitization
Clean and normalize input data:

```python
from dp_implementation import PrivacyDefense

privacy_defense = PrivacyDefense()
sanitized_data = privacy_defense.apply_input_sanitization(
    data, method="clipping", clip_value=1.0
)
```

### K-Anonymity
Ensure minimum group sizes:

```python
k_anonymous_data = privacy_defense.apply_k_anonymity(
    data, quasi_identifiers=["age", "education"], k=5
)
```

### L-Diversity
Maintain attribute diversity:

```python
l_diverse_data = privacy_defense.apply_l_diversity(
    data, quasi_identifiers=["age", "education"], 
    sensitive_attribute="income", l=2
)
```

### T-Closeness
Control distribution distance:

```python
t_close_data = privacy_defense.apply_t_closeness(
    data, quasi_identifiers=["age", "education"],
    sensitive_attribute="income", t=0.1
)
```

## Evaluation Metrics

### Privacy Metrics
- **ε (Epsilon)**: Privacy parameter measuring privacy loss
- **δ (Delta)**: Failure probability parameter
- **Privacy Budget**: Total privacy consumption
- **Sensitivity**: Maximum change in output from single record change

### Utility Metrics
- **Accuracy**: Classification accuracy
- **Precision/Recall**: Classification performance
- **Mean Absolute Error**: Regression error
- **R² Score**: Regression goodness of fit
- **Utility Loss**: Degradation due to privacy

### Trade-off Analysis
- **Privacy-Utility Curves**: Visualize trade-offs
- **Optimal Operating Points**: Find best ε for given utility
- **Efficiency Metrics**: Measure privacy-utility efficiency

## Visualization

### Static Plots
```python
from dp_implementation import PrivacyVisualizer

visualizer = PrivacyVisualizer()
fig = visualizer.plot_privacy_utility_tradeoff(
    privacy_levels, utility_scores, title="DP Tradeoff"
)
```

### Interactive Plots
```python
fig = visualizer.plot_interactive_privacy_utility(
    privacy_levels, utility_scores
)
fig.show()
```

### Comprehensive Dashboard
```python
dashboard = visualizer.create_dashboard(results)
dashboard.show()
```

## Data Schemas

### Synthetic Classification Data
```python
X, y = data_generator.generate_synthetic_classification(
    n_samples=1000, n_features=10, n_classes=2
)
# X: (1000, 10) feature matrix
# y: (1000,) class labels
```

### Income Data
```python
income_data = data_generator.generate_income_data(n_samples=1000)
# Columns: age, education, experience_years, income, zip_code, ssn_last4
```

### Health Data
```python
health_data = data_generator.generate_health_data(n_samples=1000)
# Columns: patient_id, age, gender, bmi, blood_pressure_*, cholesterol, diabetes, heart_disease
```

### Location Data
```python
location_data = data_generator.generate_location_data(n_samples=1000)
# Columns: user_id, timestamp, latitude, longitude
```

### Transaction Data
```python
transaction_data = data_generator.generate_transaction_data(n_samples=1000)
# Columns: user_id, merchant_id, timestamp, amount, category
```

## Privacy Accounting

Track privacy budget consumption:

```python
from dp_implementation import PrivacyAccountant

accountant = PrivacyAccountant(delta=1e-5)
accountant.add_mechanism(epsilon=1.0, delta=1e-5)

# Get total privacy loss
total_epsilon, total_delta = accountant.get_total_privacy_loss()
print(f"Total ε: {total_epsilon}, Total δ: {total_delta}")

# Get privacy report
report = accountant.get_privacy_report()
print(report)
```

## Advanced Features

### Composition
```python
# Basic composition
total_epsilon, total_delta = accountant.get_total_privacy_loss()

# Advanced composition
adv_epsilon, adv_delta = accountant.get_advanced_composition(k=5)

# Moments accountant (for DP-SGD)
moments_epsilon, moments_delta = accountant.get_moments_accountant(
    noise_multiplier=1.1, steps=1000
)
```

### Amplification
```python
amplification_eval = privacy_evaluator.evaluate_privacy_amplification(
    original_epsilon=1.0, original_delta=1e-5,
    amplification_factor=0.1, mechanism_type="subsampling"
)
```

### Membership Inference Defense
```python
defended_predictions = privacy_defense.apply_membership_inference_defense(
    model_predictions, defense_method="confidence_thresholding", threshold=0.5
)
```

## Contributing

We welcome contributions to improve this educational implementation:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-mechanism`
3. **Make your changes** with proper tests and documentation
4. **Run tests**: `pytest tests/`
5. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Write tests for new functionality
- Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{differential_privacy_implementation,
  title={Differential Privacy Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Differential-Privacy-Implementation}
}
```

## Acknowledgments

- **OpenDP**: For differential privacy framework inspiration
- **Diffprivlib**: For privacy mechanism implementations
- **Opacus**: For DP-SGD implementation
- **Privacy Research Community**: For theoretical foundations

## Support

For questions and support:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check the comprehensive docstrings and examples
- **Community**: Join privacy research communities for advanced topics

## Changelog

### Version 1.0.0 (2024)
- Initial release with core DP mechanisms
- DP-SGD and PATE implementations
- Privacy defense mechanisms
- Interactive Streamlit demo
- Comprehensive evaluation tools
- Full test coverage
- Documentation and examples

---

**⚠️ Important**: This implementation is for **research and educational purposes only**. See [DISCLAIMER.md](DISCLAIMER.md) for important limitations and usage guidelines.

** Privacy First**: Always prioritize privacy and consult with experts before any real-world application.

** Learn More**: Explore the interactive demo and examples to understand differential privacy concepts.
# Differential-Privacy-Implementation
