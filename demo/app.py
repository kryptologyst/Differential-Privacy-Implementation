"""Streamlit demo application for differential privacy implementation."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our DP implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dp_implementation import (
    DataGenerator, DataProcessor, DPMechanisms, DPSGD, PATE,
    PrivacyEvaluator, UtilityEvaluator, PrivacyVisualizer, PrivacyDefense,
    PrivacyAccountant, Config
)
from dp_implementation.utils.seeding import set_deterministic_seed, get_device
from dp_implementation.models.dp_sgd import SimpleDPModel

# Set page config
st.set_page_config(
    page_title="Differential Privacy Implementation",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set deterministic seed
set_deterministic_seed(42)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔒 Differential Privacy Implementation</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Disclaimer:</strong> This is a research and educational demonstration of differential privacy concepts. 
        This implementation is not intended for production security operations and may not provide complete privacy guarantees. 
        Always consult with privacy experts before deploying differential privacy in production systems.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Privacy parameters
    st.sidebar.markdown("### Privacy Parameters")
    epsilon = st.sidebar.slider("ε (Epsilon)", 0.1, 10.0, 1.0, 0.1, help="Privacy parameter - lower values provide more privacy")
    delta = st.sidebar.slider("δ (Delta)", 1e-6, 1e-3, 1e-5, 1e-6, format="%.0e", help="Privacy parameter - failure probability")
    
    # Data parameters
    st.sidebar.markdown("### Data Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000, 100)
    n_features = st.sidebar.slider("Number of Features", 2, 20, 10, 1)
    
    # Mechanism selection
    st.sidebar.markdown("### DP Mechanism")
    mechanism = st.sidebar.selectbox(
        "Select Mechanism",
        ["laplace", "gaussian", "exponential", "histogram", "quantile"],
        help="Choose the differential privacy mechanism to demonstrate"
    )
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Generation", "🔒 DP Mechanisms", "🤖 DP-SGD", "📈 Evaluation", "🛡️ Privacy Defenses"
    ])
    
    with tab1:
        show_data_generation_tab(n_samples, n_features)
    
    with tab2:
        show_dp_mechanisms_tab(mechanism, epsilon, delta, n_samples)
    
    with tab3:
        show_dp_sgd_tab(epsilon, delta, n_samples, n_features)
    
    with tab4:
        show_evaluation_tab(epsilon, delta, n_samples)
    
    with tab5:
        show_privacy_defenses_tab(n_samples, n_features)

def show_data_generation_tab(n_samples, n_features):
    """Show data generation tab."""
    st.markdown('<h2 class="section-header">📊 Synthetic Data Generation</h2>', unsafe_allow_html=True)
    
    # Initialize data generator
    data_generator = DataGenerator(random_seed=42)
    
    # Generate different types of data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Classification Data")
        X_class, y_class = data_generator.generate_synthetic_classification(
            n_samples=n_samples, n_features=n_features
        )
        
        # Display data info
        st.write(f"**Shape:** {X_class.shape}")
        st.write(f"**Classes:** {len(np.unique(y_class))}")
        
        # Plot class distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        unique, counts = np.unique(y_class, return_counts=True)
        ax.bar(unique, counts)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Regression Data")
        X_reg, y_reg = data_generator.generate_synthetic_regression(
            n_samples=n_samples, n_features=n_features
        )
        
        # Display data info
        st.write(f"**Shape:** {X_reg.shape}")
        st.write(f"**Target Range:** [{y_reg.min():.2f}, {y_reg.max():.2f}]")
        
        # Plot target distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_reg, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Target Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Target Distribution')
        st.pyplot(fig)
    
    # Generate income data
    st.markdown("### Income Data (Synthetic)")
    income_data = data_generator.generate_income_data(n_samples=n_samples)
    
    # Display sample data
    st.write("**Sample Data:**")
    st.dataframe(income_data.head(10))
    
    # Plot income distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(income_data['income'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Income')
    ax.set_ylabel('Frequency')
    ax.set_title('Income Distribution')
    st.pyplot(fig)
    
    # Data summary
    st.markdown("### Data Summary")
    summary = data_generator.get_data_summary(income_data)
    st.json(summary)

def show_dp_mechanisms_tab(mechanism, epsilon, delta, n_samples):
    """Show DP mechanisms tab."""
    st.markdown('<h2 class="section-header">🔒 Differential Privacy Mechanisms</h2>', unsafe_allow_html=True)
    
    # Initialize DP mechanisms
    dp_mechanisms = DPMechanisms(random_seed=42)
    
    # Generate sample data
    data_generator = DataGenerator(random_seed=42)
    X, y = data_generator.generate_synthetic_classification(n_samples=n_samples, n_features=5)
    data = X.flatten()  # Flatten for demonstration
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Mechanism Parameters")
        st.write(f"**Mechanism:** {mechanism}")
        st.write(f"**ε (Epsilon):** {epsilon}")
        st.write(f"**δ (Delta):** {delta}")
        st.write(f"**Data Size:** {len(data)}")
        st.write(f"**Data Range:** [{data.min():.2f}, {data.max():.2f}]")
    
    with col2:
        st.markdown("### Privacy Budget")
        privacy_accountant = PrivacyAccountant(delta)
        privacy_accountant.add_mechanism(epsilon, delta)
        privacy_report = privacy_accountant.get_privacy_report()
        
        st.metric("ε Used", f"{privacy_report['basic_composition_epsilon']:.4f}")
        st.metric("δ Used", f"{privacy_report['basic_composition_delta']:.2e}")
        st.metric("Mechanisms", privacy_report['mechanism_count'])
    
    # Apply selected mechanism
    st.markdown("### Mechanism Results")
    
    if mechanism == "laplace":
        result, metadata = dp_mechanisms.laplace_mechanism(data, epsilon, "mean", delta)
        st.write(f"**True Mean:** {metadata['true_result']:.4f}")
        st.write(f"**Private Mean:** {result:.4f}")
        st.write(f"**Noise Added:** {metadata['noise']:.4f}")
        st.write(f"**Sensitivity:** {metadata['sensitivity']:.4f}")
        
    elif mechanism == "gaussian":
        result, metadata = dp_mechanisms.gaussian_mechanism(data, epsilon, delta, "mean")
        st.write(f"**True Mean:** {metadata['true_result']:.4f}")
        st.write(f"**Private Mean:** {result:.4f}")
        st.write(f"**Noise Added:** {metadata['noise']:.4f}")
        st.write(f"**Sensitivity:** {metadata['sensitivity']:.4f}")
        
    elif mechanism == "exponential":
        candidates = ["Option A", "Option B", "Option C", "Option D"]
        scores = np.array([0.8, 0.6, 0.9, 0.7])
        result, metadata = dp_mechanisms.exponential_mechanism(candidates, scores, epsilon)
        st.write(f"**Selected Candidate:** {result}")
        st.write(f"**Candidates:** {candidates}")
        st.write(f"**Scores:** {scores}")
        st.write(f"**Probabilities:** {metadata['probabilities']}")
        
    elif mechanism == "histogram":
        result, metadata = dp_mechanisms.histogram_mechanism(data, bins=10, epsilon=epsilon)
        st.write(f"**Private Histogram:** {result}")
        st.write(f"**True Histogram:** {metadata['true_histogram']}")
        st.write(f"**Noise Added:** {metadata['noise']}")
        
    elif mechanism == "quantile":
        result, metadata = dp_mechanisms.quantile_mechanism(data, quantile=0.5, epsilon=epsilon, delta=delta)
        st.write(f"**True Median:** {metadata['true_quantile']:.4f}")
        st.write(f"**Private Median:** {result:.4f}")
        st.write(f"**Noise Added:** {metadata['noise']:.4f}")
    
    # Visualization
    st.markdown("### Visualization")
    
    if mechanism in ["laplace", "gaussian"]:
        # Plot true vs private result
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['True Result', 'Private Result'], [metadata['true_result'], result])
        ax.set_ylabel('Value')
        ax.set_title(f'{mechanism.title()} Mechanism Results')
        st.pyplot(fig)
        
        # Plot noise distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        noise_samples = np.random.laplace(0, metadata['noise_scale'], 1000)
        ax.hist(noise_samples, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(metadata['noise'], color='red', linestyle='--', label=f'Actual Noise: {metadata["noise"]:.4f}')
        ax.set_xlabel('Noise')
        ax.set_ylabel('Frequency')
        ax.set_title('Noise Distribution')
        ax.legend()
        st.pyplot(fig)

def show_dp_sgd_tab(epsilon, delta, n_samples, n_features):
    """Show DP-SGD tab."""
    st.markdown('<h2 class="section-header">🤖 Differential Privacy Stochastic Gradient Descent</h2>', unsafe_allow_html=True)
    
    # Generate data
    data_generator = DataGenerator(random_seed=42)
    X, y = data_generator.generate_synthetic_classification(n_samples=n_samples, n_features=n_features)
    
    # Split data
    X_train, X_test, y_train, y_test = data_generator.create_train_test_split(X, y, test_size=0.2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        st.write(f"**Training Samples:** {len(X_train)}")
        st.write(f"**Test Samples:** {len(X_test)}")
        st.write(f"**Features:** {n_features}")
        st.write(f"**Classes:** {len(np.unique(y))}")
        
        # Model parameters
        noise_multiplier = st.slider("Noise Multiplier", 0.5, 3.0, 1.1, 0.1)
        max_grad_norm = st.slider("Max Gradient Norm", 0.5, 2.0, 1.0, 0.1)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        epochs = st.slider("Epochs", 10, 100, 50, 10)
    
    with col2:
        st.markdown("### Privacy Configuration")
        st.write(f"**ε (Epsilon):** {epsilon}")
        st.write(f"**δ (Delta):** {delta}")
        st.write(f"**Noise Multiplier:** {noise_multiplier}")
        st.write(f"**Max Gradient Norm:** {max_grad_norm}")
    
    # Train model
    if st.button("Train DP-SGD Model"):
        with st.spinner("Training model..."):
            # Create model
            model = SimpleDPModel(input_dim=n_features, hidden_dim=128, num_classes=len(np.unique(y)))
            
            # Initialize DP-SGD
            dp_sgd = DPSGD(
                model=model,
                epsilon=epsilon,
                delta=delta,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                learning_rate=learning_rate
            )
            
            # Convert to PyTorch tensors
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Train model
            history = dp_sgd.train(train_loader, test_loader, epochs=epochs, verbose=False)
            
            # Get privacy report
            privacy_report = dp_sgd.get_privacy_report()
            
            # Display results
            st.markdown("### Training Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Train Accuracy", f"{history['train_accuracy'][-1]:.4f}")
            with col2:
                st.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.4f}")
            with col3:
                st.metric("Privacy Budget Used", f"{privacy_report['epsilon_used']:.4f}")
            
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Loss plot
            ax1.plot(history['train_loss'], label='Train Loss')
            ax1.plot(history['val_loss'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy plot
            ax2.plot(history['train_accuracy'], label='Train Accuracy')
            ax2.plot(history['val_accuracy'], label='Val Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            st.pyplot(fig)
            
            # Privacy report
            st.markdown("### Privacy Report")
            st.json(privacy_report)

def show_evaluation_tab(epsilon, delta, n_samples):
    """Show evaluation tab."""
    st.markdown('<h2 class="section-header">📈 Privacy and Utility Evaluation</h2>', unsafe_allow_html=True)
    
    # Initialize evaluators
    privacy_evaluator = PrivacyEvaluator(random_seed=42)
    utility_evaluator = UtilityEvaluator(random_seed=42)
    
    # Generate sample data
    data_generator = DataGenerator(random_seed=42)
    X, y = data_generator.generate_synthetic_classification(n_samples=n_samples, n_features=5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Privacy Evaluation")
        
        # Privacy budget evaluation
        budget_eval = privacy_evaluator.evaluate_privacy_budget(epsilon, delta, "laplace")
        st.write("**Privacy Budget:**")
        st.write(f"- ε Used: {budget_eval['epsilon_used']:.4f}")
        st.write(f"- δ Used: {budget_eval['delta_used']:.2e}")
        st.write(f"- Budget Remaining: {budget_eval['privacy_budget_remaining']:.4f}")
        
        # Privacy-utility tradeoff
        privacy_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        utility_scores = [0.6, 0.7, 0.8, 0.85, 0.9]  # Simulated
        
        tradeoff_eval = privacy_evaluator.evaluate_privacy_utility_tradeoff(
            privacy_levels, utility_scores, "epsilon"
        )
        
        st.write("**Privacy-Utility Tradeoff:**")
        st.write(f"- Correlation: {tradeoff_eval['correlation']:.4f}")
        st.write(f"- Optimal ε: {tradeoff_eval['optimal_privacy_level']:.4f}")
        st.write(f"- Optimal Utility: {tradeoff_eval['optimal_utility']:.4f}")
    
    with col2:
        st.markdown("### Utility Evaluation")
        
        # Simulate utility results
        true_statistics = {"mean": np.mean(X), "std": np.std(X), "min": np.min(X), "max": np.max(X)}
        private_statistics = {
            "mean": np.mean(X) + np.random.normal(0, 0.1),
            "std": np.std(X) + np.random.normal(0, 0.05),
            "min": np.min(X) + np.random.normal(0, 0.1),
            "max": np.max(X) + np.random.normal(0, 0.1)
        }
        
        utility_metrics = utility_evaluator.evaluate_statistical_utility(true_statistics, private_statistics)
        
        st.write("**Statistical Utility:**")
        for metric, value in utility_metrics.items():
            st.write(f"- {metric}: {value:.4f}")
    
    # Visualization
    st.markdown("### Privacy-Utility Tradeoff Visualization")
    
    # Create interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=privacy_levels,
        y=utility_scores,
        mode='lines+markers',
        name='Privacy-Utility Tradeoff',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color='blue')
    ))
    
    # Add optimal point
    optimal_idx = np.argmax(utility_scores)
    fig.add_trace(go.Scatter(
        x=[privacy_levels[optimal_idx]],
        y=[utility_scores[optimal_idx]],
        mode='markers',
        name='Optimal Point',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title='Privacy-Utility Tradeoff',
        xaxis_title='Privacy Level (ε)',
        yaxis_title='Utility Score',
        hovermode='closest',
        showlegend=True,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mechanism comparison
    st.markdown("### Mechanism Comparison")
    
    mechanisms = ["Laplace", "Gaussian", "Exponential"]
    mechanism_utilities = [0.75, 0.80, 0.70]
    mechanism_privacy = [epsilon, epsilon, epsilon]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(mechanisms, mechanism_utilities, color=['blue', 'green', 'orange'])
    ax.set_ylabel('Utility Score')
    ax.set_title('Mechanism Utility Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, utility in zip(bars, mechanism_utilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{utility:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig)

def show_privacy_defenses_tab(n_samples, n_features):
    """Show privacy defenses tab."""
    st.markdown('<h2 class="section-header">🛡️ Privacy Defenses</h2>', unsafe_allow_html=True)
    
    # Initialize privacy defense
    privacy_defense = PrivacyDefense(random_seed=42)
    
    # Generate sample data
    data_generator = DataGenerator(random_seed=42)
    income_data = data_generator.generate_income_data(n_samples=n_samples)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Sanitization")
        
        sanitization_method = st.selectbox(
            "Sanitization Method",
            ["clipping", "normalization"],
            help="Choose the input sanitization method"
        )
        
        clip_value = st.slider("Clip Value", 0.1, 5.0, 1.0, 0.1)
        
        # Apply sanitization
        sanitized_data = privacy_defense.apply_input_sanitization(
            income_data, sanitization_method, clip_value
        )
        
        st.write("**Original Data Statistics:**")
        st.write(income_data.describe())
        
        st.write("**Sanitized Data Statistics:**")
        st.write(sanitized_data.describe())
    
    with col2:
        st.markdown("### K-Anonymity")
        
        k_value = st.slider("K Value", 2, 10, 5)
        quasi_identifiers = st.multiselect(
            "Quasi-Identifiers",
            ["age", "education", "zip_code"],
            default=["age", "education"]
        )
        
        if quasi_identifiers:
            # Apply k-anonymity
            k_anonymous_data = privacy_defense.apply_k_anonymity(
                income_data, quasi_identifiers, k_value
            )
            
            st.write(f"**Original Records:** {len(income_data)}")
            st.write(f"**K-Anonymous Records:** {len(k_anonymous_data)}")
            st.write(f"**Records Removed:** {len(income_data) - len(k_anonymous_data)}")
            
            # Show sample of k-anonymous data
            if len(k_anonymous_data) > 0:
                st.write("**Sample K-Anonymous Data:**")
                st.dataframe(k_anonymous_data.head())
    
    # Synthetic data generation
    st.markdown("### Synthetic Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        synthetic_method = st.selectbox(
            "Synthetic Data Method",
            ["gaussian", "sampling"],
            help="Choose the synthetic data generation method"
        )
        
        noise_level = st.slider("Noise Level", 0.01, 0.5, 0.1, 0.01)
        
        # Generate synthetic data
        synthetic_data = privacy_defense.apply_synthetic_data_generation(
            income_data, synthetic_method, noise_level
        )
        
        st.write("**Synthetic Data Statistics:**")
        st.write(synthetic_data.describe())
    
    with col2:
        # Compare original vs synthetic
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original income distribution
        ax1.hist(income_data['income'], bins=30, alpha=0.7, edgecolor='black', color='blue')
        ax1.set_xlabel('Income')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Original Income Distribution')
        
        # Synthetic income distribution
        ax2.hist(synthetic_data['income'], bins=30, alpha=0.7, edgecolor='black', color='green')
        ax2.set_xlabel('Income')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Synthetic Income Distribution')
        
        st.pyplot(fig)
    
    # Defense effectiveness
    st.markdown("### Defense Effectiveness")
    
    # Simulate attack results
    attack_results = {
        "attack_success_rate": 0.8,
        "privacy_loss": 0.6
    }
    
    # Evaluate defense effectiveness
    effectiveness = privacy_defense.evaluate_defense_effectiveness(
        income_data, synthetic_data, attack_results
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{effectiveness.get('mse', 0):.4f}")
    with col2:
        st.metric("MAE", f"{effectiveness.get('mae', 0):.4f}")
    with col3:
        st.metric("Utility Preservation", f"{effectiveness.get('utility_preservation', 0):.4f}")
    
    # Privacy gain
    if 'privacy_gain' in effectiveness:
        st.metric("Privacy Gain", f"{effectiveness['privacy_gain']:.4f}")
    
    if 'attack_success_reduction' in effectiveness:
        st.metric("Attack Success Reduction", f"{effectiveness['attack_success_reduction']:.4f}")

if __name__ == "__main__":
    main()
