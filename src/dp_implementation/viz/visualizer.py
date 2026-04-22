"""Visualization utilities for differential privacy experiments."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns


class PrivacyVisualizer:
    """Visualization utilities for differential privacy experiments.
    
    This class provides methods for creating various visualizations
    to analyze and present differential privacy results.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 6)) -> None:
        """Initialize privacy visualizer.
        
        Args:
            style: Matplotlib style to use.
            figsize: Default figure size.
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
    
    def plot_privacy_utility_tradeoff(
        self,
        privacy_levels: List[float],
        utility_scores: List[float],
        privacy_metric: str = "epsilon",
        title: str = "Privacy-Utility Tradeoff",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot privacy-utility tradeoff curve.
        
        Args:
            privacy_levels: List of privacy parameter values.
            utility_scores: List of corresponding utility scores.
            privacy_metric: Privacy metric to use ("epsilon", "delta").
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(privacy_levels, utility_scores, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel(f'Privacy Level ({privacy_metric})')
        ax.set_ylabel('Utility Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for extreme points
        max_utility_idx = np.argmax(utility_scores)
        ax.annotate(f'Max Utility: {utility_scores[max_utility_idx]:.3f}',
                   xy=(privacy_levels[max_utility_idx], utility_scores[max_utility_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_noise_analysis(
        self,
        true_values: List[float],
        noisy_values: List[float],
        noise_levels: List[float],
        title: str = "Noise Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot noise analysis.
        
        Args:
            true_values: List of true values.
            noisy_values: List of noisy values.
            noise_levels: List of noise levels.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: True vs Noisy values
        ax1.scatter(true_values, noisy_values, alpha=0.6, c=noise_levels, cmap='viridis')
        ax1.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', alpha=0.8)
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Noisy Values')
        ax1.set_title('True vs Noisy Values')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Noise Level')
        
        # Plot 2: Noise distribution
        noise = np.array(noisy_values) - np.array(true_values)
        ax2.hist(noise, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(noise), color='red', linestyle='--', label=f'Mean: {np.mean(noise):.3f}')
        ax2.axvline(np.std(noise), color='orange', linestyle='--', label=f'Std: {np.std(noise):.3f}')
        ax2.set_xlabel('Noise')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Noise Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_mechanism_comparison(
        self,
        mechanism_results: Dict[str, Dict[str, List[float]]],
        title: str = "Mechanism Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of different mechanisms.
        
        Args:
            mechanism_results: Dictionary with mechanism results.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (mechanism_name, results) in enumerate(mechanism_results.items()):
            if 'privacy_levels' in results and 'utility_scores' in results:
                ax.plot(results['privacy_levels'], results['utility_scores'],
                       'o-', label=mechanism_name, color=colors[i % len(colors)],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Privacy Level (ε)')
        ax.set_ylabel('Utility Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_privacy_budget_consumption(
        self,
        budget_history: List[Dict[str, float]],
        title: str = "Privacy Budget Consumption",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot privacy budget consumption over time.
        
        Args:
            budget_history: List of budget consumption records.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        steps = list(range(len(budget_history)))
        epsilon_used = [record.get('epsilon_used', 0) for record in budget_history]
        delta_used = [record.get('delta_used', 0) for record in budget_history]
        
        ax.plot(steps, epsilon_used, 'b-o', label='ε used', linewidth=2, markersize=4)
        ax2 = ax.twinx()
        ax2.plot(steps, delta_used, 'r-s', label='δ used', linewidth=2, markersize=4)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('ε used', color='blue')
        ax2.set_ylabel('δ used', color='red')
        ax.set_title(title)
        
        # Add budget limit lines
        if budget_history:
            max_epsilon = max(record.get('epsilon_total', 1.0) for record in budget_history)
            max_delta = max(record.get('delta_total', 1e-5) for record in budget_history)
            ax.axhline(y=max_epsilon, color='blue', linestyle='--', alpha=0.7, label='ε limit')
            ax2.axhline(y=max_delta, color='red', linestyle='--', alpha=0.7, label='δ limit')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_utility_distribution(
        self,
        utility_scores: List[float],
        title: str = "Utility Score Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot distribution of utility scores.
        
        Args:
            utility_scores: List of utility scores.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(utility_scores, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(utility_scores), color='red', linestyle='--', label=f'Mean: {np.mean(utility_scores):.3f}')
        ax1.axvline(np.median(utility_scores), color='green', linestyle='--', label=f'Median: {np.median(utility_scores):.3f}')
        ax1.set_xlabel('Utility Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Utility Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(utility_scores, vert=True)
        ax2.set_ylabel('Utility Score')
        ax2.set_title('Utility Score Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_privacy_utility(
        self,
        privacy_levels: List[float],
        utility_scores: List[float],
        privacy_metric: str = "epsilon",
        title: str = "Interactive Privacy-Utility Tradeoff"
    ) -> go.Figure:
        """Create interactive privacy-utility tradeoff plot.
        
        Args:
            privacy_levels: List of privacy parameter values.
            utility_scores: List of corresponding utility scores.
            privacy_metric: Privacy metric to use ("epsilon", "delta").
            title: Plot title.
            
        Returns:
            Plotly figure.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=privacy_levels,
            y=utility_scores,
            mode='lines+markers',
            name='Privacy-Utility Tradeoff',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            hovertemplate=f'<b>{privacy_metric}</b>: %{{x}}<br>Utility: %{{y}}<extra></extra>'
        ))
        
        # Add optimal point
        max_utility_idx = np.argmax(utility_scores)
        fig.add_trace(go.Scatter(
            x=[privacy_levels[max_utility_idx]],
            y=[utility_scores[max_utility_idx]],
            mode='markers',
            name='Optimal Point',
            marker=dict(size=12, color='red', symbol='star'),
            hovertemplate=f'<b>Optimal Point</b><br>{privacy_metric}: %{{x}}<br>Utility: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=f'Privacy Level ({privacy_metric})',
            yaxis_title='Utility Score',
            hovermode='closest',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_interactive_mechanism_comparison(
        self,
        mechanism_results: Dict[str, Dict[str, List[float]]],
        title: str = "Interactive Mechanism Comparison"
    ) -> go.Figure:
        """Create interactive mechanism comparison plot.
        
        Args:
            mechanism_results: Dictionary with mechanism results.
            title: Plot title.
            
        Returns:
            Plotly figure.
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (mechanism_name, results) in enumerate(mechanism_results.items()):
            if 'privacy_levels' in results and 'utility_scores' in results:
                fig.add_trace(go.Scatter(
                    x=results['privacy_levels'],
                    y=results['utility_scores'],
                    mode='lines+markers',
                    name=mechanism_name,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8, color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{mechanism_name}</b><br>ε: %{{x}}<br>Utility: %{{y}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Privacy Level (ε)',
            yaxis_title='Utility Score',
            hovermode='closest',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str = "Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot heatmap.
        
        Args:
            data: 2D array of data.
            x_labels: Labels for x-axis.
            y_labels: Labels for y-axis.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=x_labels, yticklabels=y_labels, ax=ax)
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_labels: List[str],
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array.
            class_labels: List of class labels.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive dashboard.
        
        Args:
            results: Dictionary with experiment results.
            save_path: Path to save the dashboard.
            
        Returns:
            Plotly figure with subplots.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Privacy-Utility Tradeoff', 'Noise Analysis',
                          'Mechanism Comparison', 'Budget Consumption'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add plots based on available data
        if 'privacy_utility' in results:
            privacy_levels = results['privacy_utility']['privacy_levels']
            utility_scores = results['privacy_utility']['utility_scores']
            fig.add_trace(
                go.Scatter(x=privacy_levels, y=utility_scores, mode='lines+markers', name='Tradeoff'),
                row=1, col=1
            )
        
        if 'noise_analysis' in results:
            true_values = results['noise_analysis']['true_values']
            noisy_values = results['noise_analysis']['noisy_values']
            fig.add_trace(
                go.Scatter(x=true_values, y=noisy_values, mode='markers', name='Noise'),
                row=1, col=2
            )
        
        if 'mechanism_comparison' in results:
            for mechanism_name, mechanism_data in results['mechanism_comparison'].items():
                fig.add_trace(
                    go.Scatter(x=mechanism_data['privacy_levels'], 
                             y=mechanism_data['utility_scores'],
                             mode='lines+markers', name=mechanism_name),
                    row=2, col=1
                )
        
        if 'budget_consumption' in results:
            steps = results['budget_consumption']['steps']
            epsilon_used = results['budget_consumption']['epsilon_used']
            fig.add_trace(
                go.Scatter(x=steps, y=epsilon_used, mode='lines+markers', name='ε used'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Differential Privacy Experiment Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
